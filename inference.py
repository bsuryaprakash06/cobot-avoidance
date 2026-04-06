"""
inference.py — OpenEnv Hackathon Round 1 submission script.

Environment: Industrial Robot Arm Collision Avoidance
Uses an LLM (via OpenAI-compatible client) to control the cobot each step.

Required environment variables:
    API_BASE_URL   e.g. https://api.openai.com/v1
    MODEL_NAME     e.g. gpt-4o-mini
    HF_TOKEN       your API / Hugging Face key

Structured stdout log format (mandatory — do not alter field names or order):
    [START] {"task_id": ..., "episode": ...}
    [STEP]  {"step": ..., "action": ..., "reward": ..., "total_reward": ..., "done": ...}
    [END]   {"task_id": ..., "score": ..., "total_reward": ..., "steps": ..., "success": ...}
"""

from __future__ import annotations

import json
import math
import os
import time

import requests
from openai import OpenAI

# Ensure project root is on path when run directly
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------ #
#  Configuration
# ------------------------------------------------------------------ #
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ------------------------------------------------------------------ #
#  Environment HTTP helpers
# ------------------------------------------------------------------ #

def env_reset(seed: int = 42) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/reset", params={"seed": seed}, timeout=10)
    r.raise_for_status()
    return r.json()


def env_step(delta_angle: float, send_halt: bool) -> dict:
    r = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"delta_angle": delta_angle, "send_halt": send_halt},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def env_get_tasks() -> list[dict]:
    r = requests.get(f"{ENV_BASE_URL}/tasks", timeout=10)
    r.raise_for_status()
    return r.json()


def env_run_task(task_id: str, seed: int = 42) -> dict:
    r = requests.post(
        f"{ENV_BASE_URL}/tasks/{task_id}/run",
        params={"seed": seed},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()

# ------------------------------------------------------------------ #
#  LLM action generation
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """You are an AI safety controller for a collaborative robot arm (cobot) on a shared factory floor.

SETUP:
- Cobot base is fixed at (2, 10). Field is 20×20 metres.
- Obstacle arm base is at (18, 10). Both arms have 3 integrity points.
- `cobot.angle` is the end-effector direction in degrees (0=right, 90=up). Arm length = 3 m.
- `cobot.tip_x / tip_y` is the current end-effector position.

SAFETY PROTOCOL (ISO 10218 / ISO 15066):
1. When `obstacle.is_broadcasting=true`, the obstacle has announced `next_trajectory_angle`.
   You MUST replan the end-effector OUT of that trajectory path BEFORE it launches.
   Use `delta_angle` to rotate away from the broadcasted trajectory direction.
2. When `incoming_trajectories` contains active entries, check their (x, y, vx, vy).
   Rotate away from the trajectory with the smallest perpendicular distance to the tip.
3. When no active threats exist, aim at the obstacle (towards x=18, y=10) and set send_halt=true
   to dispatch a halt signal and reduce its operational cycles.

OUTPUT: Return ONLY a JSON object — no explanation, no markdown:
{"delta_angle": <float between -15.0 and 15.0>, "send_halt": <true or false>}"""


def _state_summary(state: dict) -> str:
    cobot = state["cobot"]
    obstacle = state["obstacle"]
    trajectories = [
        {"x": round(t["x"], 2), "y": round(t["y"], 2),
         "vx": round(t["vx"], 2), "vy": round(t["vy"], 2)}
        for t in state.get("incoming_trajectories", [])
        if t.get("active", True)
    ]
    return json.dumps({
        "cobot_angle": round(cobot["angle"], 1),
        "cobot_tip": [round(cobot["tip_x"], 2), round(cobot["tip_y"], 2)],
        "cobot_integrity": cobot["integrity"],
        "obstacle_integrity": obstacle["integrity"],
        "obstacle_broadcasting": obstacle["is_broadcasting"],
        "obstacle_next_trajectory_angle": obstacle.get("next_trajectory_angle"),
        "obstacle_charge_remaining": obstacle["charge_remaining"],
        "active_trajectories": trajectories,
        "last_reward": state.get("last_reward", 0),
        "last_events": state.get("last_events", []),
    })


def get_llm_action(state: dict, retries: int = 2) -> dict:
    prompt = _state_summary(state)
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Current state:\n{prompt}\n\nOutput action JSON:"},
                ],
                temperature=0.2,
                max_tokens=64,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            action = json.loads(raw)
            action["delta_angle"] = max(-15.0, min(15.0, float(action.get("delta_angle", 0))))
            action["send_halt"] = bool(action.get("send_halt", False))
            return action
        except Exception:
            if attempt == retries:
                return _fallback_action(state)
            time.sleep(0.5)
    return _fallback_action(state)


def _fallback_action(state: dict) -> dict:
    """Geometry-based fallback used when the LLM call fails."""
    cobot = state["cobot"]
    trajectories = [t for t in state.get("incoming_trajectories", []) if t.get("active", True)]
    obstacle = state["obstacle"]

    if not trajectories:
        # Aim toward obstacle and halt
        dx = obstacle["x"] - cobot["x"]
        dy = obstacle["y"] - cobot["y"]
        target = math.degrees(math.atan2(dy, dx))
        diff = (target - cobot["angle"] + 180) % 360 - 180
        delta = max(-15.0, min(15.0, diff * 0.5))
        return {"delta_angle": delta, "send_halt": abs(diff) < 12.0}

    def ray_dist(t):
        mag = math.hypot(t["vx"], t["vy"]) or 1e-9
        ux, uy = t["vx"] / mag, t["vy"] / mag
        tx, ty = cobot["tip_x"], cobot["tip_y"]
        proj = (tx - t["x"]) * ux + (ty - t["y"]) * uy
        if proj < 0:
            return math.hypot(tx - t["x"], ty - t["y"])
        return math.hypot(tx - (t["x"] + proj * ux), ty - (t["y"] + proj * uy))

    worst = min(trajectories, key=ray_dist)
    if ray_dist(worst) > 3.0:
        return {"delta_angle": 0.0, "send_halt": False}

    def tip_clear(delta):
        rad = math.radians(cobot["angle"] + delta)
        tx = cobot["x"] + 3.0 * math.cos(rad)
        ty = cobot["y"] + 3.0 * math.sin(rad)
        mag = math.hypot(worst["vx"], worst["vy"]) or 1e-9
        ux, uy = worst["vx"] / mag, worst["vy"] / mag
        proj = (tx - worst["x"]) * ux + (ty - worst["y"]) * uy
        if proj < 0:
            return math.hypot(tx - worst["x"], ty - worst["y"])
        return math.hypot(tx - (worst["x"] + proj * ux), ty - (worst["y"] + proj * uy))

    delta = 10.0 if tip_clear(10.0) > tip_clear(-10.0) else -10.0
    return {"delta_angle": delta, "send_halt": False}

# ------------------------------------------------------------------ #
#  Episode runner
# ------------------------------------------------------------------ #

def run_episode(task_id: str, episode: int, seed: int, max_steps: int = 100) -> dict:
    reset_data = env_reset(seed=seed + episode)
    state = reset_data["state"]

    print(json.dumps({
        "tag": "[START]",
        "task_id": task_id,
        "episode": episode,
        "episode_id": reset_data["episode_id"],
    }), flush=True)

    total_reward = 0.0
    steps = 0

    for step_num in range(1, max_steps + 1):
        action = get_llm_action(state)
        result = env_step(action["delta_angle"], action["send_halt"])

        reward = result["reward"]
        total_reward += reward
        state = result["state"]
        done = result["done"]
        steps = step_num

        print(json.dumps({
            "tag": "[STEP]",
            "step": step_num,
            "action": action,
            "reward": round(reward, 4),
            "total_reward": round(total_reward, 4),
            "done": done,
            "events": result.get("info", {}).get("events", []),
        }), flush=True)

        if done:
            break

    return {"steps": steps, "total_reward": round(total_reward, 4)}

# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    tasks_info = env_get_tasks()
    task_ids = [t["task_id"] for t in tasks_info]
    all_scores = []

    for i, task_id in enumerate(task_ids):
        ep = run_episode(task_id=task_id, episode=i, seed=42 + i)
        grade = env_run_task(task_id, seed=42 + i)

        print(json.dumps({
            "tag": "[END]",
            "task_id": task_id,
            "score": grade["score"],
            "total_reward": ep["total_reward"],
            "steps": ep["steps"],
            "success": grade["success"],
            "details": grade.get("details", {}),
        }), flush=True)

        all_scores.append(grade["score"])

    mean_score = round(sum(all_scores) / len(all_scores), 4)
    print(json.dumps({
        "tag": "[SUMMARY]",
        "mean_score": mean_score,
        "task_scores": dict(zip(task_ids, all_scores)),
    }), flush=True)


if __name__ == "__main__":
    main()
