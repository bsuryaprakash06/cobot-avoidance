# Industrial Robot Arm Collision Avoidance

An OpenEnv-compliant reinforcement learning environment that simulates **collaborative robot arm (cobot) safety planning** on a shared factory floor.

## Motivation

Modern manufacturing lines operate with multiple robot arms sharing the same physical workspace (shared-cell robotics). Unplanned collisions cause production halts, equipment damage, and safety incidents. Standards like **ISO 10218** and **ISO 15066** mandate that cobots broadcast their planned trajectories before executing them, giving adjacent arms a time window to replan.

This environment trains an RL agent to act as that replanning controller — reading broadcasted trajectory data, repositioning the end-effector out of collision zones *before* the obstacle arm launches, and dispatching halt signals when safe. The same problem is solved in automotive plants (BMW, Toyota) and electronics assembly (Foxconn, Flex).

---

## Environment Description

| Property | Value |
|---|---|
| Field size | 20 × 20 metres |
| Cobot base | (2, 10) — fixed |
| Obstacle arm base | (18, 10) — fixed |
| End-effector arm length | 3 m |
| Cobot integrity points | 3 |
| Obstacle integrity points | 3 |
| Max steps / episode | 100 |
| Obstacle trajectory speed | 1.5 m/step |
| Halt signal speed | 2.0 m/step |
| Obstacle launch interval | ~8 steps (±2 randomised) |
| Broadcast window | 3 steps before launch |

### Key mechanic — predictive replanning

Three steps before launching a trajectory, the obstacle arm **broadcasts** its planned angle via `obstacle.is_broadcasting=true` and `obstacle.next_trajectory_angle`. A well-trained agent reads this signal and rotates the end-effector out of the projected path *before the trajectory is launched*, achieving zero-contact safety. A reactive agent that waits until the trajectory is airborne will incur more collisions.

---

## Action Space

```json
{
  "delta_angle": -15.0,   // float [-15, 15] — rotate end-effector this many degrees
  "send_halt":  false     // bool — dispatch a halt signal in the current end-effector direction
}
```

## Observation Space

Key fields returned in every `state`:

| Field | Type | Description |
|---|---|---|
| `cobot.angle` | float | Current end-effector direction (degrees) |
| `cobot.tip_x/y` | float | End-effector position (metres) |
| `cobot.integrity` | int | 0–3; 0 = shutdown |
| `obstacle.is_broadcasting` | bool | True during ISO 10218 broadcast window |
| `obstacle.next_trajectory_angle` | float\|null | Broadcasted trajectory direction |
| `obstacle.charge_remaining` | int | Steps until trajectory launches |
| `incoming_trajectories[].x/y/vx/vy` | float | Active obstacle trajectories |

## Reward Function

| Event | Reward |
|---|---|
| End-effector > 1.5 m from trajectory ray | +0.3 per trajectory |
| Halt signal reaches obstacle arm | +2.0 |
| Obstacle fully halted (0 integrity) | +3.0 bonus |
| End-effector hit by obstacle trajectory | -1.0 |
| Cobot integrity reaches 0 | -2.0 |

Rewards provide a **dense signal** across the full trajectory — the agent receives feedback every step for maintaining safe clearance, not just at episode end.

---

## Tasks

| Task ID | Difficulty | Description | Score |
|---|---|---|---|
| `task_1_zero_collision` | Easy | Operate for 100 steps without any collision | `(integrity/3) × (steps/100)` |
| `task_2_predictive_replan` | Medium | Survive + pre-launch clearance ≥ 80% | `50% integrity + 50% pre-launch rate` |
| `task_3_halt_dispatch` | Hard | Survive + halt the obstacle arm | `40% integrity + 60% obstacle damage` |

---

## Setup & Run

### Local

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t industrial-cobot-avoidance .
docker run -p 7860:7860 industrial-cobot-avoidance
```

### Inference script

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-key-here"
export ENV_BASE_URL="http://localhost:7860"

python inference.py
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start new episode (`?seed=42`) |
| `POST` | `/step` | Apply action, advance one step |
| `GET` | `/state` | Read current state |
| `GET` | `/tasks` | List all tasks |
| `POST` | `/tasks/{id}/run` | Run task with baseline policy |
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |

---

## Stdout Log Format

```
[START] {"tag": "[START]", "task_id": "task_1_zero_collision", "episode": 0, ...}
[STEP]  {"tag": "[STEP]", "step": 1, "action": {...}, "reward": 0.3, "total_reward": 0.3, "done": false, "events": [...]}
...
[END]   {"tag": "[END]", "task_id": "task_1_zero_collision", "score": 0.91, "total_reward": 18.2, "steps": 100, "success": true}
[SUMMARY] {"tag": "[SUMMARY]", "mean_score": 0.74, "task_scores": {...}}
```

---

## Baseline Scores (heuristic policy)

| Task | Expected Score |
|---|---|
| task_1_zero_collision | ~0.85–1.0 |
| task_2_predictive_replan | ~0.70–0.85 |
| task_3_halt_dispatch | ~0.55–0.75 |
