"""
tasks.py — Graded task definitions and graders.
Task 1 (Easy)   – Zero-Collision Operation
Task 2 (Medium) – Predictive Path Replanning
Task 3 (Hard)   – Collision-Free Halt Dispatch
"""

from __future__ import annotations
import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ActionInput, TaskInfo, TaskResult
from server.robot_arm_environment import CollisionAvoidanceEnv, ray_distance


TASKS: list[TaskInfo] = [
    TaskInfo(
        task_id="task_1_zero_collision",
        name="Zero-Collision Operation",
        description=(
            "Operate the cobot for 100 steps without any end-effector collision. "
            "Score = (integrity_remaining / 3) × (steps_survived / 100). "
            "Reflects ISO 10218: zero unplanned contacts in shared cells."
        ),
        difficulty="easy",
        max_steps=100,
        reward_range=[0.0, 1.0],
    ),
    TaskInfo(
        task_id="task_2_predictive_replan",
        name="Predictive Path Replanning",
        description=(
            "Survive the full episode AND demonstrate predictive replanning: "
            "clear the end-effector from the broadcasted trajectory zone BEFORE launch. "
            "Score = 50% integrity + 50% pre-launch clearance rate. "
            "Models ISO 15066 speed-and-separation monitoring."
        ),
        difficulty="medium",
        max_steps=100,
        reward_range=[0.0, 1.0],
    ),
    TaskInfo(
        task_id="task_3_halt_dispatch",
        name="Collision-Free Halt Dispatch",
        description=(
            "Avoid all collisions AND reduce obstacle arm to 0 operational cycles. "
            "Score = 40% integrity survival + 60% obstacle damage fraction. "
            "+0.1 bonus for halting obstacle before step 80. "
            "Models industrial E-stop coordination between cobots."
        ),
        difficulty="hard",
        max_steps=100,
        reward_range=[0.0, 1.0],
    ),
]


def run_task_1(policy_fn=None, seed: int = 42) -> TaskResult:
    env = CollisionAvoidanceEnv(seed=seed)
    state = env.reset().state
    steps = 0

    for _ in range(100):
        action = _to_action(policy_fn(state.model_dump()) if policy_fn else _policy_replan(state))
        step_result = env.step(action)
        state = step_result.state
        steps += 1
        if step_result.done:
            break

    collisions = 3 - state.cobot.integrity
    score = round(max(0.0, state.cobot.integrity / 3.0) * (steps / 100.0), 4)
    return TaskResult(
        task_id="task_1_zero_collision",
        score=min(1.0, score),
        reward=state.total_reward,
        steps_taken=steps,
        success=collisions == 0,
        details={"integrity_remaining": state.cobot.integrity, "collisions": collisions, "steps_survived": steps},
    )


def run_task_2(policy_fn=None, seed: int = 42) -> TaskResult:
    env = CollisionAvoidanceEnv(seed=seed)
    state = env.reset().state
    steps = 0
    pre_clears = pre_checks = 0

    for _ in range(100):
        if state.obstacle.is_broadcasting and state.obstacle.next_trajectory_angle is not None:
            pre_checks += 1
            angle = state.obstacle.next_trajectory_angle
            dx, dy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
            if ray_distance(state.cobot.tip_x, state.cobot.tip_y,
                            state.obstacle.x, state.obstacle.y, dx, dy) > 1.5:
                pre_clears += 1

        action = _to_action(policy_fn(state.model_dump()) if policy_fn else _policy_replan(state))
        step_result = env.step(action)
        state = step_result.state
        steps += 1
        if step_result.done:
            break

    collisions = 3 - state.cobot.integrity
    preplan_rate = (pre_clears / pre_checks) if pre_checks > 0 else 0.5
    score = round(0.5 * max(0.0, state.cobot.integrity / 3.0) + 0.5 * preplan_rate, 4)
    return TaskResult(
        task_id="task_2_predictive_replan",
        score=min(1.0, score),
        reward=state.total_reward,
        steps_taken=steps,
        success=collisions == 0 and preplan_rate >= 0.8,
        details={
            "integrity_remaining": state.cobot.integrity,
            "collisions": collisions,
            "pre_launch_clearance_rate": round(preplan_rate, 4),
            "pre_launch_checks": pre_checks,
        },
    )


def run_task_3(policy_fn=None, seed: int = 42) -> TaskResult:
    env = CollisionAvoidanceEnv(seed=seed)
    state = env.reset().state
    steps = 0

    for _ in range(100):
        action = _to_action(policy_fn(state.model_dump()) if policy_fn else _policy_replan_and_halt(state))
        step_result = env.step(action)
        state = step_result.state
        steps += 1
        if step_result.done:
            break

    collisions = 3 - state.cobot.integrity
    damage = 3 - state.obstacle.integrity
    score = round(0.4 * max(0.0, state.cobot.integrity / 3.0) + 0.6 * (damage / 3.0), 4)
    if state.obstacle.integrity <= 0 and steps < 80:
        score = min(1.0, score + 0.1)

    return TaskResult(
        task_id="task_3_halt_dispatch",
        score=min(1.0, score),
        reward=state.total_reward,
        steps_taken=steps,
        success=state.obstacle.integrity <= 0 and collisions == 0,
        details={
            "integrity_remaining": state.cobot.integrity,
            "obstacle_integrity_remaining": state.obstacle.integrity,
            "obstacle_cycles_reduced": damage,
            "halt_signals_sent": state.cobot.halt_signals_sent,
            "halt_signals_effective": state.cobot.halt_signals_effective,
            "accuracy": round(state.cobot.halt_signals_effective / max(1, state.cobot.halt_signals_sent), 4),
        },
    )


# ------------------------------------------------------------------ #
#  Heuristic baseline policies
# ------------------------------------------------------------------ #

def _policy_replan_and_halt(state) -> ActionInput:
    cobot = state.cobot
    obstacle = state.obstacle

    # --- Aim toward obstacle ---
    dx = obstacle.x - cobot.x
    dy = obstacle.y - cobot.y
    target = math.degrees(math.atan2(dy, dx))
    diff = (target - cobot.angle + 180) % 360 - 180
    aim = max(-15.0, min(15.0, diff))

    trajectories = [t for t in state.incoming_trajectories if t.active]

    # --- 1. VERY EARLY threat detection ---
    immediate_threat = False

    if obstacle.is_broadcasting and obstacle.next_trajectory_angle is not None:
        angle = obstacle.next_trajectory_angle
        dx_b = math.cos(math.radians(angle))
        dy_b = math.sin(math.radians(angle))
        d = ray_distance(cobot.tip_x, cobot.tip_y,
                         obstacle.x, obstacle.y, dx_b, dy_b)
        if d < 3.0:   #  increased threshold
            immediate_threat = True

    if trajectories:
        worst = min(trajectories, key=lambda t: ray_distance(
            cobot.tip_x, cobot.tip_y, t.x, t.y, t.vx, t.vy))
        d = ray_distance(cobot.tip_x, cobot.tip_y,
                         worst.x, worst.y, worst.vx, worst.vy)
        if d < 3.0:   #  increased threshold
            immediate_threat = True

    # --- 2. If threat → prioritize survival ---
    if immediate_threat:
        replan = _policy_replan(state)
        return ActionInput(
            delta_angle=replan.delta_angle,
            send_halt=False   #  DO NOT halt during danger
        )

    # --- 3. If safe → attack (halt aggressively) ---
    return ActionInput(
        delta_angle=aim,
        send_halt=True   #  ALWAYS halt when safe
    )


def _tip_clear(cobot, angle_deg, traj) -> float:
    rad = math.radians(angle_deg)
    tx = cobot.x + 3.0 * math.cos(rad)
    ty = cobot.y + 3.0 * math.sin(rad)
    return ray_distance(tx, ty, traj.x, traj.y, traj.vx, traj.vy)


def _to_action(v) -> ActionInput:
    if isinstance(v, ActionInput):
        return v
    return ActionInput(**v)1
