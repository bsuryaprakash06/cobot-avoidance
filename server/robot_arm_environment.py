"""
robot_arm_environment.py — Core environment logic.

Industrial Robot Arm Collision Avoidance:
  A collaborative robot arm (cobot) operates on a shared factory floor alongside
  a moving obstacle arm. The obstacle broadcasts its planned trajectory 3 steps
  before executing it (ISO 10218 safety broadcast window). The cobot must replan
  its end-effector path to avoid collision zones and dispatch halt signals.
"""

import math
import random
import uuid
from typing import Optional

from models import (
    CobotArm, ObstacleArm, TrajectoryProjectile,
    EnvironmentState, ActionInput, StepResult, ResetResult,
)


class CollisionAvoidanceEnv:
    FIELD_W = 20.0
    FIELD_H = 20.0
    COBOT_X = 2.0
    COBOT_Y = 10.0
    OBSTACLE_X = 18.0
    OBSTACLE_Y = 10.0
    ARM_LENGTH = 3.0
    TRAJ_SPEED = 1.5
    HALT_SPEED = 2.0
    MAX_STEPS = 100
    BROADCAST_WINDOW = 3
    LAUNCH_INTERVAL = 8

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.rng = random.Random(seed)
        self._episode_id = str(uuid.uuid4())
        self.reset()

    def reset(self) -> ResetResult:
        self._episode_id = str(uuid.uuid4())
        self.step_count = 0
        self.done = False
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.last_events: list[str] = []

        self.cobot = CobotArm(
            x=self.COBOT_X, y=self.COBOT_Y,
            angle=90.0,
            tip_x=self.COBOT_X,
            tip_y=self.COBOT_Y + self.ARM_LENGTH,
            integrity=3,
            halt_signals_sent=0,
            halt_signals_effective=0,
        )
        self.obstacle = ObstacleArm(
            x=self.OBSTACLE_X, y=self.OBSTACLE_Y,
            integrity=3,
            charge_remaining=0,
            next_trajectory_angle=None,
            is_broadcasting=False,
            steps_until_launch=self.LAUNCH_INTERVAL,
        )
        self.incoming_trajectories: list[TrajectoryProjectile] = []
        self.halt_signals: list[TrajectoryProjectile] = []

        return ResetResult(
            episode_id=self._episode_id,
            state=self._make_state(),
            message=(
                "Environment reset. Cobot ready. "
                "Monitor obstacle broadcasts and replan end-effector path to avoid collision zones."
            ),
        )

    def step(self, action: ActionInput) -> StepResult:
        if self.done:
            return StepResult(
                state=self._make_state(), reward=0.0,
                done=True, truncated=False,
                info={"message": "Episode complete. Call reset() to start a new episode."},
            )

        self.step_count += 1
        reward = 0.0
        events: list[str] = []

        # 1. Apply cobot replanning
        delta = max(-15.0, min(15.0, action.delta_angle))
        self.cobot.angle = (self.cobot.angle + delta) % 360.0
        self._update_tip()

        # 2. Dispatch halt signal
        if action.send_halt:
            hs = TrajectoryProjectile(
                id=str(uuid.uuid4()),
                x=self.cobot.tip_x, y=self.cobot.tip_y,
                vx=self.HALT_SPEED * math.cos(math.radians(self.cobot.angle)),
                vy=self.HALT_SPEED * math.sin(math.radians(self.cobot.angle)),
                source="cobot_halt", active=True,
            )
            self.halt_signals.append(hs)
            self.cobot.halt_signals_sent += 1
            events.append("halt_signal_dispatched")

        # 3. Obstacle broadcast / launch cycle
        self.obstacle.steps_until_launch -= 1

        if self.obstacle.steps_until_launch == self.BROADCAST_WINDOW:
            self.obstacle.is_broadcasting = True
            self.obstacle.charge_remaining = self.BROADCAST_WINDOW
            dx = self.cobot.tip_x - self.obstacle.x
            dy = self.cobot.tip_y - self.obstacle.y
            self.obstacle.next_trajectory_angle = math.degrees(math.atan2(dy, dx))
            events.append("obstacle_broadcasting")

        if self.obstacle.steps_until_launch <= 0:
            angle = self.obstacle.next_trajectory_angle if self.obstacle.next_trajectory_angle is not None else 180.0
            tp = TrajectoryProjectile(
                id=str(uuid.uuid4()),
                x=self.obstacle.x, y=self.obstacle.y,
                vx=self.TRAJ_SPEED * math.cos(math.radians(angle)),
                vy=self.TRAJ_SPEED * math.sin(math.radians(angle)),
                source="obstacle", active=True,
            )
            self.incoming_trajectories.append(tp)
            self.obstacle.is_broadcasting = False
            self.obstacle.charge_remaining = 0
            self.obstacle.next_trajectory_angle = None
            self.obstacle.steps_until_launch = self.LAUNCH_INTERVAL + self.rng.randint(-2, 2)
            events.append("obstacle_trajectory_launched")

        if self.obstacle.is_broadcasting:
            self.obstacle.charge_remaining -= 1

        # 4. Propagate
        for t in self.incoming_trajectories:
            if t.active:
                t.x += t.vx; t.y += t.vy
        for h in self.halt_signals:
            if h.active:
                h.x += h.vx; h.y += h.vy

        # 5. Collision: obstacle trajectory hits cobot tip
        for t in self.incoming_trajectories:
            if not t.active:
                continue
            if math.hypot(t.x - self.cobot.tip_x, t.y - self.cobot.tip_y) < 1.2:
                t.active = False
                self.cobot.integrity -= 1
                reward -= 1.0
                events.append("cobot_collision")

        # 6. Halt signal reaches obstacle
        for h in self.halt_signals:
            if not h.active:
                continue
            if math.hypot(h.x - self.obstacle.x, h.y - self.obstacle.y) < 1.5:
                h.active = False
                self.obstacle.integrity -= 1
                self.cobot.halt_signals_effective += 1
                reward += 2.0
                events.append("obstacle_halted")

        # 7. Clearance reward
        clearance = self._clearance_reward()
        reward += clearance
        if clearance > 0.1:
            events.append("path_clear")

        # 8. Prune out-of-bounds
        self.incoming_trajectories = [t for t in self.incoming_trajectories if t.active and self._in_bounds(t)]
        self.halt_signals = [h for h in self.halt_signals if h.active and self._in_bounds(h)]

        # 9. Terminal conditions
        truncated = self.step_count >= self.MAX_STEPS
        terminal = self.cobot.integrity <= 0 or self.obstacle.integrity <= 0 or truncated

        if self.cobot.integrity <= 0:
            reward -= 2.0
            events.append("cobot_shutdown")
        if self.obstacle.integrity <= 0:
            reward += 3.0
            events.append("obstacle_fully_halted")

        self.done = terminal
        self.total_reward += reward
        self.last_reward = reward
        self.last_events = events

        return StepResult(
            state=self._make_state(),
            reward=round(reward, 4),
            done=terminal, truncated=truncated,
            info={"events": events, "step": self.step_count,
                  "total_reward": round(self.total_reward, 4)},
        )

    def get_state(self) -> EnvironmentState:
        return self._make_state()

    def _update_tip(self):
        rad = math.radians(self.cobot.angle)
        self.cobot.tip_x = self.cobot.x + self.ARM_LENGTH * math.cos(rad)
        self.cobot.tip_y = self.cobot.y + self.ARM_LENGTH * math.sin(rad)

    def _clearance_reward(self) -> float:
        if not self.incoming_trajectories:
            return 0.0
        reward = 0.0
        for t in self.incoming_trajectories:
            if t.active and ray_distance(self.cobot.tip_x, self.cobot.tip_y, t.x, t.y, t.vx, t.vy) > 1.5:
                reward += 0.3
        return reward

    def _in_bounds(self, t: TrajectoryProjectile) -> bool:
        return 0 <= t.x <= self.FIELD_W and 0 <= t.y <= self.FIELD_H

    def _make_state(self) -> EnvironmentState:
        return EnvironmentState(
            episode_id=self._episode_id,
            step=self.step_count,
            done=self.done,
            cobot=self.cobot.model_copy(),
            obstacle=self.obstacle.model_copy(),
            incoming_trajectories=[t.model_copy() for t in self.incoming_trajectories],
            halt_signals=[h.model_copy() for h in self.halt_signals],
            last_reward=self.last_reward,
            last_events=self.last_events,
            total_reward=round(self.total_reward, 4),
        )


def ray_distance(px, py, rx, ry, rdx, rdy) -> float:
    mag = math.hypot(rdx, rdy)
    if mag < 1e-9:
        return math.hypot(px - rx, py - ry)
    ux, uy = rdx / mag, rdy / mag
    t = (px - rx) * ux + (py - ry) * uy
    if t < 0:
        return math.hypot(px - rx, py - ry)
    return math.hypot(px - (rx + t * ux), py - (ry + t * uy))
