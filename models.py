"""
Typed Pydantic models for the Industrial Robot Arm Collision Avoidance environment.

Real-world scenario:
  A collaborative robot arm (cobot) operates on a shared factory floor.
  A second arm / moving obstacle broadcasts its planned trajectory.
  The cobot must replan its end-effector path to avoid collision zones
  and, when safe, send a halt signal to the obstacle arm.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class CobotArm(BaseModel):
    x: float = Field(..., description="Base X position of the cobot (fixed, metres)")
    y: float = Field(..., description="Base Y position of the cobot (fixed, metres)")
    angle: float = Field(..., description="Current joint angle in degrees (0=right, 90=up)")
    tip_x: float = Field(..., description="End-effector X position")
    tip_y: float = Field(..., description="End-effector Y position")
    integrity: int = Field(..., description="Structural integrity points remaining (0 = damaged/shutdown)")
    halt_signals_sent: int = Field(..., description="Total halt signals dispatched this episode")
    halt_signals_effective: int = Field(..., description="Halt signals that successfully stopped an obstacle trajectory")

    @property
    def is_operational(self) -> bool:
        """True if the manipulator maintains positive integrity points."""
        return self.integrity > 0


class ObstacleArm(BaseModel):
    x: float = Field(..., description="Obstacle arm base X position")
    y: float = Field(..., description="Obstacle arm base Y position")
    integrity: int = Field(..., description="Remaining operational cycles before it self-halts (0 = halted)")
    charge_remaining: int = Field(..., description="Steps until next trajectory is launched (0 = not broadcasting)")
    next_trajectory_angle: Optional[float] = Field(
        None,
        description="Broadcasted trajectory angle (degrees) — available during charge window for pre-emptive replanning"
    )
    is_broadcasting: bool = Field(..., description="True when obstacle arm is broadcasting its next trajectory")
    steps_until_launch: int = Field(..., description="Steps until obstacle arm launches next trajectory")

    @property
    def is_active(self) -> bool:
        """True if the obstacle arm maintains operational integrity remaining."""
        return self.integrity > 0


class TrajectoryProjectile(BaseModel):
    id: str = Field(..., description="Unique trajectory ID")
    x: float = Field(..., description="Current position X")
    y: float = Field(..., description="Current position Y")
    vx: float = Field(..., description="Velocity X per step (m/s equivalent)")
    vy: float = Field(..., description="Velocity Y per step")
    source: str = Field(..., description="'obstacle' or 'cobot_halt'")
    active: bool = Field(..., description="False once the trajectory has resolved (hit or cleared)")


class EnvironmentState(BaseModel):
    episode_id: str
    step: int
    done: bool
    cobot: CobotArm
    obstacle: ObstacleArm
    incoming_trajectories: list[TrajectoryProjectile] = Field(
        default_factory=list,
        description="Active obstacle arm trajectories heading toward the workspace"
    )
    halt_signals: list[TrajectoryProjectile] = Field(
        default_factory=list,
        description="Active halt signals dispatched by the cobot"
    )
    last_reward: float
    last_events: list[str]
    total_reward: float


# ------------------------------------------------------------------ #
#  API I/O models
# ------------------------------------------------------------------ #

class ActionInput(BaseModel):
    delta_angle: float = Field(
        0.0,
        ge=-15.0, le=15.0,
        description="Replan end-effector angle by this many degrees (-15 to +15)"
    )
    send_halt: bool = Field(
        False,
        description="Dispatch a halt signal toward the obstacle arm in the current end-effector direction"
    )


class StepResult(BaseModel):
    state: EnvironmentState
    reward: float
    done: bool
    truncated: bool
    info: dict


class ResetResult(BaseModel):
    episode_id: str
    state: EnvironmentState
    message: str


# ------------------------------------------------------------------ #
#  Task / Grader models
# ------------------------------------------------------------------ #

class TaskInfo(BaseModel):
    task_id: str
    name: str
    description: str
    difficulty: str
    max_steps: int
    reward_range: list[float]


class TaskResult(BaseModel):
    task_id: str
    score: float = Field(..., gt=0.0, lt=1.0)
    reward: float
    steps_taken: int
    success: bool
    details: dict