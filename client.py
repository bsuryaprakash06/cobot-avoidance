"""
client.py — Python client for the Industrial Robot Arm Collision Avoidance environment.

Usage:
    from client import CollisionAvoidanceClient

    env = CollisionAvoidanceClient("http://localhost:7860")
    state = env.reset()
    result = env.step(delta_angle=10.0, send_halt=False)
    print(result.reward)
"""

from __future__ import annotations
from typing import Optional
import requests
from models import EnvironmentState, StepResult, ResetResult, TaskInfo, TaskResult


class CollisionAvoidanceClient:
    """HTTP client wrapping the OpenEnv server API."""

    def __init__(self, base_url: str = "http://localhost:7860", timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def __repr__(self) -> str:
        return f"CollisionAvoidanceClient(base_url='{self.base_url}', timeout={self.timeout})"

    # ------------------------------------------------------------------ #
    #  OpenEnv core
    # ------------------------------------------------------------------ #

    def reset(self, seed: Optional[int] = None) -> ResetResult:
        """Reset the environment and return the initial state."""
        params = {"seed": seed} if seed is not None else {}
        r = requests.post(f"{self.base_url}/reset", params=params, timeout=self.timeout)
        r.raise_for_status()
        return ResetResult(**r.json())

    def step(self, delta_angle: float = 0.0, send_halt: bool = False) -> StepResult:
        """
        Apply one action step.

        Args:
            delta_angle: Rotate end-effector by this many degrees (-15 to +15).
            send_halt:   Dispatch a halt signal toward the obstacle arm.

        Returns:
            StepResult with updated state, reward, done flag, and info.
        """
        r = requests.post(
            f"{self.base_url}/step",
            json={"delta_angle": delta_angle, "send_halt": send_halt},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return StepResult(**r.json())

    def state(self) -> EnvironmentState:
        """Return the current environment state without advancing it."""
        r = requests.get(f"{self.base_url}/state", timeout=self.timeout)
        r.raise_for_status()
        return EnvironmentState(**r.json())

    # ------------------------------------------------------------------ #
    #  Tasks
    # ------------------------------------------------------------------ #

    def list_tasks(self) -> list[TaskInfo]:
        """List all available graded tasks."""
        r = requests.get(f"{self.base_url}/tasks", timeout=self.timeout)
        r.raise_for_status()
        return [TaskInfo(**t) for t in r.json()]

    def run_task(self, task_id: str, seed: int = 42) -> TaskResult:
        """Run a specific task with the baseline policy and return the graded score."""
        r = requests.post(
            f"{self.base_url}/tasks/{task_id}/run",
            params={"seed": seed},
            timeout=120,
        )
        r.raise_for_status()
        return TaskResult(**r.json())

    def run_all_tasks(self, seed: int = 42) -> list[TaskResult]:
        """Run all tasks and return their results."""
        return [self.run_task(t.task_id, seed) for t in self.list_tasks()]

    # ------------------------------------------------------------------ #
    #  Health
    # ------------------------------------------------------------------ #

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def is_alive(self) -> bool:
        try:
            return self.health().get("status") == "ok"
        except Exception:
            return False
