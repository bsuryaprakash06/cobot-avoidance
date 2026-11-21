"""
app.py — FastAPI application with HTTP + WebSocket endpoints.

HTTP (OpenEnv standard):
  POST /reset               → ResetResult
  POST /step                → StepResult
  GET  /state               → EnvironmentState
  GET  /tasks               → list[TaskInfo]
  POST /tasks/{task_id}/run → TaskResult
  GET  /health              → {"status": "ok"}

WebSocket:
  WS /ws  — stream state updates in real time
             send JSON action {"delta_angle": float, "send_halt": bool}
             receive JSON StepResult each step
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from models import ActionInput, StepResult, ResetResult, EnvironmentState, TaskInfo, TaskResult
from server.robot_arm_environment import CollisionAvoidanceEnv
from server.tasks import TASKS, run_task_1, run_task_2, run_task_3

app = FastAPI(
    title="Industrial Robot Arm Collision Avoidance — OpenEnv",
    description=(
        "A collaborative robot arm (cobot) operates on a shared factory floor alongside "
        "a moving obstacle arm. The cobot must replan its end-effector path to avoid "
        "collision zones using the obstacle's broadcasted trajectory (ISO 10218 / ISO 15066), "
        "and dispatch halt signals to stop the obstacle arm."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_env = CollisionAvoidanceEnv(seed=42)


# ------------------------------------------------------------------ #
#  OpenEnv HTTP endpoints
# ------------------------------------------------------------------ #

@app.post("/reset", response_model=ResetResult, tags=["OpenEnv"])
def reset(seed: int | None = None):
    """Reset the environment and start a new episode."""
    global _env
    if seed is not None:
        _env = CollisionAvoidanceEnv(seed=seed)
    return _env.reset()


@app.post("/step", response_model=StepResult, tags=["OpenEnv"])
def step(action: ActionInput):
    """Apply a replanning action and advance the environment by one step."""
    return _env.step(action)


@app.get("/state", response_model=EnvironmentState, tags=["OpenEnv"])
def state():
    """Return the current environment state without advancing it."""
    return _env.get_state()


# ------------------------------------------------------------------ #
#  Task endpoints
# ------------------------------------------------------------------ #

@app.get("/tasks", response_model=list[TaskInfo], tags=["Tasks"])
def list_tasks():
    """List all available graded tasks."""
    return TASKS


@app.post("/tasks/{task_id}/run", response_model=TaskResult, tags=["Tasks"])
def run_task(task_id: str, seed: int = 42):
    """Run a task with the built-in heuristic baseline and return the graded score."""
    runners = {
        "task_1_zero_collision": run_task_1,
        "task_2_predictive_replan": run_task_2,
        "task_3_halt_dispatch": run_task_3,
    }
    if task_id not in runners:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_id}' not found. Available tasks: {list(runners.keys())}"
        )
    return runners[task_id](policy_fn=None, seed=seed)


# ------------------------------------------------------------------ #
#  WebSocket endpoint — real-time streaming
# ------------------------------------------------------------------ #

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket interface for real-time interaction.
    Client sends: {"delta_angle": float, "send_halt": bool}
    Server sends:  StepResult JSON each step, or {"error": "..."} on bad input.
    Connect and send {"reset": true} to start a new episode.
    """
    await websocket.accept()
    ws_env = CollisionAvoidanceEnv(seed=42)
    reset_result = ws_env.reset()
    await websocket.send_json(reset_result.model_dump())

    try:
        while True:
            data = await websocket.receive_json()
            if data.get("reset"):
                seed = data.get("seed", 42)
                ws_env = CollisionAvoidanceEnv(seed=seed)
                result = ws_env.reset()
                await websocket.send_json(result.model_dump())
                continue
            try:
                action = ActionInput(**data)
            except Exception as e:
                await websocket.send_json({"error": str(e)})
                continue
            result = ws_env.step(action)
            await websocket.send_json(result.model_dump())
    except WebSocketDisconnect:
        pass


# ------------------------------------------------------------------ #
#  Health / root
# ------------------------------------------------------------------ #

@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "environment": "industrial_robot_arm_collision_avoidance",
        "version": "1.0.0"
    }


@app.get("/", tags=["Health"])
def root():
    return {
        "name": "Industrial Robot Arm Collision Avoidance",
        "version": "1.0.0",
        "description": "OpenEnv environment — cobot safety planning on shared factory floor",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/health", "/ws"],
    }


def main():
    """Start the ASGI server.

    This function is provided so the project can expose a callable entrypoint
    (e.g. for packaging or multi-mode deployment).
    """
    import uvicorn
    # Use the configured app directly. In deployment the host/port may be
    # overridden by the environment or the process manager.
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
