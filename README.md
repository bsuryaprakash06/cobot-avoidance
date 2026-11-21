---
title: Industrial Robot Arm Collision Avoidance
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Industrial Robot Arm Collision Avoidance

An OpenEnv-compliant reinforcement learning and simulation environment modeling collaborative robot arm (cobot) safety planning on a shared industrial factory floor, adhering to ISO 10218-1 and ISO 15066 predictive replanning standards.

---

## Motivation & Domain Background

Modern industrial production lines increasingly deploy multiple robotic manipulators within shared physical cells. Unplanned trajectory intersections cause mechanical damage, production halts, and personnel hazards. 

Safety standards **ISO 10218** (Robots and robotic devices — Safety requirements for industrial robots) and **ISO 15066** (Collaborative robots) specify that automated manipulators broadcast intended motion vectors prior to actuation. This provides adjacent controllers a critical time window to replan end-effector trajectories and dispatch safety halt signals.

This environment trains an intelligent control policy to parse broadcasted trajectory metadata, dynamically rotate the manipulator out of projected intersection corridors *before* the hazard launches, and dispatch targeted halt commands when safety margins are compromised.

```
+-----------------------------------------------------------------------------------+
| Factory Floor Environment (20m x 20m)                                             |
|                                                                                   |
|   Cobot Arm [Base: (2, 10)]                         Obstacle Arm [Base: (18, 10)] |
|   Length: 3.0 m                                     Length: 3.0 m                 |
|                                                                                   |
|   (Cobot)================>[Tip]                     (Obstacle)                    |
|             \                                             /                       |
|              \--- ISO 10218 Trajectory Broadcast <-------/                        |
|                   (Pre-launch Warning: 3 steps)                                   |
|                                                                                   |
|              [Cobot Actions]                                                      |
|              - delta_angle: [-15°, +15°]                                          |
|              - send_halt: True / False                                            |
+-----------------------------------------------------------------------------------+
```

---

## Environment Kinematics & Mechanics

| Parameter | Specification | Description |
|:---|:---|:---|
| **Arena Dimensions** | `20 x 20` meters | Cartesian boundary with coordinate origin `(0, 0)` at lower-left |
| **Cobot Base Position** | `(2.0, 10.0)` | Fixed planar mounting coordinate |
| **Obstacle Base Position** | `(18.0, 10.0)` | Fixed planar mounting coordinate |
| **Manipulator Reach** | `3.0` meters | End-effector radius from base |
| **Cobot Integrity** | `3` points | Health points; collision decrements integrity; 0 = emergency shutdown |
| **Obstacle Integrity** | `3` points | Health points; successful halt signal decrements integrity |
| **Episode Length** | `100` steps | Maximum duration per simulation episode |
| **Obstacle Speed** | `1.5` m / step | Velocity of launched trajectory rays |
| **Halt Signal Speed** | `2.0` m / step | Velocity of transmitted safety stop signal |
| **Launch Cadence** | `8 ± 2` steps | Random uniform launch interval |
| **Broadcast Horizon** | `3` steps | Pre-launch window alerting adjacent arms of planned trajectory angle |

### Predictive Replanning (ISO 10218 Mechanics)

Three discrete timesteps prior to launching an obstacle trajectory, the hazard arm broadcasts its target angle:
- `obstacle.is_broadcasting`: Flag switches to `True`.
- `obstacle.next_trajectory_angle`: Publishes the planned firing angle (degrees).
- `obstacle.charge_remaining`: Countdown indicator (steps remaining until execution).

An optimal agent uses this preview signal to rotate its end-effector clear of the projected ray *before launch*, achieving deterministic clearance without relying solely on airborne obstacle reactivity.

---

## State, Observation & Action Spaces

### Action Space

```json
{
  "delta_angle": 10.0,
  "send_halt": false
}
```

- **`delta_angle`** (`float`, range `[-15.0, 15.0]`): Angular rotation applied to the end-effector in degrees.
- **`send_halt`** (`boolean`): Dispatches a directional safety halt signal along the end-effector's current heading vector.

### Observation Space

The state payload returned on each step contains structured telemetry:

| Field | Type | Description |
|:---|:---|:---|
| `cobot.angle` | `float` | Current end-effector heading angle in degrees `[-180°, 180°]` |
| `cobot.tip_x`, `cobot.tip_y` | `float` | Cartesian coordinates of the end-effector tip (meters) |
| `cobot.integrity` | `integer` | Remaining health points (`0–3`) |
| `obstacle.is_broadcasting` | `boolean` | Active pre-launch warning status |
| `obstacle.next_trajectory_angle`| `float` or `null` | Broadcasted trajectory firing angle |
| `obstacle.charge_remaining` | `integer` | Timesteps until trajectory actuation |
| `obstacle.integrity` | `integer` | Remaining integrity of obstacle arm (`0–3`) |
| `incoming_trajectories` | `list[object]` | Active airborne rays with coordinates `(x, y)` and velocities `(vx, vy)` |

---

## Reward Formulation

The environment provides a dense step-wise reward signal:

| Operational Event | Reward Weight | Description |
|:---|:---:|:---|
| **Safe Clearance** | `+0.3` | End-effector maintains `> 1.5m` clearance from active trajectory line |
| **Halt Connection** | `+2.0` | Dispatched halt signal successfully strikes obstacle arm |
| **Obstacle Disabled** | `+3.0` | Obstacle integrity fully depleted to 0 (bonus reward) |
| **Collision Incurred** | `-1.0` | End-effector struck by an incoming trajectory |
| **Manipulator Shutdown** | `-2.0` | Cobot integrity reaches 0 (critical failure) |

---

## Benchmark Evaluation Tasks

The environment implements three standardized benchmark tasks with quantitative scoring formulas:

| Task Identifier | Difficulty | Objective | Scoring Formulation |
|:---|:---:|:---|:---|
| **`task_1_zero_collision`** | Basic | Operate across 100 steps with zero collisions | `Score = (integrity / 3.0) * (steps / 100.0)` |
| **`task_2_predictive_replan`** | Intermediate | Survive while maintaining `>= 80%` pre-launch clearance | `Score = 0.5 * (integrity / 3.0) + 0.5 * clearance_rate` |
| **`task_3_halt_dispatch`** | Advanced | Neutralize the hazard arm using directed halt signals | `Score = 0.4 * (integrity / 3.0) + 0.6 * (damage / 3.0)` |

### Baseline Policy Benchmark Scores

| Task Identifier | Heuristic Baseline Score Range |
|:---|:---:|
| `task_1_zero_collision` | `0.85 – 1.00` |
| `task_2_predictive_replan` | `0.70 – 0.85` |
| `task_3_halt_dispatch` | `0.55 – 0.75` |

---

## Installation & Local Execution

### 1. Prerequisites

- Python 3.11+
- Virtual environment manager (`venv`, `uv`, or `conda`)
- Docker & Docker Compose *(optional)*

### 2. Setup Environment

```bash
# Clone repository
git clone https://github.com/bsuryaprakash06/cobot-avoidance.git
cd cobot-avoidance

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch Environment Server

Start the FastAPI / OpenEnv simulation server on port `7860`:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Alternatively, run the server entrypoint directly:

```bash
python server/app.py
```

The interactive OpenAPI documentation will be available at `http://localhost:7860/docs`.

---

## Interactive Terminal Visualizer

The project includes an ASCII terminal visualizer (`visualize.py`) rendering real-time kinematics:

```bash
# Auto-play using the heuristic avoidance policy
python visualize.py

# Interactive manual control mode
python visualize.py --manual

# Slow down visualization for detailed inspection (0.3s per step)
python visualize.py --delay 0.3
```

---

## Python Client SDK Usage

```python
from client import CollisionAvoidanceClient

# Connect to running environment server
env = CollisionAvoidanceClient("http://localhost:7860")

# Reset episode with deterministic seed
reset_result = env.reset(seed=42)
print("Initial State:", reset_result.state)

# Step the simulation
step_result = env.step(delta_angle=10.0, send_halt=False)
print("Reward:", step_result.reward)
print("Done:", step_result.done)
```

---

## LLM-Driven Agent Inference

The repository contains an automated LLM controller (`inference.py`) supporting OpenAI-compatible endpoints:

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key-here"
export ENV_BASE_URL="http://localhost:7860"

python inference.py
```

The inference pipeline emits structured JSON Lines stdout logs:

```json
[START] {"tag": "[START]", "task_id": "task_1_zero_collision", "episode": 0}
[STEP]  {"tag": "[STEP]", "step": 1, "action": {"delta_angle": 5.0, "send_halt": false}, "reward": 0.3, "total_reward": 0.3, "done": false}
[END]   {"tag": "[END]", "task_id": "task_1_zero_collision", "score": 0.94, "total_reward": 22.4, "steps": 100, "success": true}
[SUMMARY] {"tag": "[SUMMARY]", "mean_score": 0.81, "task_scores": {"task_1_zero_collision": 0.94}}
```

---

## Docker Deployment

### Single-Command Launch with Docker Compose

```bash
docker compose up --build
```

### Manual Docker Build

```bash
docker build -t cobot-avoidance-env .
docker run -p 7860:7860 cobot-avoidance-env
```

---

## REST & WebSocket API Reference

| Protocol | Method | Endpoint | Description |
|:---|:---:|:---|:---|
| **HTTP** | `POST` | `/reset` | Initialize simulation episode (accepts optional `?seed=int`) |
| **HTTP** | `POST` | `/step` | Submit action payload (`delta_angle`, `send_halt`) |
| **HTTP** | `GET` | `/state` | Retrieve current telemetry state without advancing timestep |
| **HTTP** | `GET` | `/tasks` | List all standardized benchmark tasks |
| **HTTP** | `POST` | `/tasks/{task_id}/run` | Execute specified task against baseline policy |
| **HTTP** | `GET` | `/health` | Health check probe (returns `{"status": "ok"}`) |
| **WS** | `WS` | `/ws` | Bidirectional real-time WebSocket state and action streaming |

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for complete details.
