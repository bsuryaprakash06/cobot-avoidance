"""
visualize.py — ASCII terminal visualizer for the Industrial Robot Arm Collision Avoidance env.

Run:
    python visualize.py                  # auto-play with heuristic policy
    python visualize.py --manual         # you control the arm interactively
    python visualize.py --delay 0.3      # slow it down (seconds per step)
"""

import argparse
import math
import os
import sys
import time

import requests

ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

# Field dimensions (must match env.py)
FIELD_W = 20
FIELD_H = 20

# Display scale: 1 env unit = 2 terminal cols, 1 terminal row
COLS = FIELD_W * 2 + 4   # +4 for border + label
ROWS = FIELD_H + 2        # +2 for border

ANSI_RESET  = "\033[0m"
ANSI_RED    = "\033[91m"
ANSI_GREEN  = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_CYAN   = "\033[96m"
ANSI_BLUE   = "\033[94m"
ANSI_BOLD   = "\033[1m"
ANSI_DIM    = "\033[2m"


def env_reset(seed=42):
    r = requests.post(f"{ENV_BASE_URL}/reset", params={"seed": seed}, timeout=5)
    r.raise_for_status()
    return r.json()["state"]


def env_step(delta_angle, send_halt):
    r = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"delta_angle": delta_angle, "send_halt": send_halt},
        timeout=5,
    )
    r.raise_for_status()
    d = r.json()
    return d["state"], d["reward"], d["done"], d.get("info", {}).get("events", [])


# ------------------------------------------------------------------ #
#  Rendering
# ------------------------------------------------------------------ #

def _to_grid(ex, ey):
    """Convert env coordinates to grid (col, row). Y is flipped (env Y=0 is bottom)."""
    col = round(ex * 2)          # 2 cols per unit
    row = FIELD_H - round(ey)    # flip Y
    return col, row


def render(state, reward, events, step, total_reward):
    # Build empty grid (list of list of chars)
    grid = [[" " for _ in range(FIELD_W * 2 + 1)] for _ in range(FIELD_H + 1)]
    colors = [[ANSI_RESET for _ in range(FIELD_W * 2 + 1)] for _ in range(FIELD_H + 1)]

    def place(ex, ey, char, color=ANSI_RESET, clamp=True):
        col, row = _to_grid(ex, ey)
        if clamp:
            col = max(0, min(FIELD_W * 2, col))
            row = max(0, min(FIELD_H, row))
        if 0 <= row <= FIELD_H and 0 <= col <= FIELD_W * 2:
            grid[row][col] = char
            colors[row][col] = color

    # --- Incoming trajectories (red *)
    for t in state.get("incoming_trajectories", []):
        if t.get("active", True):
            place(t["x"], t["y"], "*", ANSI_RED)

    # --- Halt signals (cyan -)
    for h in state.get("halt_signals", []):
        if h.get("active", True):
            place(h["x"], h["y"], "-", ANSI_CYAN)

    # --- Obstacle arm (red O)
    obs = state["obstacle"]
    place(obs["x"], obs["y"], "O", ANSI_RED)

    # --- Cobot arm: draw line from base to tip
    cobot = state["cobot"]
    base_x, base_y = cobot["x"], cobot["y"]
    tip_x, tip_y = cobot["tip_x"], cobot["tip_y"]
    angle_rad = math.radians(cobot["angle"])
    arm_len = 3.0

    # Draw arm as 6 segments
    for i in range(1, 7):
        frac = i / 6.0
        ax = base_x + arm_len * frac * math.cos(angle_rad)
        ay = base_y + arm_len * frac * math.sin(angle_rad)
        char = "=" if i < 6 else ">"
        place(ax, ay, char, ANSI_GREEN)

    # Cobot base
    place(base_x, base_y, "C", ANSI_GREEN + ANSI_BOLD)
    # Tip (end-effector)
    place(tip_x, tip_y, "E", ANSI_GREEN + ANSI_BOLD)

    # ---- Print ----
    os.system("clear")

    # Top border
    print(ANSI_DIM + "┌" + "─" * (FIELD_W * 2 + 1) + "┐" + ANSI_RESET)

    for row in range(FIELD_H + 1):
        line = ANSI_DIM + "│" + ANSI_RESET
        for col in range(FIELD_W * 2 + 1):
            line += colors[row][col] + grid[row][col] + ANSI_RESET
        line += ANSI_DIM + "│" + ANSI_RESET
        print(line)

    # Bottom border
    print(ANSI_DIM + "└" + "─" * (FIELD_W * 2 + 1) + "┘" + ANSI_RESET)

    # ---- Legend ----
    integrity_bar = "█" * cobot["integrity"] + "░" * (3 - cobot["integrity"])
    obs_bar       = "█" * obs["integrity"]   + "░" * (3 - obs["integrity"])

    broadcast_str = ""
    if obs["is_broadcasting"]:
        broadcast_str = (
            f"  {ANSI_YELLOW}⚠ BROADCASTING → angle {obs.get('next_trajectory_angle', '?'):.1f}°  "
            f"launches in {obs['charge_remaining']} step(s){ANSI_RESET}"
        )

    event_str = ""
    if events:
        event_colors = {
            "cobot_collision":       ANSI_RED,
            "cobot_shutdown":        ANSI_RED + ANSI_BOLD,
            "obstacle_halted":       ANSI_GREEN,
            "obstacle_fully_halted": ANSI_GREEN + ANSI_BOLD,
            "halt_signal_dispatched":ANSI_CYAN,
            "path_clear":            ANSI_GREEN,
            "obstacle_broadcasting": ANSI_YELLOW,
            "obstacle_trajectory_launched": ANSI_RED,
        }
        colored_events = []
        for e in events:
            c = event_colors.get(e, ANSI_RESET)
            colored_events.append(f"{c}{e}{ANSI_RESET}")
        event_str = "  Events: " + ", ".join(colored_events)

    print(
        f"\n"
        f"  Step {ANSI_BOLD}{step:>3}{ANSI_RESET}   "
        f"Reward: {ANSI_YELLOW}{reward:+.2f}{ANSI_RESET}   "
        f"Total: {ANSI_YELLOW}{total_reward:+.2f}{ANSI_RESET}"
    )
    print(
        f"  {ANSI_GREEN}[C] Cobot{ANSI_RESET}  angle={cobot['angle']:.1f}°  "
        f"tip=({tip_x:.1f},{tip_y:.1f})  "
        f"integrity: {ANSI_GREEN}{integrity_bar}{ANSI_RESET}"
    )
    print(
        f"  {ANSI_RED}[O] Obstacle{ANSI_RESET}  "
        f"launches in {obs['steps_until_launch']} step(s)  "
        f"integrity: {ANSI_RED}{obs_bar}{ANSI_RESET}"
        f"{broadcast_str}"
    )
    if event_str:
        print(event_str)

    print(
        f"\n  Legend:  "
        f"{ANSI_GREEN}C{ANSI_RESET}=cobot base  "
        f"{ANSI_GREEN}E{ANSI_RESET}=end-effector  "
        f"{ANSI_GREEN}==>{ANSI_RESET}=arm  "
        f"{ANSI_RED}O{ANSI_RESET}=obstacle  "
        f"{ANSI_RED}*{ANSI_RESET}=trajectory  "
        f"{ANSI_CYAN}-{ANSI_RESET}=halt signal"
    )


# ------------------------------------------------------------------ #
#  Heuristic auto-policy (same as tasks.py baseline)
# ------------------------------------------------------------------ #

def heuristic_action(state):
    cobot = state["cobot"]
    obstacle = state["obstacle"]
    trajectories = [t for t in state.get("incoming_trajectories", []) if t.get("active", True)]

    def ray_dist(tx, ty, rx, ry, rdx, rdy):
        mag = math.hypot(rdx, rdy) or 1e-9
        ux, uy = rdx / mag, rdy / mag
        proj = (tx - rx) * ux + (ty - ry) * uy
        if proj < 0:
            return math.hypot(tx - rx, ty - ry)
        return math.hypot(tx - (rx + proj * ux), ty - (ry + proj * uy))

    if not trajectories:
        dx = obstacle["x"] - cobot["x"]
        dy = obstacle["y"] - cobot["y"]
        target = math.degrees(math.atan2(dy, dx))
        diff = (target - cobot["angle"] + 180) % 360 - 180
        delta = max(-15.0, min(15.0, diff * 0.5))
        return delta, abs(diff) < 12.0

    worst = min(
        trajectories,
        key=lambda t: ray_dist(cobot["tip_x"], cobot["tip_y"], t["x"], t["y"], t["vx"], t["vy"])
    )

    def tip_clear(d):
        rad = math.radians(cobot["angle"] + d)
        tx = cobot["x"] + 3.0 * math.cos(rad)
        ty = cobot["y"] + 3.0 * math.sin(rad)
        return ray_dist(tx, ty, worst["x"], worst["y"], worst["vx"], worst["vy"])

    delta = 10.0 if tip_clear(10.0) > tip_clear(-10.0) else -10.0
    return delta, False


# ------------------------------------------------------------------ #
#  Manual control
# ------------------------------------------------------------------ #

def get_manual_action():
    print(
        "\n  Controls: "
        "[a] rotate -10°  [d] rotate +10°  "
        "[h] halt signal  [s] stay still  [q] quit"
        "\n  > ",
        end="", flush=True,
    )
    try:
        key = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None, None, True

    if key == "q":
        return None, None, True
    elif key == "a":
        return -10.0, False, False
    elif key == "d":
        return 10.0, False, False
    elif key == "h":
        return 0.0, True, False
    else:
        return 0.0, False, False


# ------------------------------------------------------------------ #
#  Main loop
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual", action="store_true", help="Control the arm yourself")
    parser.add_argument("--delay", type=float, default=0.15, help="Seconds between steps (auto mode)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    state = env_reset(args.seed)
    render(state, 0.0, [], 0, 0.0)

    total_reward = 0.0
    step = 0

    while True:
        if args.manual:
            delta, halt, quit_ = get_manual_action()
            if quit_:
                print("\nExiting.")
                break
        else:
            delta, halt = heuristic_action(state)
            time.sleep(args.delay)

        state, reward, done, events = env_step(delta, halt)
        total_reward += reward
        step += 1

        render(state, reward, events, step, total_reward)

        if done:
            outcome = (
                f"{ANSI_GREEN}COBOT SURVIVED!{ANSI_RESET}"
                if state["cobot"]["integrity"] > 0
                else f"{ANSI_RED}COBOT SHUTDOWN{ANSI_RESET}"
            )
            print(f"\n  Episode finished after {step} steps. {outcome}  Total reward: {total_reward:.2f}")
            if not args.manual:
                break
            print("  Press Enter to reset, q to quit.")
            try:
                key = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                break
            if key == "q":
                break
            state = env_reset(args.seed)
            total_reward = 0.0
            step = 0
            render(state, 0.0, [], 0, 0.0)


if __name__ == "__main__":
    main()
