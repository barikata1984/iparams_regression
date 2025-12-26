#!/usr/bin/env python3
"""
Realtime Display: Continuous ROLS and RTLS estimation with live terminal output.

This script continuously loops through the data and displays ROLS/RTLS estimates
at 2 Hz until the user presses Esc.

Usage:
    python -m src.realtime_display
"""

import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path

try:
    import termios
    import tty
    import select

    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False

from .recursive_ols import RecursiveOLS
from .recursive_tls import RecursiveTLS


def load_data(json_path: str) -> tuple:
    """Load regressor matrices and force/torque measurements from JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    regressors = []
    ft_measurements = []

    for frame in data["frames"]:
        regressor = np.array(frame["regressor"])
        ft_sen = np.array(frame["ft_sen"])
        regressors.append(regressor)
        ft_measurements.append(ft_sen)

    ground_truth = None
    if "global_gt" in data:
        ground_truth = np.array(data["global_gt"][:10])

    return regressors, ft_measurements, ground_truth


def check_for_esc() -> bool:
    """Check if Esc key was pressed (non-blocking)."""
    if not HAS_TERMIOS:
        return False

    if select.select([sys.stdin], [], [], 0)[0]:
        ch = sys.stdin.read(1)
        if ch == "\x1b":  # Esc key
            return True
    return False


def clear_screen():
    """Clear the terminal screen."""
    print("\033[2J\033[H", end="")


def move_cursor_home():
    """Move cursor to home position."""
    print("\033[H", end="")


def format_params(params: np.ndarray, name: str) -> str:
    """Format parameters for display."""
    lines = [f"{name}:"]
    param_names = [
        "mass",
        "com_x",
        "com_y",
        "com_z",
        "Ixx",
        "Iyy",
        "Izz",
        "Ixy",
        "Iyz",
        "Izx",
    ]
    for i, (pname, val) in enumerate(zip(param_names, params)):
        lines.append(f"  θ{i + 1:2d} ({pname:5s}): {val:12.6f}")
    return "\n".join(lines)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Realtime ROLS/RTLS display with continuous loop"
    )
    parser.add_argument(
        "--hz", type=float, default=2.0, help="Display update rate in Hz (default: 2.0)"
    )
    parser.add_argument(
        "--lambda",
        dest="forgetting_factor",
        type=float,
        default=0.99,
        help="Forgetting factor for recursive estimators (default: 0.99)",
    )
    args = parser.parse_args()

    display_interval = 1.0 / args.hz
    forgetting_factor = args.forgetting_factor

    print("=" * 60)
    print("Realtime ROLS/RTLS Display")
    print("Press Esc to exit")
    print("=" * 60)
    print(f"Update rate: {args.hz} Hz (interval: {display_interval:.3f}s)")
    print(f"Forgetting factor: {forgetting_factor}")

    # Load data
    data_path = Path(__file__).parent.parent / "data" / "transforms.json"
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return

    regressors, ft_measurements, ground_truth = load_data(str(data_path))
    n_frames = len(regressors)
    n_params = regressors[0].shape[1]

    print(f"Data loaded: {n_frames} frames, {n_params} parameters")
    print("Starting in 2 seconds...\n")
    time.sleep(2)

    # Initialize estimators
    rols = RecursiveOLS(n_params=n_params, forgetting_factor=forgetting_factor)
    rtls = RecursiveTLS(n_params=n_params, forgetting_factor=forgetting_factor)

    # Setup terminal for non-blocking input
    if HAS_TERMIOS:
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            run_loop(
                rols,
                rtls,
                regressors,
                ft_measurements,
                ground_truth,
                n_frames,
                n_params,
                forgetting_factor,
                display_interval,
            )
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    else:
        print("Warning: termios not available. Press Ctrl+C to exit.")
        try:
            run_loop(
                rols,
                rtls,
                regressors,
                ft_measurements,
                ground_truth,
                n_frames,
                n_params,
                forgetting_factor,
                display_interval,
            )
        except KeyboardInterrupt:
            pass

    print("\n\nExiting...")


def run_loop(
    rols,
    rtls,
    regressors,
    ft_measurements,
    ground_truth,
    n_frames,
    n_params,
    forgetting_factor,
    display_interval,
):
    """Main display loop - processes one frame per display cycle."""
    frame_idx = 0
    loop_count = 0

    clear_screen()

    while True:
        # Check for Esc key
        if check_for_esc():
            break

        # Get current frame data
        regressor = regressors[frame_idx]
        ft_sen = ft_measurements[frame_idx]

        # Update estimators with each row of the frame
        for row_idx in range(len(ft_sen)):
            rols.update(regressor[row_idx], ft_sen[row_idx])
            rtls.update(regressor[row_idx], ft_sen[row_idx])

        # Display current state
        move_cursor_home()

        rols_est = rols.get_estimate()
        rtls_est = rtls.get_estimate()

        print("=" * 60)
        print("Realtime ROLS/RTLS Display (Press Esc to exit)")
        print("=" * 60)
        print(
            f"Frame: {frame_idx:4d} / {n_frames} | Loop: {loop_count} | λ = {forgetting_factor}"
        )
        print(f"Total updates: {rols.n_updates}")
        print("-" * 60)

        # Display ROLS
        print(format_params(rols_est, "ROLS"))
        print()

        # Display RTLS
        print(format_params(rtls_est, "RTLS"))
        print()

        # Display ground truth and errors if available
        if ground_truth is not None:
            rols_error = np.linalg.norm(rols_est - ground_truth)
            rtls_error = np.linalg.norm(rtls_est - ground_truth)
            print("-" * 60)
            print("Error vs Ground Truth:")
            print(f"  ROLS: ||error|| = {rols_error:.6f}")
            print(f"  RTLS: ||error|| = {rtls_error:.6f}")

        print("-" * 60)
        print("Press Esc to exit...")

        # Move to next frame
        frame_idx += 1
        if frame_idx >= n_frames:
            frame_idx = 0
            loop_count += 1

        # Wait for next display cycle (2 Hz)
        time.sleep(display_interval)


if __name__ == "__main__":
    main()
