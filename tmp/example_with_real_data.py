#!/usr/bin/env python3
"""
Example: Using ROLS and RTLS with Real Robot Data

This script demonstrates recursive OLS and TLS for inertial parameter
estimation using real data from rtls_eval.json.

Usage:
    python -m src.example_with_real_data
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from .recursive_ols import RecursiveOLS, solve_ols_batch
from .recursive_tls import RecursiveTLS, solve_tls_batch


def load_data(json_path: str) -> tuple:
    """
    Load regressor matrices and force/torque measurements from JSON file.

    Args:
        json_path: Path to the JSON data file

    Returns:
        Tuple of (regressors, ft_measurements, datetime_str, ground_truth)
        - regressors: List of (n_measurements, n_params) arrays per frame
        - ft_measurements: List of (n_measurements,) arrays per frame
        - datetime_str: Recording datetime string
        - ground_truth: Ground truth parameters (first 10 elements of global_gt)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    regressors = []
    ft_measurements = []

    # Sort frames by index (file_path contains frame number)
    frames = data["frames"]

    for frame in frames:
        regressor = np.array(frame["regressor"])  # (6, 10) matrix
        ft_sen = np.array(frame["ft_sen"])  # (6,) vector
        regressors.append(regressor)
        ft_measurements.append(ft_sen)

    # Extract ground truth (first 10 elements, excluding aabb_scale)
    ground_truth = None
    if "global_gt" in data:
        ground_truth = np.array(data["global_gt"][:10])

    datetime_str = data.get("date_time", "unknown")

    return regressors, ft_measurements, datetime_str, ground_truth


def run_recursive_estimation(
    regressors: list,
    ft_measurements: list,
    n_params: int = 10,
    forgetting_factor: float = 1.0,
) -> tuple:
    """
    Run recursive OLS and TLS estimation on the data.

    Args:
        regressors: List of regressor matrices per frame
        ft_measurements: List of F/T measurements per frame
        n_params: Number of parameters to estimate
        forgetting_factor: Forgetting factor for recursive estimators

    Returns:
        Tuple of (ols_estimates, tls_estimates, ols_history, tls_history)
    """
    rols = RecursiveOLS(n_params=n_params, forgetting_factor=forgetting_factor)
    rtls = RecursiveTLS(n_params=n_params, forgetting_factor=forgetting_factor)

    ols_history = []
    tls_history = []

    for frame_idx, (regressor, ft_sen) in enumerate(zip(regressors, ft_measurements)):
        # Each frame has 6 rows (6 F/T components)
        for row_idx in range(len(ft_sen)):
            rols.update(regressor[row_idx], ft_sen[row_idx])
            rtls.update(regressor[row_idx], ft_sen[row_idx])

        # Record estimates after each frame
        ols_history.append(rols.get_estimate().copy())
        tls_history.append(rtls.get_estimate().copy())

    return (
        rols.get_estimate(),
        rtls.get_estimate(),
        np.array(ols_history),
        np.array(tls_history),
    )


def run_batch_estimation(
    regressors: list,
    ft_measurements: list,
) -> tuple:
    """
    Run batch OLS and TLS estimation on all data.

    Args:
        regressors: List of regressor matrices per frame
        ft_measurements: List of F/T measurements per frame

    Returns:
        Tuple of (ols_estimate, tls_estimate)
    """
    # Stack all data
    A_all = np.vstack(regressors)
    y_all = np.concatenate(ft_measurements)

    ols_estimate = solve_ols_batch(A_all, y_all)
    tls_estimate, min_singular_value = solve_tls_batch(A_all, y_all)

    print(f"Batch data shape: A={A_all.shape}, y={y_all.shape}")
    print(f"TLS minimum singular value: {min_singular_value:.6f}")

    return ols_estimate, tls_estimate


def create_results_dataframe(
    ols_batch: np.ndarray,
    tls_batch: np.ndarray,
    ols_recursive: np.ndarray,
    tls_recursive: np.ndarray,
    n_params: int,
) -> pd.DataFrame:
    """Create a DataFrame with final estimation results."""
    param_names = [f"θ{i + 1}" for i in range(n_params)]
    df = pd.DataFrame(
        {
            "OLS": ols_batch,
            "TLS": tls_batch,
            "ROLS": ols_recursive,
            "RTLS": tls_recursive,
        },
        index=param_names,
    )
    return df


def create_history_dataframe(
    ols_history: np.ndarray,
    tls_history: np.ndarray,
    n_params: int,
) -> tuple:
    """Create DataFrames with convergence history."""
    param_names = [f"θ{i + 1}" for i in range(n_params)]
    ols_df = pd.DataFrame(ols_history, columns=param_names)
    ols_df.index.name = "frame"
    tls_df = pd.DataFrame(tls_history, columns=param_names)
    tls_df.index.name = "frame"
    return ols_df, tls_df


def main():
    print("=" * 70)
    print("Recursive OLS and TLS for Inertial Parameter Estimation")
    print("=" * 70)

    # Load data
    data_path = Path(__file__).parent.parent / "data" / "transforms.json"
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return None, None, None

    regressors, ft_measurements, datetime_str, ground_truth = load_data(str(data_path))
    n_frames = len(regressors)
    n_params = regressors[0].shape[1]

    print(f"\nData loaded from: {data_path}")
    print(f"Recording datetime: {datetime_str}")
    print(f"Number of frames: {n_frames}")
    print(f"Parameters per frame: {n_params}")
    print(f"Measurements per frame: {regressors[0].shape[0]}")

    if ground_truth is not None:
        print(f"\nGround truth (global_gt, first 10 elements):")
        print(f"  {ground_truth}")

    # Run batch estimation
    print("\n" + "-" * 70)
    print("Batch Estimation")
    print("-" * 70)
    ols_batch, tls_batch = run_batch_estimation(regressors, ft_measurements)

    # Run recursive estimation
    print("\n" + "-" * 70)
    print("Recursive Estimation (forgetting_factor=0.99)")
    print("-" * 70)
    ols_recursive, tls_recursive, ols_history, tls_history = run_recursive_estimation(
        regressors, ft_measurements, n_params=n_params, forgetting_factor=0.99
    )

    # Create DataFrames
    results_df = create_results_dataframe(
        ols_batch, tls_batch, ols_recursive, tls_recursive, n_params
    )

    # Add ground truth to results DataFrame
    if ground_truth is not None:
        results_df["Ground_Truth"] = ground_truth

    ols_history_df, tls_history_df = create_history_dataframe(
        ols_history, tls_history, n_params
    )

    # Display final estimates comparison (all methods vs Ground Truth)
    print("\n" + "=" * 70)
    print("Final Estimates Comparison (OLS, TLS, ROLS, RTLS vs Ground Truth)")
    print("=" * 70)
    param_names = [f"θ{i + 1}" for i in range(n_params)]
    final_comparison_df = pd.DataFrame(
        {
            "OLS": ols_batch,
            "TLS": tls_batch,
            "ROLS": ols_recursive,
            "RTLS": tls_recursive,
            "Ground_Truth": ground_truth
            if ground_truth is not None
            else np.full(n_params, np.nan),
        },
        index=param_names,
    )
    print(final_comparison_df.to_string())

    # Display results
    print("\n" + "=" * 70)
    print("Results DataFrame (All Methods)")
    print("=" * 70)
    print(results_df.to_string())

    print("\n" + "=" * 70)
    print("ROLS History DataFrame (last 10 frames)")
    print("=" * 70)
    print(ols_history_df.tail(10).to_string())

    print("\n" + "=" * 70)
    print("RTLS History DataFrame (last 10 frames)")
    print("=" * 70)
    print(tls_history_df.tail(10).to_string())

    # Compare batch vs recursive
    print("\n" + "-" * 70)
    print("Comparison: Batch vs Recursive")
    print("-" * 70)
    ols_diff = np.linalg.norm(ols_batch - ols_recursive)
    tls_diff = np.linalg.norm(tls_batch - tls_recursive)
    print(f"ROLS: ||batch - recursive|| = {ols_diff:.6e}")
    print(f"RTLS: ||batch - recursive|| = {tls_diff:.6e}")

    # Compare against ground truth
    if ground_truth is not None:
        print("\n" + "=" * 70)
        print("Error vs Ground Truth (global_gt)")
        print("=" * 70)

        ols_batch_gt_error = np.linalg.norm(ols_batch - ground_truth)
        tls_batch_gt_error = np.linalg.norm(tls_batch - ground_truth)
        ols_recursive_gt_error = np.linalg.norm(ols_recursive - ground_truth)
        tls_recursive_gt_error = np.linalg.norm(tls_recursive - ground_truth)

        print(f"OLS  vs GT: ||error|| = {ols_batch_gt_error:.6f}")
        print(f"TLS  vs GT: ||error|| = {tls_batch_gt_error:.6f}")
        print(f"ROLS vs GT: ||error|| = {ols_recursive_gt_error:.6f}")
        print(f"RTLS vs GT: ||error|| = {tls_recursive_gt_error:.6f}")

        # Per-parameter error
        print("\nPer-parameter error (estimate - GT):")
        error_df = pd.DataFrame(
            {
                "Ground_Truth": ground_truth,
                "OLS": ols_batch,
                "TLS": tls_batch,
                "ROLS": ols_recursive,
                "RTLS": tls_recursive,
                "OLS_error": ols_batch - ground_truth,
                "TLS_error": tls_batch - ground_truth,
            },
            index=[f"θ{i + 1}" for i in range(n_params)],
        )
        print(error_df.to_string())

    # Save DataFrames to CSV
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_dir / "estimation_results.csv")
    ols_history_df.to_csv(results_dir / "rols_history.csv")
    tls_history_df.to_csv(results_dir / "rtls_history.csv")
    print(f"\nSaved CSVs to: {results_dir}")

    # Plot convergence if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        ax1 = axes[0]
        for i in range(n_params):
            ax1.plot(ols_history[:, i], label=f"θ{i + 1}")
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("Parameter Value")
        ax1.set_title("ROLS Parameter Convergence")
        ax1.legend(loc="upper right", ncol=5, fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        for i in range(n_params):
            ax2.plot(tls_history[:, i], label=f"θ{i + 1}")
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Parameter Value")
        ax2.set_title("RTLS Parameter Convergence")
        ax2.legend(loc="upper right", ncol=5, fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = results_dir / "real_data_convergence.png"
        plt.savefig(output_path, dpi=150)
        print(f"Saved convergence plot to '{output_path}'")
        plt.show()

    except ImportError:
        print("\nNote: Install matplotlib to generate plots")

    return results_df, ols_history_df, tls_history_df


if __name__ == "__main__":
    main()
