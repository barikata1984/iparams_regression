#!/usr/bin/env python3
"""
Test script to verify the RecursiveOLS and RecursiveTLS implementations
against the reference data in transforms.json.

This script loads pre-computed regressor matrices and force/torque measurements,
runs both batch and recursive estimation, and compares results with expected values.

Usage:
    rosrun iparams_regression test_estimators.py
    OR
    python3 test_estimators.py
"""

import json
import numpy as np
from pathlib import Path

# Try to import from the package (when run via rosrun)
try:
    from iparams_regression import (
        RecursiveOLS,
        RecursiveTLS,
        solve_ols_batch,
        solve_tls_batch,
        InertialParameters,
    )
except ImportError:
    # Fallback for direct execution
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from iparams_regression import (
        RecursiveOLS,
        RecursiveTLS,
        solve_ols_batch,
        solve_tls_batch,
        InertialParameters,
    )


def load_test_data(json_path: str):
    """Load test data from JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    regressors = []
    ft_measurements = []

    for frame in data["frames"]:
        regressor = np.array(frame["regressor"])  # (6, 10)
        ft_sen = np.array(frame["ft_sen"])  # (6,)
        regressors.append(regressor)
        ft_measurements.append(ft_sen)

    ground_truth = None
    if "global_gt" in data:
        ground_truth = np.array(data["global_gt"][:10])

    expected_ls = None
    if "ls" in data:
        expected_ls = np.array(data["ls"][:10])

    expected_tls = None
    if "tls" in data:
        expected_tls = np.array(data["tls"][:10])

    return regressors, ft_measurements, ground_truth, expected_ls, expected_tls


def run_batch_estimation(regressors, ft_measurements):
    """Run batch OLS and TLS estimation."""
    # Stack all data
    A_list = []
    y_list = []

    for regressor, ft_sen in zip(regressors, ft_measurements):
        for i in range(6):
            A_list.append(regressor[i])
            y_list.append(ft_sen[i])

    A = np.array(A_list)  # (n_samples * 6, 10)
    y = np.array(y_list)  # (n_samples * 6,)

    # Batch OLS
    ols_estimate = solve_ols_batch(A, y)

    # Batch TLS
    tls_estimate, _ = solve_tls_batch(A, y)

    return ols_estimate, tls_estimate


def run_recursive_estimation(regressors, ft_measurements, forgetting_factor=1.0):
    """Run recursive OLS and TLS estimation."""
    n_params = 10

    rols = RecursiveOLS(n_params=n_params, forgetting_factor=forgetting_factor)
    rtls = RecursiveTLS(n_params=n_params, forgetting_factor=forgetting_factor)

    for regressor, ft_sen in zip(regressors, ft_measurements):
        for i in range(6):
            rols.update(regressor[i], ft_sen[i])
            rtls.update(regressor[i], ft_sen[i])

    return rols.get_estimate(), rtls.get_estimate()


def format_params(params: np.ndarray, name: str) -> str:
    """Format parameters for display."""
    param_names = [
        "mass",
        "m*cx",
        "m*cy",
        "m*cz",
        "Ixx",
        "Ixy",
        "Ixz",
        "Iyy",
        "Iyz",
        "Izz",
    ]
    lines = [f"{name}:"]
    for i, (pname, val) in enumerate(zip(param_names, params)):
        lines.append(f"  θ{i + 1:2d} ({pname:5s}): {val:12.6f}")
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("Inertial Parameter Estimation - Verification Test")
    print("=" * 60)

    # Find data file
    script_dir = Path(__file__).parent
    data_paths = [
        script_dir.parent / "data" / "transforms.json",
        script_dir.parent.parent / "data" / "transforms.json",
        Path(
            "/home/ak/osx_ose_for_learning_manipulation/catkin_ws/src/iparams_regression/data/transforms.json"
        ),
    ]

    data_path = None
    for p in data_paths:
        if p.exists():
            data_path = p
            break

    if data_path is None:
        print("ERROR: Could not find transforms.json")
        return 1

    print(f"Loading data from: {data_path}")

    # Load data
    regressors, ft_measurements, ground_truth, expected_ls, expected_tls = (
        load_test_data(str(data_path))
    )

    n_frames = len(regressors)
    print(f"Loaded {n_frames} frames")

    # Run batch estimation
    print("\n" + "-" * 60)
    print("Running Batch Estimation...")
    print("-" * 60)

    batch_ols, batch_tls = run_batch_estimation(regressors, ft_measurements)

    print("\n" + format_params(batch_ols, "Batch OLS"))
    print("\n" + format_params(batch_tls, "Batch TLS"))

    # Run recursive estimation
    print("\n" + "-" * 60)
    print("Running Recursive Estimation (λ=1.0)...")
    print("-" * 60)

    recursive_ols, recursive_tls = run_recursive_estimation(
        regressors, ft_measurements, forgetting_factor=1.0
    )

    print("\n" + format_params(recursive_ols, "Recursive OLS"))
    print("\n" + format_params(recursive_tls, "Recursive TLS"))

    # Compare with expected values
    print("\n" + "=" * 60)
    print("Comparison with Ground Truth and Expected Values")
    print("=" * 60)

    if ground_truth is not None:
        print("\n" + format_params(ground_truth, "Ground Truth"))

    if expected_ls is not None:
        print("\n" + format_params(expected_ls, "Expected LS (from JSON)"))
        ols_diff = np.linalg.norm(batch_ols - expected_ls)
        print(f"\n  Batch OLS vs Expected LS:     ||diff|| = {ols_diff:.6e}")

        rols_diff = np.linalg.norm(recursive_ols - expected_ls)
        print(f"  Recursive OLS vs Expected LS: ||diff|| = {rols_diff:.6e}")

    if expected_tls is not None:
        print("\n" + format_params(expected_tls, "Expected TLS (from JSON)"))
        tls_diff = np.linalg.norm(batch_tls - expected_tls)
        print(f"\n  Batch TLS vs Expected TLS:     ||diff|| = {tls_diff:.6e}")

        rtls_diff = np.linalg.norm(recursive_tls - expected_tls)
        print(f"  Recursive TLS vs Expected TLS: ||diff|| = {rtls_diff:.6e}")

    if ground_truth is not None:
        print("\n" + "-" * 60)
        print("Error vs Ground Truth:")
        print("-" * 60)

        for name, estimate in [
            ("Batch OLS", batch_ols),
            ("Batch TLS", batch_tls),
            ("Recursive OLS", recursive_ols),
            ("Recursive TLS", recursive_tls),
        ]:
            error = np.linalg.norm(estimate - ground_truth)
            print(f"  {name:20s}: ||error|| = {error:.6f}")

    # Verify batch and recursive results match
    print("\n" + "-" * 60)
    print("Batch vs Recursive Consistency Check:")
    print("-" * 60)

    ols_consistency = np.linalg.norm(batch_ols - recursive_ols)
    tls_consistency = np.linalg.norm(batch_tls - recursive_tls)

    print(f"  OLS: ||batch - recursive|| = {ols_consistency:.6e}")
    print(f"  TLS: ||batch - recursive|| = {tls_consistency:.6e}")

    if ols_consistency < 1e-6 and tls_consistency < 1e-6:
        print("\n  ✓ Batch and Recursive implementations are consistent!")
    else:
        print("\n  ✗ WARNING: Batch and Recursive implementations differ!")

    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
