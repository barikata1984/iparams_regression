#!/usr/bin/env python3
"""
Example: Comparison of OLS and TLS for Inertial Parameter Estimation

This script demonstrates the difference between Ordinary Least Squares (OLS)
and Total Least Squares (TLS) for estimating inertial parameters in the
presence of noise in both the regressor matrix and measurements.

Usage:
    python -m src.example_simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from .recursive_ols import RecursiveOLS, solve_ols_batch
from .recursive_tls import RecursiveTLS, solve_tls_batch


def generate_simple_example(
    n_samples: int = 200,
    noise_output: float = 0.1,
    noise_input: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a simple linear regression example with noise in both input and output.

    Model: y = A @ x_true + noise_y
           A_noisy = A + noise_A

    Args:
        n_samples: Number of samples
        noise_output: Standard deviation of output noise
        noise_input: Standard deviation of input (regressor) noise
        seed: Random seed

    Returns:
        Tuple of (A_noisy, y_noisy, A_true, x_true)
    """
    np.random.seed(seed)

    # True parameters
    x_true = np.array([2.5, -1.3, 0.8])
    n_params = len(x_true)

    # Generate true regressor matrix (input data)
    t = np.linspace(0, 4 * np.pi, n_samples)
    A_true = np.column_stack(
        [
            np.sin(t),
            np.cos(2 * t),
            np.ones(n_samples),
        ]
    )

    # True output
    y_true = A_true @ x_true

    # Add noise to output
    y_noisy = y_true + noise_output * np.random.randn(n_samples)

    # Add noise to input (regressor matrix)
    A_noisy = A_true + noise_input * np.random.randn(n_samples, n_params)

    return A_noisy, y_noisy, A_true, x_true


def run_comparison():
    """
    Run a comparison between OLS and TLS with varying input noise levels.
    """
    print("=" * 60)
    print("Comparison of OLS and TLS for Parameter Estimation")
    print("=" * 60)

    n_samples = 500
    output_noise = 0.1
    input_noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

    results = {
        "input_noise": [],
        "ols_error": [],
        "tls_error": [],
        "ols_batch_error": [],
        "tls_batch_error": [],
    }

    for input_noise in input_noise_levels:
        print(f"\n--- Input Noise σ = {input_noise:.2f} ---")

        A_noisy, y_noisy, A_true, x_true = generate_simple_example(
            n_samples=n_samples,
            noise_output=output_noise,
            noise_input=input_noise,
        )

        n_params = len(x_true)

        # Batch OLS
        x_ols_batch = solve_ols_batch(A_noisy, y_noisy)
        ols_batch_error = np.linalg.norm(x_ols_batch - x_true)

        # Batch TLS
        x_tls_batch, _ = solve_tls_batch(A_noisy, y_noisy)
        tls_batch_error = np.linalg.norm(x_tls_batch - x_true)

        # Recursive OLS
        rols = RecursiveOLS(n_params=n_params, forgetting_factor=1.0)
        for i in range(n_samples):
            rols.update(A_noisy[i], y_noisy[i])
        x_ols_recursive = rols.get_estimate()
        ols_error = np.linalg.norm(x_ols_recursive - x_true)

        # Recursive TLS
        rtls = RecursiveTLS(n_params=n_params, forgetting_factor=1.0)
        for i in range(n_samples):
            rtls.update(A_noisy[i], y_noisy[i])
        x_tls_recursive = rtls.get_estimate()
        tls_error = np.linalg.norm(x_tls_recursive - x_true)

        print(f"True parameters:      {x_true}")
        print(f"OLS (batch):          {x_ols_batch} (error: {ols_batch_error:.4f})")
        print(f"TLS (batch):          {x_tls_batch} (error: {tls_batch_error:.4f})")
        print(f"OLS (recursive):      {x_ols_recursive} (error: {ols_error:.4f})")
        print(f"TLS (recursive):      {x_tls_recursive} (error: {tls_error:.4f})")

        results["input_noise"].append(input_noise)
        results["ols_error"].append(ols_error)
        results["tls_error"].append(tls_error)
        results["ols_batch_error"].append(ols_batch_error)
        results["tls_batch_error"].append(tls_batch_error)

    return results


def plot_convergence():
    """
    Plot the convergence of recursive OLS and TLS estimators.
    """
    print("\n" + "=" * 60)
    print("Convergence Analysis")
    print("=" * 60)

    n_samples = 300
    input_noise = 0.2
    output_noise = 0.1

    A_noisy, y_noisy, A_true, x_true = generate_simple_example(
        n_samples=n_samples,
        noise_output=output_noise,
        noise_input=input_noise,
    )

    n_params = len(x_true)

    # Track convergence
    ols_history = []
    tls_history = []

    rols = RecursiveOLS(n_params=n_params, forgetting_factor=1.0)
    rtls = RecursiveTLS(n_params=n_params, forgetting_factor=1.0)

    for i in range(n_samples):
        rols.update(A_noisy[i], y_noisy[i])
        rtls.update(A_noisy[i], y_noisy[i])

        ols_history.append(rols.get_estimate().copy())
        tls_history.append(rtls.get_estimate().copy())

    ols_history = np.array(ols_history)
    tls_history = np.array(tls_history)

    # Compute errors over time
    ols_errors = np.linalg.norm(ols_history - x_true, axis=1)
    tls_errors = np.linalg.norm(tls_history - x_true, axis=1)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Parameter convergence
    ax1 = axes[0]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for j in range(n_params):
        ax1.plot(
            ols_history[:, j], "--", color=colors[j], alpha=0.7, label=f"OLS θ{j + 1}"
        )
        ax1.plot(tls_history[:, j], "-", color=colors[j], label=f"TLS θ{j + 1}")
        ax1.axhline(y=x_true[j], color=colors[j], linestyle=":", alpha=0.5)

    ax1.set_xlabel("Sample")
    ax1.set_ylabel("Parameter Value")
    ax1.set_title(f"Parameter Convergence (Input Noise σ={input_noise})")
    ax1.legend(loc="upper right", ncol=2)
    ax1.grid(True, alpha=0.3)

    # Error convergence
    ax2 = axes[1]
    ax2.semilogy(ols_errors, "--", label="OLS Error", linewidth=2)
    ax2.semilogy(tls_errors, "-", label="TLS Error", linewidth=2)
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Parameter Error (log scale)")
    ax2.set_title("Estimation Error Over Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("convergence_comparison.png", dpi=150)
    print("\nSaved convergence plot to 'convergence_comparison.png'")
    plt.show()


def plot_noise_sensitivity():
    """
    Plot how OLS and TLS errors change with input noise level.
    """
    results = run_comparison()

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        results["input_noise"],
        results["ols_batch_error"],
        "o--",
        label="OLS (batch)",
        linewidth=2,
        markersize=8,
    )
    ax.plot(
        results["input_noise"],
        results["tls_batch_error"],
        "s-",
        label="TLS (batch)",
        linewidth=2,
        markersize=8,
    )
    ax.plot(
        results["input_noise"],
        results["ols_error"],
        "^--",
        label="OLS (recursive)",
        linewidth=2,
        markersize=8,
        alpha=0.7,
    )
    ax.plot(
        results["input_noise"],
        results["tls_error"],
        "v-",
        label="TLS (recursive)",
        linewidth=2,
        markersize=8,
        alpha=0.7,
    )

    ax.set_xlabel("Input Noise Standard Deviation")
    ax.set_ylabel("Parameter Estimation Error")
    ax.set_title("OLS vs TLS: Sensitivity to Input Noise")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("noise_sensitivity.png", dpi=150)
    print("\nSaved noise sensitivity plot to 'noise_sensitivity.png'")
    plt.show()


if __name__ == "__main__":
    # Run comparison
    run_comparison()

    try:
        import importlib.util

        if importlib.util.find_spec("matplotlib") is not None:
            print("\n" + "=" * 60)
            print("Generating plots...")
            print("=" * 60)
            plot_convergence()
            plot_noise_sensitivity()
        else:
            print("\nNote: Install matplotlib to generate plots")
            print("  pip install matplotlib")
    except Exception as e:
        print(f"\nNote: Could not generate plots: {e}")
