import csv
from pathlib import Path
import numpy as np

PHI = 1.61803398875

REGIMES = {
    "phi_inv2": 1 / (PHI ** 2),
    "phi_inv": 1 / PHI,
    "one": 1.0,
    "phi": PHI,
    "phi2": PHI ** 2,
}

LOCAL_REGIMES = {
    "v1": 1 / PHI,
    "v2": 1.0,
    "v3": PHI,
    "v4": 1 / (PHI ** 2),
    "v5": PHI ** 2,
}

TRUE_MEANS = {
    "v1": 0.62,
    "v2": 0.55,
    "v3": 0.48,
    "v4": 0.71,
    "v5": 0.39,
}

DATA_MEANS = {
    "v1": 0.88,
    "v2": 0.76,
    "v3": 0.67,
    "v4": 0.91,
    "v5": 0.58,
}

INTUITION_MEANS = {
    "v1": 0.18,
    "v2": 0.34,
    "v3": 0.41,
    "v4": 0.27,
    "v5": 0.22,
}

DATA_SIGMAS = {
    "v1": 0.16,
    "v2": 0.14,
    "v3": 0.18,
    "v4": 0.12,
    "v5": 0.20,
}

INTUITION_SIGMAS = {
    "v1": 0.22,
    "v2": 0.19,
    "v3": 0.17,
    "v4": 0.24,
    "v5": 0.21,
}

N = 20000
SEED = 42


def ara_balance(x_d, x_i, alpha):
    return (x_d + alpha * x_i) / (1.0 + alpha)


def mae(x, truth):
    return float(np.mean(np.abs(x - truth)))


def mse(x, truth):
    return float(np.mean((x - truth) ** 2))


def simulate_variable(rng, mean_d, sigma_d, mean_i, sigma_i, n):
    x_d = np.clip(rng.normal(mean_d, sigma_d, n), 0.0, 1.0)
    x_i = np.clip(rng.normal(mean_i, sigma_i, n), 0.0, 1.0)

    # occasional data shocks
    mask_d = rng.random(n) < 0.15
    x_d[mask_d] = np.clip(
        rng.normal(mean_d - 0.12, sigma_d + 0.05, mask_d.sum()),
        0.0,
        1.0,
    )

    # occasional intuition corrections
    mask_i = rng.random(n) < 0.25
    x_i[mask_i] = np.clip(
        rng.normal(mean_i + 0.10, sigma_i, mask_i.sum()),
        0.0,
        1.0,
    )

    return x_d, x_i


def main():
    rng = np.random.default_rng(SEED)

    variables = list(TRUE_MEANS.keys())

    truth_matrix = {}
    data_matrix = {}
    intuition_matrix = {}

    for var in variables:
        truth_matrix[var] = np.full(N, TRUE_MEANS[var])
        x_d, x_i = simulate_variable(
            rng,
            DATA_MEANS[var],
            DATA_SIGMAS[var],
            INTUITION_MEANS[var],
            INTUITION_SIGMAS[var],
            N,
        )
        data_matrix[var] = x_d
        intuition_matrix[var] = x_i

    # Save into extended_tests/results/csv
    base_dir = Path(__file__).resolve().parents[1]   # extended_tests
    out_dir = base_dir / "results" / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Architecture ARA
    arch_outputs = []
    for var in variables:
        alpha = LOCAL_REGIMES[var]
        xb = ara_balance(data_matrix[var], intuition_matrix[var], alpha)
        arch_outputs.append(xb)

    arch_outputs = np.vstack(arch_outputs)
    truth_all = np.vstack([truth_matrix[v] for v in variables])

    architecture_ara = np.mean(arch_outputs, axis=0)
    architecture_truth = np.mean(truth_all, axis=0)

    # 2) Global ARA regimes
    global_results = []
    for regime, alpha in REGIMES.items():
        per_var = []
        for var in variables:
            xb = ara_balance(data_matrix[var], intuition_matrix[var], alpha)
            per_var.append(xb)

        per_var = np.vstack(per_var)
        output = np.mean(per_var, axis=0)

        global_results.append({
            "method": f"Global ARA ({regime})",
            "alpha": alpha,
            "mean_output": float(np.mean(output)),
            "mae_vs_truth": mae(output, architecture_truth),
            "mse_vs_truth": mse(output, architecture_truth),
        })

    # 3) Baselines
    data_only = np.mean(np.vstack([data_matrix[v] for v in variables]), axis=0)
    intuition_only = np.mean(np.vstack([intuition_matrix[v] for v in variables]), axis=0)

    arithmetic_mean_per_var = []
    linear_pool_per_var = []
    log_pool_per_var = []

    eps = 1e-9
    for var in variables:
        x_d = data_matrix[var]
        x_i = intuition_matrix[var]

        arithmetic_mean_per_var.append((x_d + x_i) / 2.0)
        linear_pool_per_var.append((x_d + x_i) / 2.0)
        log_pool_per_var.append(np.exp((np.log(x_d + eps) + np.log(x_i + eps)) / 2.0))

    arithmetic_mean = np.mean(np.vstack(arithmetic_mean_per_var), axis=0)
    linear_pool = np.mean(np.vstack(linear_pool_per_var), axis=0)
    log_pool = np.mean(np.vstack(log_pool_per_var), axis=0)

    comparison_rows = [
        {
            "method": "Architecture ARA",
            "alpha": "local",
            "mean_output": float(np.mean(architecture_ara)),
            "mae_vs_truth": mae(architecture_ara, architecture_truth),
            "mse_vs_truth": mse(architecture_ara, architecture_truth),
        },
        *global_results,
        {
            "method": "Arithmetic mean",
            "alpha": "",
            "mean_output": float(np.mean(arithmetic_mean)),
            "mae_vs_truth": mae(arithmetic_mean, architecture_truth),
            "mse_vs_truth": mse(arithmetic_mean, architecture_truth),
        },
        {
            "method": "Linear opinion pool",
            "alpha": "",
            "mean_output": float(np.mean(linear_pool)),
            "mae_vs_truth": mae(linear_pool, architecture_truth),
            "mse_vs_truth": mse(linear_pool, architecture_truth),
        },
        {
            "method": "Log opinion pool",
            "alpha": "",
            "mean_output": float(np.mean(log_pool)),
            "mae_vs_truth": mae(log_pool, architecture_truth),
            "mse_vs_truth": mse(log_pool, architecture_truth),
        },
        {
            "method": "Data only",
            "alpha": "",
            "mean_output": float(np.mean(data_only)),
            "mae_vs_truth": mae(data_only, architecture_truth),
            "mse_vs_truth": mse(data_only, architecture_truth),
        },
        {
            "method": "Intuition only",
            "alpha": "",
            "mean_output": float(np.mean(intuition_only)),
            "mae_vs_truth": mae(intuition_only, architecture_truth),
            "mse_vs_truth": mse(intuition_only, architecture_truth),
        },
    ]

    with open(out_dir / "test2_noisy_architecture_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "alpha",
                "mean_output",
                "mae_vs_truth",
                "mse_vs_truth",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    # 4) Variable-level architecture diagnostics
    variable_rows = []
    for var in variables:
        alpha = LOCAL_REGIMES[var]
        xb = ara_balance(data_matrix[var], intuition_matrix[var], alpha)
        xt = truth_matrix[var]

        variable_rows.append({
            "variable": var,
            "local_alpha": alpha,
            "truth_mean": float(np.mean(xt)),
            "data_mean": float(np.mean(data_matrix[var])),
            "intuition_mean": float(np.mean(intuition_matrix[var])),
            "architecture_output_mean": float(np.mean(xb)),
            "mae_vs_truth": mae(xb, xt),
            "mse_vs_truth": mse(xb, xt),
        })

    with open(out_dir / "test2_noisy_architecture_variables.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variable",
                "local_alpha",
                "truth_mean",
                "data_mean",
                "intuition_mean",
                "architecture_output_mean",
                "mae_vs_truth",
                "mse_vs_truth",
            ],
        )
        writer.writeheader()
        writer.writerows(variable_rows)

    print(f"Saved: {out_dir / 'test2_noisy_architecture_comparison.csv'}")
    print(f"Saved: {out_dir / 'test2_noisy_architecture_variables.csv'}")


if __name__ == "__main__":
    main()
