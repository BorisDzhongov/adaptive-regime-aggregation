import csv
from pathlib import Path

PHI = 1.61803398875

REGIMES = {
    "phi_inv2": 1 / (PHI ** 2),
    "phi_inv": 1 / PHI,
    "one": 1.0,
    "phi": PHI,
    "phi2": PHI ** 2,
}

TRUE_STATE = 0.5
ANCHOR = 0.5

# Example architecture
VARIABLES = [
    {"name": "v1", "x_d": 0.95, "x_i": 0.05, "regime": "phi"},
    {"name": "v2", "x_d": 0.80, "x_i": 0.30, "regime": "one"},
    {"name": "v3", "x_d": 0.60, "x_i": 0.40, "regime": "phi_inv"},
    {"name": "v4", "x_d": 0.70, "x_i": 0.20, "regime": "phi2"},
    {"name": "v5", "x_d": 0.90, "x_i": 0.10, "regime": "phi_inv2"},
]


def ara_balance(x_d, x_i, alpha):
    return (x_d + alpha * x_i) / (1 + alpha)


def abs_error(x, truth):
    return abs(x - truth)


def architecture_ara():
    total = 0
    for v in VARIABLES:
        alpha = REGIMES[v["regime"]]
        x_b = ara_balance(v["x_d"], v["x_i"], alpha)
        total += x_b
    return total / len(VARIABLES)


def arithmetic_mean():
    total = 0
    for v in VARIABLES:
        total += (v["x_d"] + v["x_i"]) / 2
    return total / len(VARIABLES)


def data_only():
    total = 0
    for v in VARIABLES:
        total += v["x_d"]
    return total / len(VARIABLES)


def intuition_only():
    total = 0
    for v in VARIABLES:
        total += v["x_i"]
    return total / len(VARIABLES)


def global_ara(alpha):
    total = 0
    for v in VARIABLES:
        total += ara_balance(v["x_d"], v["x_i"], alpha)
    return total / len(VARIABLES)


def main():

    out_dir = Path("extended_tests/results/csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "ara_architecture_regime_comparison.csv"

    results = []

    tests = {
        "Architecture ARA": architecture_ara(),
        "Arithmetic mean": arithmetic_mean(),
        "Data only": data_only(),
        "Intuition only": intuition_only(),
        "Global ARA (φ)": global_ara(PHI),
        "Global ARA (1)": global_ara(1),
    }

    print("\nARA Architecture Regime Comparison")
    print("=" * 40)

    for name, value in tests.items():

        error = abs_error(value, TRUE_STATE)

        print(f"\nMethod: {name}")
        print(f"Output = {value:.6f}")
        print(f"|output - true_state| = {error:.6f}")

        results.append({
            "method": name,
            "output": round(value, 6),
            "abs_error_true_state": round(error, 6)
        })

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "output",
                "abs_error_true_state",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
