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

# Test scenario with multiple variables
# Each variable has its own architectural regime
VARIABLES = [
    {"name": "v1", "x_d": 0.95, "x_i": 0.05, "regime": "phi"},
    {"name": "v2", "x_d": 0.80, "x_i": 0.30, "regime": "one"},
    {"name": "v3", "x_d": 0.60, "x_i": 0.40, "regime": "phi_inv"},
    {"name": "v4", "x_d": 0.70, "x_i": 0.20, "regime": "phi2"},
    {"name": "v5", "x_d": 0.90, "x_i": 0.10, "regime": "phi_inv2"},
]

TRUE_STATE = 0.50
ANCHOR = 0.50


def ara_balance(x_d, x_i, alpha):
    return (x_d + alpha * x_i) / (1 + alpha)


def abs_error(x, truth):
    return abs(x - truth)


def main():

    out_dir = Path("extended_tests/results/csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "ara_architecture_weighted_test.csv"

    rows = []

    print("\nARA Architecture-Weighted Test")
    print("=" * 40)

    total = 0.0

    for v in VARIABLES:

        regime = v["regime"]
        alpha = REGIMES[regime]

        x_b = ara_balance(v["x_d"], v["x_i"], alpha)

        print(f"\nVariable: {v['name']}")
        print(f"Regime: {regime} (alpha={alpha:.6f})")
        print(f"x_d = {v['x_d']}")
        print(f"x_i = {v['x_i']}")
        print(f"x_b = {x_b:.6f}")

        total += x_b

        rows.append({
            "variable": v["name"],
            "regime": regime,
            "alpha": alpha,
            "x_d": v["x_d"],
            "x_i": v["x_i"],
            "x_b": round(x_b, 6),
        })

    X_b = total / len(VARIABLES)

    error = abs_error(X_b, TRUE_STATE)

    print("\nGlobal balanced output")
    print("-" * 40)
    print(f"X_b = {X_b:.6f}")
    print(f"|X_b - true_state| = {error:.6f}")

    rows.append({
        "variable": "GLOBAL",
        "regime": "mixed",
        "alpha": "",
        "x_d": "",
        "x_i": "",
        "x_b": round(X_b, 6),
    })

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variable",
                "regime",
                "alpha",
                "x_d",
                "x_i",
                "x_b",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
