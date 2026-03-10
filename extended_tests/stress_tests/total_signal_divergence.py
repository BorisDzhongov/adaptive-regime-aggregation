import csv
from pathlib import Path

TRUE_STATE = 0.5
X_D = 0.95
X_I = 0.05
ANCHOR = 0.5
PHI = 1.61803398875

REGIMES = {
    "phi_inv2": 1 / (PHI ** 2),
    "phi_inv": 1 / PHI,
    "one": 1.0,
    "phi": PHI,
    "phi2": PHI ** 2,
}


def ara_balance(x_d, x_i, alpha):
    return (x_d + alpha * x_i) / (1.0 + alpha)


def abs_error(x, truth):
    return abs(x - truth)


def main():
    out_dir = Path("extended_tests/results/csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "total_signal_divergence.csv"

    rows = []

    print("\nARA Total Signal Divergence Test")
    print("=" * 40)
    print(f"True state: {TRUE_STATE:.2f}")
    print(f"Data subsystem (x_d): {X_D:.2f}")
    print(f"Intuition subsystem (x_i): {X_I:.2f}")
    print(f"Anchor: {ANCHOR:.2f}")

    for regime_name, alpha in REGIMES.items():
        x_b = ara_balance(X_D, X_I, alpha)
        error = abs_error(x_b, TRUE_STATE)
        anchor_error = abs_error(x_b, ANCHOR)

        print(f"\nRegime: {regime_name} (alpha={alpha:.6f})")
        print(f"Balanced output x_b = {x_b:.6f}")
        print(f"|x_b - true_state| = {error:.6f}")
        print(f"|x_b - anchor|     = {anchor_error:.6f}")

        interpretation = (
            "exact neutral balance"
            if abs(x_b - 0.5) < 1e-12
            else "leans toward data subsystem"
            if x_b > 0.5
            else "leans toward intuition subsystem"
        )
        print(f"Interpretation: {interpretation}")

        rows.append({
            "regime": regime_name,
            "alpha": round(alpha, 10),
            "x_d": X_D,
            "x_i": X_I,
            "true_state": TRUE_STATE,
            "anchor": ANCHOR,
            "x_b": round(x_b, 6),
            "abs_error_true_state": round(error, 6),
            "abs_error_anchor": round(anchor_error, 6),
            "interpretation": interpretation,
        })

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "regime",
                "alpha",
                "x_d",
                "x_i",
                "true_state",
                "anchor",
                "x_b",
                "abs_error_true_state",
                "abs_error_anchor",
                "interpretation",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
