import csv
import math
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


def arithmetic_mean(x_d, x_i):
    return 0.5 * (x_d + x_i)


def linear_opinion_pool(x_d, x_i, w_d=0.5):
    w_i = 1.0 - w_d
    return w_d * x_d + w_i * x_i


def log_pool(x_d, x_i, w_d=0.5, eps=1e-12):
    """
    Equal-weight logarithmic pooling for probabilities in (0,1).
    """
    x_d = min(max(x_d, eps), 1.0 - eps)
    x_i = min(max(x_i, eps), 1.0 - eps)
    w_i = 1.0 - w_d

    num = (x_d ** w_d) * (x_i ** w_i)
    den = num + ((1.0 - x_d) ** w_d) * ((1.0 - x_i) ** w_i)
    return num / den


def abs_error(x, truth):
    return abs(x - truth)


def interpretation(x):
    if abs(x - 0.5) < 1e-12:
        return "exact neutral balance"
    if x > 0.5:
        return "leans toward data subsystem"
    return "leans toward intuition subsystem"


def print_method_block(method_name, output_value):
    err = abs_error(output_value, TRUE_STATE)
    anch = abs_error(output_value, ANCHOR)

    print(f"\nMethod: {method_name}")
    print(f"Output = {output_value:.6f}")
    print(f"|output - true_state| = {err:.6f}")
    print(f"|output - anchor|     = {anch:.6f}")
    print(f"Interpretation: {interpretation(output_value)}")


def main():
    out_dir = Path("extended_tests/results/csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "total_signal_divergence_comparison.csv"

    rows = []

    print("\nTotal Signal Divergence Comparison Test")
    print("=" * 50)
    print(f"True state: {TRUE_STATE:.2f}")
    print(f"Data subsystem (x_d): {X_D:.2f}")
    print(f"Intuition subsystem (x_i): {X_I:.2f}")
    print(f"Anchor: {ANCHOR:.2f}")

    # Baseline methods
    mean_val = arithmetic_mean(X_D, X_I)
    lop_val = linear_opinion_pool(X_D, X_I, w_d=0.5)
    log_val = log_pool(X_D, X_I, w_d=0.5)

    print("\nBaseline methods")
    print("-" * 50)
    print_method_block("Arithmetic mean", mean_val)
    print_method_block("Linear opinion pool (equal weights)", lop_val)
    print_method_block("Logarithmic pool (equal weights)", log_val)

    for method_name, output_value, alpha in [
        ("Arithmetic mean", mean_val, None),
        ("Linear opinion pool (equal weights)", lop_val, None),
        ("Logarithmic pool (equal weights)", log_val, None),
    ]:
        rows.append({
            "method": method_name,
            "regime": "baseline",
            "alpha": "",
            "x_d": X_D,
            "x_i": X_I,
            "true_state": TRUE_STATE,
            "anchor": ANCHOR,
            "output": round(output_value, 6),
            "abs_error_true_state": round(abs_error(output_value, TRUE_STATE), 6),
            "abs_error_anchor": round(abs_error(output_value, ANCHOR), 6),
            "interpretation": interpretation(output_value),
        })

    # ARA across regimes
    print("\nARA across regimes")
    print("-" * 50)

    for regime_name, alpha in REGIMES.items():
        x_b = ara_balance(X_D, X_I, alpha)

        print(f"\nRegime: {regime_name} (alpha={alpha:.6f})")
        print(f"Output = {x_b:.6f}")
        print(f"|output - true_state| = {abs_error(x_b, TRUE_STATE):.6f}")
        print(f"|output - anchor|     = {abs_error(x_b, ANCHOR):.6f}")
        print(f"Interpretation: {interpretation(x_b)}")

        rows.append({
            "method": "ARA",
            "regime": regime_name,
            "alpha": round(alpha, 10),
            "x_d": X_D,
            "x_i": X_I,
            "true_state": TRUE_STATE,
            "anchor": ANCHOR,
            "output": round(x_b, 6),
            "abs_error_true_state": round(abs_error(x_b, TRUE_STATE), 6),
            "abs_error_anchor": round(abs_error(x_b, ANCHOR), 6),
            "interpretation": interpretation(x_b),
        })

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "regime",
                "alpha",
                "x_d",
                "x_i",
                "true_state",
                "anchor",
                "output",
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
