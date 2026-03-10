import csv
from pathlib import Path

TRUE_STATE = 0.50
X_D = 0.99
X_I = 0.20
ANCHOR = 0.50
PHI = 1.61803398875

# Optional reliability assumptions for a stronger benchmark
REL_D = 0.35
REL_I = 0.65

REGIMES = {
    "phi_inv2": 1 / (PHI ** 2),
    "phi_inv": 1 / PHI,
    "one": 1.0,
    "phi": PHI,
    "phi2": PHI ** 2,
}


def abs_error(x, truth):
    return abs(x - truth)


def interpretation(x, anchor=ANCHOR):
    if abs(x - anchor) < 1e-12:
        return "exact anchor balance"
    if x > anchor:
        return "leans toward data subsystem"
    return "leans toward intuition subsystem"


# -----------------------------
# Baseline methods
# -----------------------------

def arithmetic_mean(x_d, x_i):
    return 0.5 * (x_d + x_i)


def linear_opinion_pool(x_d, x_i, w_d=0.5):
    w_i = 1.0 - w_d
    return w_d * x_d + w_i * x_i


def logarithmic_opinion_pool(x_d, x_i, w_d=0.5, eps=1e-12):
    """
    Equal-weight logarithmic pooling for probabilities in (0,1).
    """
    x_d = min(max(x_d, eps), 1.0 - eps)
    x_i = min(max(x_i, eps), 1.0 - eps)
    w_i = 1.0 - w_d

    num = (x_d ** w_d) * (x_i ** w_i)
    den = num + ((1.0 - x_d) ** w_d) * ((1.0 - x_i) ** w_i)
    return num / den


def winner_takes_most(x_d, x_i):
    """
    Chooses the more extreme/confident subsystem.
    In this scenario, confidence is proxied by distance from anchor.
    """
    d_conf = abs(x_d - ANCHOR)
    i_conf = abs(x_i - ANCHOR)
    return x_d if d_conf >= i_conf else x_i


def closest_to_anchor_selector(x_d, x_i):
    """
    Chooses the subsystem already closer to the anchor.
    """
    d_dist = abs(x_d - ANCHOR)
    i_dist = abs(x_i - ANCHOR)
    return x_d if d_dist <= i_dist else x_i


def reliability_weighted_pool(x_d, x_i, rel_d=REL_D, rel_i=REL_I):
    total = rel_d + rel_i
    if total <= 0:
        raise ValueError("Reliabilities must sum to a positive value.")
    w_d = rel_d / total
    w_i = rel_i / total
    return w_d * x_d + w_i * x_i


# -----------------------------
# ARA
# -----------------------------

def ara_balance(x_d, x_i, alpha):
    return (x_d + alpha * x_i) / (1.0 + alpha)


# -----------------------------
# Reporting helpers
# -----------------------------

def evaluate_method(method_name, output_value, regime="", alpha=""):
    err_true = abs_error(output_value, TRUE_STATE)
    err_anchor = abs_error(output_value, ANCHOR)

    return {
        "method": method_name,
        "regime": regime,
        "alpha": alpha,
        "x_d": X_D,
        "x_i": X_I,
        "true_state": TRUE_STATE,
        "anchor": ANCHOR,
        "output": round(output_value, 6),
        "abs_error_true_state": round(err_true, 6),
        "abs_error_anchor": round(err_anchor, 6),
        "interpretation": interpretation(output_value),
    }


def print_result_block(result):
    print(f"\nMethod: {result['method']}" if result["regime"] == "" else
          f"\nMethod: {result['method']} | Regime: {result['regime']} (alpha={result['alpha']})")
    print(f"Output = {result['output']:.6f}")
    print(f"|output - true_state| = {result['abs_error_true_state']:.6f}")
    print(f"|output - anchor|     = {result['abs_error_anchor']:.6f}")
    print(f"Interpretation: {result['interpretation']}")


def main():
    out_dir = Path("extended_tests/results/csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "total_signal_divergence_comparison.csv"

    print("\nARA Boundary Benchmark: Extreme Asymmetric Conflict")
    print("=" * 66)
    print(f"True state: {TRUE_STATE:.2f}")
    print(f"Data subsystem (x_d): {X_D:.2f}")
    print(f"Intuition subsystem (x_i): {X_I:.2f}")
    print(f"Anchor: {ANCHOR:.2f}")
    print(f"Reliability assumptions: rel_d={REL_D:.2f}, rel_i={REL_I:.2f}")

    print("\nScenario summary:")
    print("- strong data-side extremity")
    print("- lower but non-symmetric intuition signal")
    print("- neutral truth/anchor used as reference")
    print("- intended to differentiate aggregation behavior under boundary stress")

    rows = []

    # -----------------------------
    # Standard baselines
    # -----------------------------
    print("\nStandard baseline methods")
    print("-" * 66)

    baseline_results = [
        evaluate_method("Arithmetic mean", arithmetic_mean(X_D, X_I)),
        evaluate_method("Linear opinion pool (equal weights)", linear_opinion_pool(X_D, X_I, 0.5)),
        evaluate_method("Logarithmic opinion pool (equal weights)", logarithmic_opinion_pool(X_D, X_I, 0.5)),
        evaluate_method("Winner-takes-most selector", winner_takes_most(X_D, X_I)),
        evaluate_method("Closest-to-anchor selector", closest_to_anchor_selector(X_D, X_I)),
        evaluate_method(
            "Reliability-weighted linear pool",
            reliability_weighted_pool(X_D, X_I, REL_D, REL_I),
        ),
    ]

    for result in baseline_results:
        print_result_block(result)
        rows.append(result)

    # -----------------------------
    # ARA across regimes
    # -----------------------------
    print("\nARA across regimes")
    print("-" * 66)

    ara_results = []
    for regime_name, alpha in REGIMES.items():
        output_value = ara_balance(X_D, X_I, alpha)
        result = evaluate_method("ARA", output_value, regime=regime_name, alpha=round(alpha, 10))
        print_result_block(result)
        rows.append(result)
        ara_results.append(result)

    # -----------------------------
    # Summary
    # -----------------------------
    best_baseline = min(baseline_results, key=lambda r: r["abs_error_true_state"])
    best_ara = min(ara_results, key=lambda r: r["abs_error_true_state"])
    best_overall = min(rows, key=lambda r: r["abs_error_true_state"])

    print("\nSummary")
    print("-" * 66)
    print(
        f"Best baseline: {best_baseline['method']} | "
        f"output={best_baseline['output']:.6f} | "
        f"abs_error={best_baseline['abs_error_true_state']:.6f}"
    )
    print(
        f"Best ARA regime: {best_ara['regime']} | "
        f"output={best_ara['output']:.6f} | "
        f"abs_error={best_ara['abs_error_true_state']:.6f}"
    )
    print(
        f"Best overall performer: {best_overall['method']}"
        + (f" [{best_overall['regime']}]" if best_overall["regime"] else "")
        + f" | output={best_overall['output']:.6f}"
        + f" | abs_error={best_overall['abs_error_true_state']:.6f}"
    )

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
