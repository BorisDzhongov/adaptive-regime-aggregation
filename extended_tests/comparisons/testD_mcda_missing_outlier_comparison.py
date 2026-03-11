import numpy as np
import csv
import os

PHI = (1 + 5 ** 0.5) / 2
PHI_INV = 1.0 / PHI
PHI2 = PHI ** 2
PHI_INV2 = 1.0 / (PHI ** 2)

L = 5.0
PROJECTS = ["A", "B", "C", "D"]

G = np.array([
    [8.5, 7.0, 9.0, 6.5],
    [7.5, 8.5, 6.5, 6.2],
    [6.0, 8.1, 5.5, 8.0],
    [6.5, 7.0, 5.0, 8.5],
    [6.0, 7.5, 8.5, 5.8],
    [7.0, 6.0, 8.0, 6.5],
    [6.5, 7.5, 6.0, 8.0],
    [5.5, 6.5, 4.5, 7.5],
    [5.5, 6.5, 4.0, 7.0],
    [6.0, 6.5, 5.0, 7.5],
    [7.0, 6.5, 8.5, 6.0],
], dtype=float)

I = np.array([
    [8.0, 6.5, 9.5, 6.0],
    [7.0, 9.0, 6.0, 7.0],
    [5.0, 6.5, 4.5, 7.5],
    [6.0, 7.5, 4.0, 8.0],
    [5.5, 7.0, 8.0, 6.0],
    [6.5, 5.0, 7.0, 6.0],
    [6.0, 7.0, 5.5, 7.5],
    [4.0, 5.5, 3.0, 7.0],
    [4.5, 6.0, 3.0, 6.5],
    [5.0, 6.0, 4.0, 7.0],
    [6.5, 6.0, 9.0, 5.5],
], dtype=float)

M, N_PROJECTS = G.shape


def clip010(x):
    return np.clip(x, 0.0, 10.0)


def fill_anchor(x, mask, anchor=L):
    y = x.copy()
    y[mask] = anchor
    return y


def aggregate_mean(g, i):
    return 0.5 * (g + i)


def score_mean(x):
    return x.mean(axis=0)


def score_topsis(x):
    # x shape: (criteria, projects, mc)
    denom = np.sqrt(np.sum(x ** 2, axis=1, keepdims=True))
    denom = np.where(denom == 0.0, 1.0, denom)

    r = x / denom
    v = r / M

    ideal_best = np.max(v, axis=1, keepdims=True)
    ideal_worst = np.min(v, axis=1, keepdims=True)

    d_pos = np.sqrt(np.sum((v - ideal_best) ** 2, axis=0))
    d_neg = np.sqrt(np.sum((v - ideal_worst) ** 2, axis=0))

    return d_neg / (d_pos + d_neg + 1e-12)


def score_promethee_ii(x):
    # x shape: (criteria, projects, mc)
    minv = np.min(x, axis=1, keepdims=True)
    maxv = np.max(x, axis=1, keepdims=True)
    rng = np.where((maxv - minv) == 0.0, 1.0, (maxv - minv))
    xn = (x - minv) / rng

    flows = np.zeros((N_PROJECTS, x.shape[2]), dtype=float)

    for a in range(N_PROJECTS):
        phi_plus = np.zeros(x.shape[2], dtype=float)
        phi_minus = np.zeros(x.shape[2], dtype=float)

        for b in range(N_PROJECTS):
            if a == b:
                continue

            diff_ab = xn[:, a, :] - xn[:, b, :]
            pref_ab = np.mean(np.maximum(diff_ab, 0.0), axis=0)

            diff_ba = xn[:, b, :] - xn[:, a, :]
            pref_ba = np.mean(np.maximum(diff_ba, 0.0), axis=0)

            phi_plus += pref_ab
            phi_minus += pref_ba

        flows[a, :] = (phi_plus - phi_minus) / (N_PROJECTS - 1)

    return flows


def score_electre_i(x, concordance_threshold=0.60, discordance_threshold=0.40):
    """
    Simplified ELECTRE I-style outranking score.
    Returns net outranking flow per project per MC run.
    x shape: (criteria, projects, mc)
    """
    minv = np.min(x, axis=1, keepdims=True)
    maxv = np.max(x, axis=1, keepdims=True)
    rng = np.where((maxv - minv) == 0.0, 1.0, (maxv - minv))
    xn = (x - minv) / rng

    mc = x.shape[2]
    outrank = np.zeros((N_PROJECTS, N_PROJECTS, mc), dtype=float)

    for a in range(N_PROJECTS):
        for b in range(N_PROJECTS):
            if a == b:
                continue

            diff = xn[:, a, :] - xn[:, b, :]

            concordance = np.mean(diff >= 0.0, axis=0)
            discordance = np.max(np.maximum(-diff, 0.0), axis=0)

            outrank[a, b, :] = (
                (concordance >= concordance_threshold) &
                (discordance <= discordance_threshold)
            ).astype(float)

    phi_plus = np.sum(outrank, axis=1) / (N_PROJECTS - 1)
    phi_minus = np.sum(outrank, axis=0) / (N_PROJECTS - 1)

    return phi_plus - phi_minus


def winners(scores):
    return np.argmax(scores, axis=0)


def winner_entropy(freq):
    return -np.sum(freq * np.log(freq + 1e-12))


def oracle_scores(g_true, i_true, sig_g, sig_i):
    wg = 1.0 / (sig_g[:, None, None] ** 2)
    wi = 1.0 / (sig_i[:, None, None] ** 2)
    x = (wg * g_true + wi * i_true) / (wg + wi)
    return score_mean(x)


def select_adaptive_alpha(g_obs, i_obs, sig_g, sig_i):
    """
    Local adaptive alpha selection for each cell (criterion, project, mc),
    combining:
    1) direction and magnitude of disagreement (i_obs - g_obs)
    2) criterion-level relative reliability from sig_g and sig_i

    Reliability ratio:
        r = sig_g / sig_i
    If r > 1, intuition is more reliable.
    If r < 1, data is more reliable.

    Base direction from reliability:
    - strong intuition reliability advantage -> phi^2
    - mild intuition reliability advantage   -> phi
    - balanced reliability                   -> 1
    - mild data reliability advantage        -> 1/phi
    - strong data reliability advantage      -> 1/phi^2

    Then disagreement magnitude amplifies the base regime one step outward.
    """
    delta = i_obs - g_obs
    abs_delta = np.abs(delta)

    ratio = sig_g / sig_i
    ratio_3d = ratio[:, None, None]

    alpha = np.ones_like(g_obs, dtype=float)

    # Base regime from relative reliability
    strong_i = ratio_3d >= 1.75
    mild_i = (ratio_3d >= 1.20) & (ratio_3d < 1.75)
    mild_g = (ratio_3d > 0.57) & (ratio_3d < 0.83)
    strong_g = ratio_3d <= 0.57

    alpha[strong_i] = PHI2
    alpha[mild_i] = PHI
    alpha[mild_g] = PHI_INV
    alpha[strong_g] = PHI_INV2

    # Disagreement amplification
    moderate = (abs_delta >= 0.75) & (abs_delta < 1.50)
    strong = abs_delta >= 1.50

    # move one step outward for moderate disagreement
    alpha[(alpha == 1.0) & moderate & (delta > 0)] = PHI
    alpha[(alpha == 1.0) & moderate & (delta < 0)] = PHI_INV

    alpha[(alpha == PHI) & moderate] = PHI2
    alpha[(alpha == PHI_INV) & moderate] = PHI_INV2

    # strong disagreement pushes directly to the extreme in the observed direction
    alpha[strong & (delta > 0)] = PHI2
    alpha[strong & (delta < 0)] = PHI_INV2

    return alpha


def ara_adaptive_operator(g_obs, i_obs, sig_g, sig_i, ell=L):
    alpha = select_adaptive_alpha(g_obs, i_obs, sig_g, sig_i)
    x = (ell + g_obs + alpha * i_obs) / (2.0 + alpha)
    return x, alpha


def alpha_shares(alpha):
    total = alpha.size
    return {
        "share_phi_inv2": float(np.sum(np.isclose(alpha, PHI_INV2)) / total),
        "share_phi_inv": float(np.sum(np.isclose(alpha, PHI_INV)) / total),
        "share_one": float(np.sum(np.isclose(alpha, 1.0)) / total),
        "share_phi": float(np.sum(np.isclose(alpha, PHI)) / total),
        "share_phi2": float(np.sum(np.isclose(alpha, PHI2)) / total),
    }


def evaluate_single_method(method_name, scores, oracle, oracle_w, oracle_best, extra=None):
    w = winners(scores)
    chosen = oracle[w, np.arange(oracle.shape[1])]
    regret = oracle_best - chosen
    freq = np.bincount(w, minlength=N_PROJECTS) / oracle.shape[1]

    row = {
        "method": method_name,
        "accuracy_vs_oracle": float(np.mean(w == oracle_w)),
        "mean_regret": float(np.mean(regret)),
        "p95_regret": float(np.quantile(regret, 0.95)),
        "catastrophic_regret_rate": float(np.mean(regret > 0.50)),
        "winner_entropy": float(winner_entropy(freq)),
        "P_win_A": float(freq[0]),
        "P_win_B": float(freq[1]),
        "P_win_C": float(freq[2]),
        "P_win_D": float(freq[3]),
        "share_phi_inv2": "",
        "share_phi_inv": "",
        "share_one": "",
        "share_phi": "",
        "share_phi2": "",
    }

    if extra is not None:
        row.update(extra)

    return row


def evaluate(g_obs, i_obs, g_true, i_true, sig_g, sig_i):
    oracle = oracle_scores(g_true, i_true, sig_g, sig_i)
    oracle_w = winners(oracle)
    oracle_best = oracle[oracle_w, np.arange(oracle.shape[1])]

    rows = []

    # Adaptive ARA only
    x_ara, alpha = ara_adaptive_operator(g_obs, i_obs, sig_g, sig_i)
    rows.append(
        evaluate_single_method(
            method_name="ARA_adaptive",
            scores=score_mean(x_ara),
            oracle=oracle,
            oracle_w=oracle_w,
            oracle_best=oracle_best,
            extra=alpha_shares(alpha),
        )
    )

    # Direct MCDA baselines
    x_plain = aggregate_mean(g_obs, i_obs)

    rows.append(
        evaluate_single_method(
            method_name="WSM_direct",
            scores=score_mean(x_plain),
            oracle=oracle,
            oracle_w=oracle_w,
            oracle_best=oracle_best,
        )
    )

    rows.append(
        evaluate_single_method(
            method_name="TOPSIS_direct",
            scores=score_topsis(x_plain),
            oracle=oracle,
            oracle_w=oracle_w,
            oracle_best=oracle_best,
        )
    )

    rows.append(
        evaluate_single_method(
            method_name="PROMETHEE_II_direct",
            scores=score_promethee_ii(x_plain),
            oracle=oracle,
            oracle_w=oracle_w,
            oracle_best=oracle_best,
        )
    )

    rows.append(
        evaluate_single_method(
            method_name="ELECTRE_I_direct",
            scores=score_electre_i(x_plain),
            oracle=oracle,
            oracle_w=oracle_w,
            oracle_best=oracle_best,
        )
    )

    return rows


def save_csv(rows, output_path, missing_rate_g, missing_rate_i, outlier_rate):
    if not rows:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "method",
        "accuracy_vs_oracle",
        "mean_regret",
        "p95_regret",
        "catastrophic_regret_rate",
        "winner_entropy",
        "P_win_A",
        "P_win_B",
        "P_win_C",
        "P_win_D",
        "share_phi_inv2",
        "share_phi_inv",
        "share_one",
        "share_phi",
        "share_phi2",
        "missing_rate_g",
        "missing_rate_i",
        "outlier_rate",
    ]

    for row in rows:
        row["missing_rate_g"] = missing_rate_g
        row["missing_rate_i"] = missing_rate_i
        row["outlier_rate"] = outlier_rate

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output_path}")


def print_summary(rows):
    print("\n=== SUMMARY ===")
    rows_sorted = sorted(rows, key=lambda r: (-r["accuracy_vs_oracle"], r["mean_regret"]))

    for r in rows_sorted:
        print(
            f"{r['method']:20s} | "
            f"acc={r['accuracy_vs_oracle']:.4f} | "
            f"mean_regret={r['mean_regret']:.6f} | "
            f"p95={r['p95_regret']:.6f} | "
            f"catastrophic={r['catastrophic_regret_rate']:.6f} | "
            f"entropy={r['winner_entropy']:.6f}"
        )

        if r["method"] == "ARA_adaptive":
            print(
                "   alpha shares: "
                f"phi^-2={float(r['share_phi_inv2']):.4f}, "
                f"phi^-1={float(r['share_phi_inv']):.4f}, "
                f"1={float(r['share_one']):.4f}, "
                f"phi={float(r['share_phi']):.4f}, "
                f"phi^2={float(r['share_phi2']):.4f}"
            )


def run_test(
    n_mc=120000,
    seed=123,
    missing_rate_g=0.08,
    missing_rate_i=0.14,
    outlier_rate=0.05,
    outlier_scale_g=2.0,
    outlier_scale_i=2.8,
    t_df=3,
):
    rng = np.random.default_rng(seed)

    sig_g = np.full(M, 0.85)
    sig_i = np.full(M, 0.85)

    # Risk-like criteria: intuition more reliable
    sig_g[7:10] = 1.25
    sig_i[7:10] = 0.50

    # Budget/measurement-like criteria: data more reliable
    sig_g[4:6] = 0.45
    sig_i[4:6] = 1.15

    g_true = clip010(
        G[:, :, None] + rng.normal(0, sig_g[:, None, None], size=(M, N_PROJECTS, n_mc))
    )
    i_true = clip010(
        I[:, :, None] + rng.normal(0, sig_i[:, None, None], size=(M, N_PROJECTS, n_mc))
    )

    g_obs = g_true.copy()
    i_obs = i_true.copy()

    out_g = rng.random(size=g_obs.shape) < outlier_rate
    out_i = rng.random(size=i_obs.shape) < outlier_rate

    g_obs = clip010(
        g_obs + out_g * rng.standard_t(df=t_df, size=g_obs.shape) * outlier_scale_g
    )
    i_obs = clip010(
        i_obs + out_i * rng.standard_t(df=t_df, size=i_obs.shape) * outlier_scale_i
    )

    miss_g = rng.random(size=g_obs.shape) < missing_rate_g
    miss_i = rng.random(size=i_obs.shape) < missing_rate_i

    g_obs = fill_anchor(g_obs, miss_g, anchor=L)
    i_obs = fill_anchor(i_obs, miss_i, anchor=L)

    rows = evaluate(g_obs, i_obs, g_true, i_true, sig_g, sig_i)

    output_path = "extended_tests/results/csv/testD_adaptive_ara_missing_outlier_vs_mcda.csv"
    save_csv(rows, output_path, missing_rate_g, missing_rate_i, outlier_rate)
    print_summary(rows)


if __name__ == "__main__":
    run_test()
