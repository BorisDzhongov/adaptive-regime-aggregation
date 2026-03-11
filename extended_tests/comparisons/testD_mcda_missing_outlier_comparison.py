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

# 1 = benefit, -1 = cost
CRITERION_SIGN = np.array([1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1], dtype=float)

# criterion-level standard deviations (smaller = more reliable)
SIG_G = np.array([0.75, 0.80, 0.95, 0.82, 0.45, 0.45, 0.88, 1.25, 1.25, 1.25, 0.78], dtype=float)
SIG_I = np.array([0.82, 0.78, 0.98, 0.85, 1.15, 1.15, 0.90, 0.50, 0.50, 0.50, 0.75], dtype=float)

GROUPS = {
    "economic": [0, 1, 4],
    "operational": [2, 3, 5, 6],
    "risk": [7, 8],
    "strategic": [9, 10],
}


def clip010(x):
    return np.clip(x, 0.0, 10.0)


def to_benefit_space(x):
    sign = CRITERION_SIGN[:, None, None]
    return np.where(sign > 0, x, 10.0 - x)


def safe_fill_nan_with_anchor(x, anchor=L):
    return np.where(np.isnan(x), anchor, x)


def safe_fill_nan_with_criterion_mean(x):
    means = np.nanmean(x, axis=1, keepdims=True)
    means = np.where(np.isnan(means), L, means)
    return np.where(np.isnan(x), means, x)


def aggregate_mean(g, i):
    return 0.5 * (g + i)


def score_mean(x):
    x = safe_fill_nan_with_criterion_mean(x)
    return x.mean(axis=0)


def score_topsis(x):
    x = safe_fill_nan_with_criterion_mean(x)

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
    x = safe_fill_nan_with_criterion_mean(x)

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
    x = safe_fill_nan_with_criterion_mean(x)

    minv = np.min(x, axis=1, keepdims=True)
    maxv = np.max(x, axis=1, keepdims=True)
    rng = np.where((maxv - minv) == 0.0, 1.0, (maxv - minv))
    xn = (x - minv) / rng

    outrank = np.zeros((N_PROJECTS, N_PROJECTS, x.shape[2]), dtype=float)

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


def sigma_to_prior_reliability(sig):
    """
    Smaller sigma -> higher prior reliability in [0,1].
    """
    rel = 1.0 / (1.0 + sig ** 2)
    return np.clip(rel, 0.0, 1.0)


def expand_prior_reliability(rel_1d, shape):
    return np.repeat(rel_1d[:, None, None], shape[1], axis=1).repeat(shape[2], axis=2)


def boundary_stress_indicator(raw):
    raw_filled = np.where(np.isnan(raw), L, raw)

    outside = ((raw_filled < 0.0) | (raw_filled > 10.0)).astype(float)
    dist_to_boundary = np.minimum(np.abs(raw_filled), np.abs(10.0 - raw_filled))
    near_boundary = np.clip((1.0 - dist_to_boundary) / 1.0, 0.0, 1.0)

    penalty = np.maximum(outside, near_boundary)
    penalty = np.where(np.isnan(raw), 1.0, penalty)
    return penalty


def group_instability_indicator(x_benefit):
    x = safe_fill_nan_with_criterion_mean(x_benefit)
    penalty = np.zeros_like(x, dtype=float)

    for _, idxs in GROUPS.items():
        idxs = np.array(idxs, dtype=int)
        group_mean = np.mean(x[idxs, :, :], axis=0, keepdims=True)
        dev = np.abs(x[idxs, :, :] - group_mean)
        penalty[idxs, :, :] = np.clip(dev / 3.0, 0.0, 1.0)

    return penalty


def interpret_reliability(raw_obs, benefit_obs, sig):
    """
    Reliability is architecturally interpreted from:
    - prior sigma-based reliability
    - missingness
    - boundary/clipping stress
    - group instability
    """
    prior_1d = sigma_to_prior_reliability(sig)
    prior = expand_prior_reliability(prior_1d, raw_obs.shape)

    missing_pen = np.isnan(raw_obs).astype(float)
    boundary_pen = boundary_stress_indicator(raw_obs)
    group_pen = group_instability_indicator(benefit_obs)

    rel = (
        0.65 * prior
        + 0.15 * (1.0 - missing_pen)
        + 0.10 * (1.0 - boundary_pen)
        + 0.10 * (1.0 - group_pen)
    )

    return np.clip(rel, 0.0, 1.0)


def select_adaptive_alpha_from_reliability(rel_g, rel_i):
    """
    Choose alpha only from interpreted reliability gap.
    """
    rel_gap = rel_i - rel_g
    abs_gap = np.abs(rel_gap)

    alpha = np.ones_like(rel_gap, dtype=float)

    mild = (abs_gap >= 0.05) & (abs_gap < 0.15)
    strong = abs_gap >= 0.15

    alpha[mild & (rel_gap > 0)] = PHI
    alpha[mild & (rel_gap < 0)] = PHI_INV

    alpha[strong & (rel_gap > 0)] = PHI2
    alpha[strong & (rel_gap < 0)] = PHI_INV2

    return alpha


def ara_adaptive_operator(g_obs, i_obs, sig_g, sig_i, ell=L):
    g_filled = safe_fill_nan_with_anchor(g_obs, anchor=L)
    i_filled = safe_fill_nan_with_anchor(i_obs, anchor=L)

    rel_g = interpret_reliability(g_obs, g_filled, sig_g)
    rel_i = interpret_reliability(i_obs, i_filled, sig_i)

    alpha = select_adaptive_alpha_from_reliability(rel_g, rel_i)
    x = (ell + g_filled + alpha * i_filled) / (2.0 + alpha)
    return x, alpha, rel_g, rel_i


def alpha_shares(alpha):
    total = alpha.size
    return {
        "share_phi_inv2": float(np.sum(np.isclose(alpha, PHI_INV2)) / total),
        "share_phi_inv": float(np.sum(np.isclose(alpha, PHI_INV)) / total),
        "share_one": float(np.sum(np.isclose(alpha, 1.0)) / total),
        "share_phi": float(np.sum(np.isclose(alpha, PHI)) / total),
        "share_phi2": float(np.sum(np.isclose(alpha, PHI2)) / total),
    }


def add_group_correlated_noise(rng, sigma, shape):
    out = np.zeros(shape, dtype=float)
    _, n_projects, n_mc = shape

    for _, idxs in GROUPS.items():
        idxs = np.array(idxs, dtype=int)
        common = rng.normal(0.0, sigma, size=(1, n_projects, n_mc))
        idio = rng.normal(0.0, sigma * 0.55, size=(len(idxs), n_projects, n_mc))
        out[idxs, :, :] = 0.70 * common + 0.60 * idio

    return out


def add_heavytail_noise(rng, sigma, shape, df=3):
    return rng.standard_t(df=df, size=shape) * sigma


def inject_outliers(rng, x, p_outlier=0.04, magnitude=3.0, df=3):
    y = x.copy()
    mask = rng.random(size=x.shape) < p_outlier
    shocks = rng.standard_t(df=df, size=x.shape) * magnitude
    y[mask] = y[mask] + shocks[mask]
    return y


def inject_regime_conflict_shift(g, i, strength=0.90):
    g2 = g.copy()
    i2 = i.copy()

    for k in range(M):
        for p in range(N_PROJECTS):
            sign = 1.0 if ((k + p) % 2 == 0) else -1.0
            shift = strength * sign
            g2[k, p, :] += shift
            i2[k, p, :] -= shift

    return g2, i2


def inject_nonrandom_missingness(rng, x, base_rate=0.10, tail_boost=0.12):
    prob = np.full(x.shape, base_rate, dtype=float)

    extreme = (x <= 2.0) | (x >= 8.0)
    prob = prob + tail_boost * extreme

    criterion_boost = np.linspace(0.00, 0.05, M)[:, None, None]
    prob = prob + criterion_boost

    prob = np.clip(prob, 0.0, 0.85)
    mask = rng.random(size=x.shape) < prob

    y = x.copy()
    y[mask] = np.nan
    return y, mask


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


def evaluate(g_obs_raw, i_obs_raw, g_true, i_true, sig_g, sig_i):
    g_obs = to_benefit_space(clip010(g_obs_raw))
    i_obs = to_benefit_space(clip010(i_obs_raw))

    oracle = oracle_scores(g_true, i_true, sig_g, sig_i)
    oracle_w = winners(oracle)
    oracle_best = oracle[oracle_w, np.arange(oracle.shape[1])]

    rows = []

    x_ara, alpha, rel_g, rel_i = ara_adaptive_operator(g_obs, i_obs, sig_g, sig_i)
    rows.append(
        evaluate_single_method(
            method_name="ARA_adaptive_reliability",
            scores=score_mean(x_ara),
            oracle=oracle,
            oracle_w=oracle_w,
            oracle_best=oracle_best,
            extra=alpha_shares(alpha),
        )
    )

    x_plain = aggregate_mean(
        safe_fill_nan_with_anchor(g_obs, anchor=L),
        safe_fill_nan_with_anchor(i_obs, anchor=L),
    )

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


def save_csv(rows, output_path, meta):
    if not rows:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "scenario",
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
        "outlier_rate_g",
        "outlier_rate_i",
    ]

    for row in rows:
        row["missing_rate_g"] = meta["missing_rate_g"]
        row["missing_rate_i"] = meta["missing_rate_i"]
        row["outlier_rate_g"] = meta["outlier_rate_g"]
        row["outlier_rate_i"] = meta["outlier_rate_i"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output_path}")


def print_summary(rows):
    print("\n=== SUMMARY ===")
    grouped = {}
    for row in rows:
        grouped.setdefault(row["scenario"], []).append(row)

    for scenario, scenario_rows in grouped.items():
        print(f"\nScenario: {scenario}")
        scenario_rows = sorted(
            scenario_rows,
            key=lambda r: (-r["accuracy_vs_oracle"], r["mean_regret"])
        )

        for r in scenario_rows:
            print(
                f"{r['method']:28s} | "
                f"acc={r['accuracy_vs_oracle']:.4f} | "
                f"mean_regret={r['mean_regret']:.6f} | "
                f"p95={r['p95_regret']:.6f} | "
                f"catastrophic={r['catastrophic_regret_rate']:.6f} | "
                f"entropy={r['winner_entropy']:.6f}"
            )

            if r["method"] == "ARA_adaptive_reliability":
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
    sigma_corr=0.55,
    sigma_tail_g=0.90,
    sigma_tail_i=1.05,
    outlier_rate_g=0.04,
    outlier_rate_i=0.06,
    outlier_scale_g=2.8,
    outlier_scale_i=3.3,
    missing_base_g=0.07,
    missing_base_i=0.12,
    t_df=3,
):
    rng = np.random.default_rng(seed)

    g_true_raw = np.repeat(G[:, :, None], n_mc, axis=2)
    i_true_raw = np.repeat(I[:, :, None], n_mc, axis=2)

    g_true_raw = clip010(
        g_true_raw + rng.normal(0.0, SIG_G[:, None, None], size=g_true_raw.shape)
    )
    i_true_raw = clip010(
        i_true_raw + rng.normal(0.0, SIG_I[:, None, None], size=i_true_raw.shape)
    )

    g_true = to_benefit_space(g_true_raw)
    i_true = to_benefit_space(i_true_raw)

    scenarios = {}

    g_obs = g_true_raw + add_group_correlated_noise(rng, sigma_corr, g_true_raw.shape)
    i_obs = i_true_raw + add_group_correlated_noise(rng, sigma_corr, i_true_raw.shape)
    g_obs, i_obs = inject_regime_conflict_shift(g_obs, i_obs, strength=0.75)
    g_obs, miss_g = inject_nonrandom_missingness(rng, g_obs, base_rate=missing_base_g, tail_boost=0.10)
    i_obs, miss_i = inject_nonrandom_missingness(rng, i_obs, base_rate=missing_base_i, tail_boost=0.14)
    scenarios["missingness_dominant"] = (g_obs, i_obs, miss_g, miss_i, None, None)

    g_obs = g_true_raw + add_heavytail_noise(rng, sigma_tail_g, g_true_raw.shape, df=t_df)
    i_obs = i_true_raw + add_heavytail_noise(rng, sigma_tail_i, i_true_raw.shape, df=t_df)
    g_obs, i_obs = inject_regime_conflict_shift(g_obs, i_obs, strength=0.85)
    g_obs = inject_outliers(rng, g_obs, p_outlier=outlier_rate_g, magnitude=outlier_scale_g, df=t_df)
    i_obs = inject_outliers(rng, i_obs, p_outlier=outlier_rate_i, magnitude=outlier_scale_i, df=t_df)
    scenarios["outlier_dominant"] = (g_obs, i_obs, None, None, outlier_rate_g, outlier_rate_i)

    g_obs = (
        g_true_raw
        + add_group_correlated_noise(rng, sigma_corr, g_true_raw.shape)
        + add_heavytail_noise(rng, sigma_tail_g, g_true_raw.shape, df=t_df)
    )
    i_obs = (
        i_true_raw
        + add_group_correlated_noise(rng, sigma_corr, i_true_raw.shape)
        + add_heavytail_noise(rng, sigma_tail_i, i_true_raw.shape, df=t_df)
    )

    g_obs, i_obs = inject_regime_conflict_shift(g_obs, i_obs, strength=1.00)
    g_obs = inject_outliers(rng, g_obs, p_outlier=outlier_rate_g, magnitude=outlier_scale_g, df=t_df)
    i_obs = inject_outliers(rng, i_obs, p_outlier=outlier_rate_i, magnitude=outlier_scale_i, df=t_df)
    g_obs, miss_g = inject_nonrandom_missingness(rng, g_obs, base_rate=missing_base_g, tail_boost=0.12)
    i_obs, miss_i = inject_nonrandom_missingness(rng, i_obs, base_rate=missing_base_i, tail_boost=0.16)

    scenarios["full_robustness_stress"] = (g_obs, i_obs, miss_g, miss_i, outlier_rate_g, outlier_rate_i)

    all_rows = []

    for scenario_name, (g_obs, i_obs, miss_g, miss_i, og, oi) in scenarios.items():
        rows = evaluate(g_obs, i_obs, g_true, i_true, SIG_G, SIG_I)

        missing_rate_g = float(np.mean(miss_g)) if miss_g is not None else 0.0
        missing_rate_i = float(np.mean(miss_i)) if miss_i is not None else 0.0
        out_rate_g = float(og) if og is not None else 0.0
        out_rate_i = float(oi) if oi is not None else 0.0

        for row in rows:
            row["scenario"] = scenario_name

        meta = {
            "missing_rate_g": missing_rate_g,
            "missing_rate_i": missing_rate_i,
            "outlier_rate_g": out_rate_g,
            "outlier_rate_i": out_rate_i,
        }

        output_path = f"extended_tests/results/csv/testD_{scenario_name}.csv"
        save_csv(rows, output_path, meta)
        all_rows.extend(rows)

    print_summary(all_rows)


if __name__ == "__main__":
    run_test()
