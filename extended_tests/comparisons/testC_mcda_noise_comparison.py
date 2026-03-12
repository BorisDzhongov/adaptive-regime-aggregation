import numpy as np
import csv
import os

PHI = (1 + 5 ** 0.5) / 2
PHI_INV = 1.0 / PHI
PHI2 = PHI ** 2
PHI_INV2 = 1.0 / (PHI ** 2)

L = 5.0
PROJECTS = ["A", "B", "C", "D"]

# ---------------------------------------------------------------------
# Base matrices from Example II
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Criterion meta-structure
# ---------------------------------------------------------------------
CRITERION_SIGN = np.array([1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1], dtype=float)

RELIABILITY_G = np.array([0.92, 0.88, 0.72, 0.85, 0.76, 0.90, 0.82, 0.68, 0.65, 0.80, 0.91], dtype=float)
RELIABILITY_I = np.array([0.80, 0.86, 0.70, 0.83, 0.74, 0.78, 0.79, 0.64, 0.60, 0.76, 0.88], dtype=float)

GROUPS = {
    "economic": [0, 1, 4],
    "operational": [2, 3, 5, 6],
    "risk": [7, 8],
    "strategic": [9, 10],
}

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def clip010(x):
    return np.clip(x, 0.0, 10.0)

def to_benefit_space(x):
    sign = CRITERION_SIGN[:, None, None]
    return np.where(sign > 0, x, 10.0 - x)

def safe_fill_nan_with_criterion_mean(x):
    means = np.nanmean(x, axis=1, keepdims=True)
    means = np.where(np.isnan(means), L, means)
    return np.where(np.isnan(x), means, x)

def aggregate_mean(g, i):
    return 0.5 * (g + i)

def winner_entropy(freq):
    return -np.sum(freq * np.log(freq + 1e-12))

def expand_prior_reliability(rel_1d, shape):
    return np.repeat(rel_1d[:, None, None], shape[1], axis=1).repeat(shape[2], axis=2)

# ---------------------------------------------------------------------
# Tie-aware utilities
# ---------------------------------------------------------------------
def winner_mask(scores, tol=1e-10):
    max_scores = np.max(scores, axis=0, keepdims=True)
    return np.isclose(scores, max_scores, atol=tol, rtol=0.0)

def tie_rate(scores, tol=1e-10):
    wm = winner_mask(scores, tol=tol)
    n_winners = np.sum(wm, axis=0)
    return float(np.mean(n_winners > 1))

def tie_aware_accuracy(scores, oracle_scores, tol=1e-10):
    wm = winner_mask(scores, tol=tol)
    oracle_wm = winner_mask(oracle_scores, tol=tol)

    acc = []
    for j in range(scores.shape[1]):
        winners_j = np.where(wm[:, j])[0]
        oracle_j = np.where(oracle_wm[:, j])[0]

        overlap = len(set(winners_j).intersection(set(oracle_j)))
        if overlap == 0:
            acc.append(0.0)
        else:
            acc.append(overlap / len(winners_j))

    return float(np.mean(acc))

def tie_aware_regret(scores, oracle_scores, tol=1e-10):
    wm = winner_mask(scores, tol=tol)

    regrets = []
    for j in range(scores.shape[1]):
        winners_j = np.where(wm[:, j])[0]
        best_oracle = float(np.max(oracle_scores[:, j]))
        chosen_oracle_mean = float(np.mean(oracle_scores[winners_j, j]))
        regrets.append(best_oracle - chosen_oracle_mean)

    return np.array(regrets, dtype=float)

def tie_aware_win_probabilities(scores, tol=1e-10):
    wm = winner_mask(scores, tol=tol)
    probs = np.zeros(scores.shape[0], dtype=float)

    for j in range(scores.shape[1]):
        winners_j = np.where(wm[:, j])[0]
        share = 1.0 / len(winners_j)
        for idx in winners_j:
            probs[idx] += share

    probs /= scores.shape[1]
    return probs

# ---------------------------------------------------------------------
# MCDA scoring
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Reliability interpretation for ARA
# ---------------------------------------------------------------------
def boundary_stress_indicator(raw):
    raw_filled = np.where(np.isnan(raw), L, raw)

    clipped_outside = ((raw_filled < 0.0) | (raw_filled > 10.0)).astype(float)

    dist_to_boundary = np.minimum(np.abs(raw_filled - 0.0), np.abs(raw_filled - 10.0))
    near_boundary = np.clip((1.0 - dist_to_boundary) / 1.0, 0.0, 1.0)

    penalty = np.maximum(clipped_outside, near_boundary)
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

def interpret_reliability(raw_obs, benefit_obs, prior_rel_1d):
    prior = expand_prior_reliability(prior_rel_1d, raw_obs.shape)

    missing_pen = np.isnan(raw_obs).astype(float)
    boundary_pen = boundary_stress_indicator(raw_obs)
    group_pen = group_instability_indicator(benefit_obs)

    rel = (
        0.70 * prior
        + 0.15 * (1.0 - missing_pen)
        + 0.10 * (1.0 - boundary_pen)
        + 0.05 * (1.0 - group_pen)
    )

    return np.clip(rel, 0.0, 1.0)

# ---------------------------------------------------------------------
# ARA
# ---------------------------------------------------------------------
def select_adaptive_alpha_from_reliability(rel_g, rel_i):
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

def ara_adaptive_operator(g_obs_raw, i_obs_raw, ell=L):
    g_obs = to_benefit_space(clip010(g_obs_raw))
    i_obs = to_benefit_space(clip010(i_obs_raw))

    g_obs = safe_fill_nan_with_criterion_mean(g_obs)
    i_obs = safe_fill_nan_with_criterion_mean(i_obs)

    rel_g = interpret_reliability(
        raw_obs=g_obs_raw,
        benefit_obs=g_obs,
        prior_rel_1d=RELIABILITY_G,
    )

    rel_i = interpret_reliability(
        raw_obs=i_obs_raw,
        benefit_obs=i_obs,
        prior_rel_1d=RELIABILITY_I,
    )

    alpha = select_adaptive_alpha_from_reliability(rel_g, rel_i)
    x = (ell + g_obs + alpha * i_obs) / (2.0 + alpha)

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

# ---------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------
def oracle_scores(g_true, i_true):
    prior_g = expand_prior_reliability(RELIABILITY_G, g_true.shape)
    prior_i = expand_prior_reliability(RELIABILITY_I, i_true.shape)

    denom = prior_g + prior_i + 1e-12
    x = (prior_g * g_true + prior_i * i_true) / denom
    return score_mean(x)

# ---------------------------------------------------------------------
# Latent truth variation
# ---------------------------------------------------------------------
def generate_latent_truth(rng, n_mc, truth_sigma=0.35):
    g_true_raw = np.repeat(G[:, :, None], n_mc, axis=2)
    i_true_raw = np.repeat(I[:, :, None], n_mc, axis=2)

    latent_g = g_true_raw + rng.normal(0.0, truth_sigma, size=g_true_raw.shape)
    latent_i = i_true_raw + rng.normal(0.0, truth_sigma, size=i_true_raw.shape)

    latent_g = clip010(latent_g)
    latent_i = clip010(latent_i)

    return latent_g, latent_i

# ---------------------------------------------------------------------
# Noise / corruption generators
# ---------------------------------------------------------------------
def add_group_correlated_noise(rng, sigma, shape):
    out = np.zeros(shape, dtype=float)
    _, n_projects, n_mc = shape

    for _, idxs in GROUPS.items():
        idxs = np.array(idxs, dtype=int)
        common = rng.normal(0.0, sigma, size=(1, n_projects, n_mc))
        idio = rng.normal(0.0, sigma * 0.55, size=(len(idxs), n_projects, n_mc))
        out[idxs, :, :] = 0.70 * common + 0.60 * idio

    return out

def add_reliability_scaled_noise(rng, sigma, shape, reliability):
    base = rng.normal(0.0, sigma, size=shape)
    scale = (1.35 - reliability)[:, None, None]
    return base * scale

def add_asymmetric_bias(rng, sigma, shape):
    z = rng.normal(0.0, sigma * 0.45, size=shape)
    skew = rng.exponential(scale=sigma * 0.28, size=shape) - sigma * 0.28
    return z + skew

def add_heavytail_noise(rng, sigma, shape, df=3):
    return rng.standard_t(df=df, size=shape) * sigma

def inject_missingness(rng, x, p_missing=0.08):
    mask = rng.random(size=x.shape) < p_missing
    y = x.copy()
    y[mask] = np.nan
    return y

def inject_outliers(rng, x, p_outlier=0.025, magnitude=3.2):
    y = x.copy()
    mask = rng.random(size=x.shape) < p_outlier
    shocks = rng.normal(0.0, magnitude, size=x.shape)
    y[mask] = y[mask] + shocks[mask]
    return y

def inject_regime_conflict_shift(g, i, strength=0.9):
    g2 = g.copy()
    i2 = i.copy()

    for k in range(M):
        for p in range(N_PROJECTS):
            sign = 1.0 if ((k + p) % 2 == 0) else -1.0
            shift = strength * sign
            g2[k, p, :] += shift
            i2[k, p, :] -= shift

    return g2, i2

# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
def evaluate_single_method(method_name, scores, oracle, extra=None):
    regrets = tie_aware_regret(scores, oracle)
    probs = tie_aware_win_probabilities(scores)

    row = {
        "method": method_name,
        "accuracy_vs_oracle": tie_aware_accuracy(scores, oracle),
        "mean_regret": float(np.mean(regrets)),
        "p95_regret": float(np.quantile(regrets, 0.95)),
        "winner_entropy": float(winner_entropy(probs)),
        "tie_rate": tie_rate(scores),
        "P_win_A": float(probs[0]),
        "P_win_B": float(probs[1]),
        "P_win_C": float(probs[2]),
        "P_win_D": float(probs[3]),
        "share_phi_inv2": "",
        "share_phi_inv": "",
        "share_one": "",
        "share_phi": "",
        "share_phi2": "",
    }

    if extra is not None:
        row.update(extra)

    return row

def evaluate_methods(g_obs_raw, i_obs_raw, g_true, i_true):
    g_obs = to_benefit_space(clip010(g_obs_raw))
    i_obs = to_benefit_space(clip010(i_obs_raw))

    oracle = oracle_scores(g_true, i_true)
    rows = []

    x_ara, alpha, rel_g, rel_i = ara_adaptive_operator(g_obs_raw, i_obs_raw)
    rows.append(
        evaluate_single_method(
            method_name="ARA_adaptive_reliability",
            scores=score_mean(x_ara),
            oracle=oracle,
            extra=alpha_shares(alpha),
        )
    )

    x_plain = aggregate_mean(g_obs, i_obs)

    rows.append(
        evaluate_single_method(
            method_name="WSM_direct",
            scores=score_mean(x_plain),
            oracle=oracle,
        )
    )

    rows.append(
        evaluate_single_method(
            method_name="TOPSIS_direct",
            scores=score_topsis(x_plain),
            oracle=oracle,
        )
    )

    rows.append(
        evaluate_single_method(
            method_name="PROMETHEE_II_direct",
            scores=score_promethee_ii(x_plain),
            oracle=oracle,
        )
    )

    rows.append(
        evaluate_single_method(
            method_name="ELECTRE_I_direct",
            scores=score_electre_i(x_plain),
            oracle=oracle,
        )
    )

    return rows

# ---------------------------------------------------------------------
# CSV / reporting
# ---------------------------------------------------------------------
def save_csv(rows, output_path):
    if not rows:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "scenario",
        "method",
        "accuracy_vs_oracle",
        "mean_regret",
        "p95_regret",
        "winner_entropy",
        "tie_rate",
        "P_win_A",
        "P_win_B",
        "P_win_C",
        "P_win_D",
        "share_phi_inv2",
        "share_phi_inv",
        "share_one",
        "share_phi",
        "share_phi2",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output_path}")

def print_summary(rows):
    print("\n=== SUMMARY ===")
    grouped = {}
    for row in rows:
        scenario = row["scenario"]
        grouped.setdefault(scenario, []).append(row)

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
                f"tie_rate={r['tie_rate']:.4f} | "
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

# ---------------------------------------------------------------------
# Main stress test
# ---------------------------------------------------------------------
def run_test(n_mc=100000, sigma=0.95, seed=42, truth_sigma=0.35):
    rng = np.random.default_rng(seed)

    g_true_raw, i_true_raw = generate_latent_truth(rng, n_mc=n_mc, truth_sigma=truth_sigma)

    g_true = to_benefit_space(g_true_raw)
    i_true = to_benefit_space(i_true_raw)

    scenarios = {}

    eps_g = (
        add_group_correlated_noise(rng, sigma, g_true_raw.shape)
        + add_reliability_scaled_noise(rng, sigma, g_true_raw.shape, RELIABILITY_G)
    )
    eps_i = (
        add_group_correlated_noise(rng, sigma, i_true_raw.shape)
        + add_reliability_scaled_noise(rng, sigma, i_true_raw.shape, RELIABILITY_I)
    )
    scenarios["correlated_reliability"] = (g_true_raw + eps_g, i_true_raw + eps_i)

    eps_g = add_asymmetric_bias(rng, sigma, g_true_raw.shape)
    eps_i = add_asymmetric_bias(rng, sigma, i_true_raw.shape)
    g_obs = g_true_raw + eps_g
    i_obs = i_true_raw + eps_i
    g_obs, i_obs = inject_regime_conflict_shift(g_obs, i_obs, strength=0.85)
    scenarios["asymmetric_conflict"] = (g_obs, i_obs)

    eps_g = add_heavytail_noise(rng, sigma, g_true_raw.shape, df=3)
    eps_i = add_heavytail_noise(rng, sigma, i_true_raw.shape, df=3)
    g_obs = inject_outliers(rng, g_true_raw + eps_g, p_outlier=0.03, magnitude=3.5)
    i_obs = inject_outliers(rng, i_true_raw + eps_i, p_outlier=0.03, magnitude=3.5)
    scenarios["heavytail_outliers"] = (g_obs, i_obs)

    eps_g = (
        add_group_correlated_noise(rng, sigma, g_true_raw.shape)
        + add_reliability_scaled_noise(rng, sigma, g_true_raw.shape, RELIABILITY_G)
        + add_asymmetric_bias(rng, sigma, g_true_raw.shape)
    )
    eps_i = (
        add_group_correlated_noise(rng, sigma, i_true_raw.shape)
        + add_reliability_scaled_noise(rng, sigma, i_true_raw.shape, RELIABILITY_I)
        + add_asymmetric_bias(rng, sigma, i_true_raw.shape)
    )

    g_obs = g_true_raw + eps_g
    i_obs = i_true_raw + eps_i
    g_obs, i_obs = inject_regime_conflict_shift(g_obs, i_obs, strength=1.00)
    g_obs = inject_outliers(rng, g_obs, p_outlier=0.035, magnitude=3.8)
    i_obs = inject_outliers(rng, i_obs, p_outlier=0.035, magnitude=3.8)
    g_obs = inject_missingness(rng, g_obs, p_missing=0.10)
    i_obs = inject_missingness(rng, i_obs, p_missing=0.10)
    scenarios["full_stress"] = (g_obs, i_obs)

    eps_g = (
        add_group_correlated_noise(rng, sigma * 1.8, g_true_raw.shape)
        + add_heavytail_noise(rng, sigma * 1.6, g_true_raw.shape, df=2)
        + add_reliability_scaled_noise(rng, sigma * 1.4, g_true_raw.shape, RELIABILITY_G)
    )
    eps_i = (
        add_group_correlated_noise(rng, sigma * 1.8, i_true_raw.shape)
        + add_heavytail_noise(rng, sigma * 1.6, i_true_raw.shape, df=2)
        + add_reliability_scaled_noise(rng, sigma * 1.4, i_true_raw.shape, RELIABILITY_I)
    )

    g_obs = g_true_raw + eps_g
    i_obs = i_true_raw + eps_i
    g_obs, i_obs = inject_regime_conflict_shift(g_obs, i_obs, strength=1.40)
    g_obs = inject_outliers(rng, g_obs, p_outlier=0.10, magnitude=4.5)
    i_obs = inject_outliers(rng, i_obs, p_outlier=0.10, magnitude=4.5)
    g_obs = inject_missingness(rng, g_obs, p_missing=0.22)
    i_obs = inject_missingness(rng, i_obs, p_missing=0.22)

    g_obs = np.clip(g_obs, -2.0, 12.0)
    i_obs = np.clip(i_obs, -2.0, 12.0)

    scenarios["breakdown_limit"] = (g_obs, i_obs)

    all_rows = []

    for scenario_name, (g_obs, i_obs) in scenarios.items():
        rows = evaluate_methods(g_obs, i_obs, g_true, i_true)
        for row in rows:
            row["scenario"] = scenario_name
        all_rows.extend(rows)

    output_path = "extended_tests/results/csv/testC_adaptive_ara_vs_mcda_stress.csv"
    save_csv(all_rows, output_path)
    print_summary(all_rows)

if __name__ == "__main__":
    run_test()
