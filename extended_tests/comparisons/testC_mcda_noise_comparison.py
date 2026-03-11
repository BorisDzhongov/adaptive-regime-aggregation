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


def aggregate_mean(g, i):
    return 0.5 * (g + i)


def score_mean(x):
    # x shape: (criteria, projects, mc)
    return x.mean(axis=0)


def score_topsis(x):
    # x shape: (criteria, projects, mc)
    # normalize by criterion within each mc run
    denom = np.sqrt(np.sum(x ** 2, axis=1, keepdims=True))
    denom = np.where(denom == 0.0, 1.0, denom)

    r = x / denom
    v = r / M  # equal weights

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
    xn = (x - minv) / rng  # normalized benefit matrix

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


def oracle_scores(g_true, i_true):
    """
    Oracle = true balanced state based on the latent clean signal.
    """
    x = 0.5 * (g_true + i_true)
    return score_mean(x)


def noise_symmetric(rng, sigma, shape):
    return rng.normal(0.0, sigma, size=shape)


def noise_asymmetric(rng, sigma, shape):
    z = rng.normal(0.0, sigma, size=shape)
    skew = rng.exponential(scale=sigma * 0.35, size=shape) - sigma * 0.35
    return z + skew


def noise_heavytail(rng, sigma, shape, df=3):
    return rng.standard_t(df=df, size=shape) * sigma


def select_adaptive_alpha(g_obs, i_obs):
    """
    Local adaptive alpha selection for each cell (criterion, project, mc).

    Logic:
    - small disagreement -> alpha = 1
    - moderate disagreement:
        I > G -> phi
        I < G -> 1/phi
    - strong disagreement:
        I > G -> phi^2
        I < G -> 1/phi^2

    This makes ARA adaptive per local situation, not constant per test.
    """
    delta = i_obs - g_obs
    abs_delta = np.abs(delta)

    alpha = np.ones_like(delta, dtype=float)

    mild = (abs_delta >= 0.50) & (abs_delta < 1.25)
    strong = abs_delta >= 1.25

    alpha[mild & (delta > 0)] = PHI
    alpha[mild & (delta < 0)] = PHI_INV

    alpha[strong & (delta > 0)] = PHI2
    alpha[strong & (delta < 0)] = PHI_INV2

    return alpha


def ara_adaptive_operator(g_obs, i_obs, ell=L):
    alpha = select_adaptive_alpha(g_obs, i_obs)
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


def evaluate_methods(g_obs, i_obs, g_true, i_true):
    oracle = oracle_scores(g_true, i_true)
    oracle_w = winners(oracle)
    oracle_best = oracle[oracle_w, np.arange(oracle.shape[1])]

    rows = []

    # 1) Adaptive ARA only
    x_ara, alpha = ara_adaptive_operator(g_obs, i_obs)
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

    # 2) Plain direct MCDA methods on the observed combined matrix
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
                f"{r['method']:20s} | "
                f"acc={r['accuracy_vs_oracle']:.4f} | "
                f"mean_regret={r['mean_regret']:.6f} | "
                f"p95={r['p95_regret']:.6f} | "
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


def run_test(n_mc=100000, sigma=0.8, seed=42):
    rng = np.random.default_rng(seed)

    g_true = np.repeat(G[:, :, None], n_mc, axis=2)
    i_true = np.repeat(I[:, :, None], n_mc, axis=2)

    scenarios = {}

    eps_g = noise_symmetric(rng, sigma, g_true.shape)
    eps_i = noise_symmetric(rng, sigma, i_true.shape)
    scenarios["symmetric"] = (clip010(g_true + eps_g), clip010(i_true + eps_i))

    eps_g = noise_asymmetric(rng, sigma, g_true.shape)
    eps_i = noise_asymmetric(rng, sigma, i_true.shape)
    scenarios["asymmetric"] = (clip010(g_true + eps_g), clip010(i_true + eps_i))

    eps_g = noise_heavytail(rng, sigma, g_true.shape)
    eps_i = noise_heavytail(rng, sigma, i_true.shape)
    scenarios["heavytail"] = (clip010(g_true + eps_g), clip010(i_true + eps_i))

    all_rows = []

    for scenario_name, (g_obs, i_obs) in scenarios.items():
        rows = evaluate_methods(g_obs, i_obs, g_true, i_true)
        for row in rows:
            row["scenario"] = scenario_name
        all_rows.extend(rows)

    output_path = "extended_tests/results/csv/testC_adaptive_ara_vs_mcda.csv"
    save_csv(all_rows, output_path)
    print_summary(all_rows)


if __name__ == "__main__":
    run_test()
