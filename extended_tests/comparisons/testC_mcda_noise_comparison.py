import numpy as np
import pandas as pd

PHI = (1 + 5 ** 0.5) / 2
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


def ara_operator(g, i, alpha, ell=L):
    return (ell + g + alpha * i) / (2.0 + alpha)


def score_mean(x):
    return x.mean(axis=0)


def score_topsis(x):
    # x shape: (criteria, projects, mc)
    denom = np.sqrt(np.sum(x**2, axis=1, keepdims=True))
    denom = np.where(denom == 0.0, 1.0, denom)
    r = x / denom
    v = r / M  # equal weights
    ideal_best = np.max(v, axis=1, keepdims=True)
    ideal_worst = np.min(v, axis=1, keepdims=True)
    d_pos = np.sqrt(np.sum((v - ideal_best) ** 2, axis=0))
    d_neg = np.sqrt(np.sum((v - ideal_worst) ** 2, axis=0))
    return d_neg / (d_pos + d_neg + 1e-12)


def score_promethee_ii(x):
    # simple usual preference with normalized pairwise preference intensity
    # returns net flow per project for each MC replicate
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


def winners(scores):
    return np.argmax(scores, axis=0)


def oracle_scores(g_true, i_true):
    # latent neutral benchmark
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


def evaluate_methods(g_obs, i_obs, g_true, i_true):
    methods = {
        "WSM": aggregate_mean(g_obs, i_obs),
        "ARA_one": ara_operator(g_obs, i_obs, 1.0),
        "ARA_phi": ara_operator(g_obs, i_obs, PHI),
        "ARA_phi2": ara_operator(g_obs, i_obs, PHI**2),
    }

    oracle = oracle_scores(g_true, i_true)
    oracle_w = winners(oracle)
    oracle_best = oracle[oracle_w, np.arange(oracle.shape[1])]

    rows = []

    for name, x in methods.items():
        mean_scores = score_mean(x)
        top_scores = score_topsis(x)
        prom_scores = score_promethee_ii(x)

        for rule_name, scores in {
            "mean": mean_scores,
            "TOPSIS": top_scores,
            "PROMETHEE_II": prom_scores,
        }.items():
            w = winners(scores)
            chosen = oracle[w, np.arange(oracle.shape[1])]
            regret = oracle_best - chosen
            freq = np.bincount(w, minlength=N_PROJECTS) / oracle.shape[1]

            rows.append({
                "method": f"{name}+{rule_name}",
                "accuracy_vs_oracle": float(np.mean(w == oracle_w)),
                "mean_regret": float(np.mean(regret)),
                "p95_regret": float(np.quantile(regret, 0.95)),
                "winner_entropy": float(-np.sum(freq * np.log(freq + 1e-12))),
                "P(win A)": float(freq[0]),
                "P(win B)": float(freq[1]),
                "P(win C)": float(freq[2]),
                "P(win D)": float(freq[3]),
            })

    # direct MCDA baselines on plain merged matrix
    x_plain = aggregate_mean(g_obs, i_obs)
    for rule_name, scores in {
        "TOPSIS_direct": score_topsis(x_plain),
        "PROMETHEE_II_direct": score_promethee_ii(x_plain),
        "WSM_direct": score_mean(x_plain),
    }.items():
        w = winners(scores)
        chosen = oracle[w, np.arange(oracle.shape[1])]
        regret = oracle_best - chosen
        freq = np.bincount(w, minlength=N_PROJECTS) / oracle.shape[1]

        rows.append({
            "method": rule_name,
            "accuracy_vs_oracle": float(np.mean(w == oracle_w)),
            "mean_regret": float(np.mean(regret)),
            "p95_regret": float(np.quantile(regret, 0.95)),
            "winner_entropy": float(-np.sum(freq * np.log(freq + 1e-12))),
            "P(win A)": float(freq[0]),
            "P(win B)": float(freq[1]),
            "P(win C)": float(freq[2]),
            "P(win D)": float(freq[3]),
        })

    return pd.DataFrame(rows)


def run_test(n_mc=100_000, sigma=0.8, seed=42):
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

    out = []
    for scenario_name, (g_obs, i_obs) in scenarios.items():
        df = evaluate_methods(g_obs, i_obs, g_true, i_true)
        df.insert(0, "scenario", scenario_name)
        out.append(df)

    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    df = run_test().round(6)
    print(df.sort_values(["scenario", "mean_regret", "accuracy_vs_oracle"]))
    df.to_csv("testC_mcda_noise_comparison_results.csv", index=False)
