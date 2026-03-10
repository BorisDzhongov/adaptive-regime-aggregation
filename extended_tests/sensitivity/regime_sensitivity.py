import math
import csv
from pathlib import Path

# Fixed deterministic scores from Example II
# Replace these later if you want to connect directly to the core ARA code.
PROJECTS = ["A", "B", "C", "D"]

G = {
    "A": [8.5, 7.5, 6.0, 6.5, 6.0, 7.0, 5.5, 8.0, 6.5, 7.0, 6.0],
    "B": [7.0, 8.5, 8.1, 7.0, 7.5, 6.5, 7.0, 6.0, 8.0, 7.5, 7.2],
    "C": [9.0, 6.5, 5.5, 5.0, 6.5, 7.5, 8.0, 7.0, 6.0, 5.5, 6.5],
    "D": [6.5, 6.2, 8.0, 8.5, 7.0, 6.0, 7.5, 8.5, 7.0, 8.0, 8.0],
}

I = {
    "A": [5.0, 6.0, 6.5, 6.0, 5.5, 6.5, 6.0, 5.5, 6.0, 6.0, 5.5],
    "B": [6.5, 6.0, 6.0, 6.5, 6.0, 6.0, 6.5, 6.0, 6.5, 6.0, 6.0],
    "C": [5.5, 6.5, 7.0, 6.5, 6.0, 6.5, 6.0, 6.5, 6.0, 6.5, 6.0],
    "D": [7.0, 6.5, 6.5, 7.0, 6.5, 6.5, 6.0, 6.5, 6.5, 6.0, 6.5],
}

ANCHOR = 5.0

REGIMES = {
    "phi_inv2": 1 / (1.61803398875 ** 2),
    "phi_inv": 1 / 1.61803398875,
    "one": 1.0,
    "phi": 1.61803398875,
    "phi2": 1.61803398875 ** 2,
}


def ara_score(g_values, i_values, alpha, anchor=5.0):
    """
    Simple deterministic coordination score for exploratory regime scan.
    This is an extended test, not the official paper pipeline.
    """
    total = 0.0
    for g, i in zip(g_values, i_values):
        coordinated = (g + alpha * i) / (1.0 + alpha)
        penalty = ((g - i) ** 2) / (1.0 + alpha)
        total += coordinated - 0.05 * penalty + 0.02 * (coordinated - anchor)
    return total / len(g_values)


def rank_scores(score_dict):
    return sorted(score_dict.items(), key=lambda x: x[1], reverse=True)


def main():
    out_dir = Path("extended_tests/results/csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "regime_sensitivity_scores.csv"

    rows = []
    for regime_name, alpha in REGIMES.items():
        scores = {}
        for p in PROJECTS:
            scores[p] = ara_score(G[p], I[p], alpha, ANCHOR)

        ranking = rank_scores(scores)
        winner = ranking[0][0]

        for rank, (project, score) in enumerate(ranking, start=1):
            rows.append({
                "regime": regime_name,
                "alpha": alpha,
                "project": project,
                "score": round(score, 6),
                "rank": rank,
                "winner": winner,
            })

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["regime", "alpha", "project", "score", "rank", "winner"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
