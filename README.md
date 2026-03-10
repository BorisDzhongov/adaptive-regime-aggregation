# Adaptive Regime Aggregation (ARA)

Reference implementation of Adaptive Regime Aggregation (ARA), a formal framework for aggregating heterogeneous criteria and coordinating multiple informational subsystems under limited information.

The method is based on a strictly convex quadratic coordination functional that yields a unique global solution under explicit structural assumptions.

ARA is designed for decision systems in which multiple informational regimes (e.g. data-driven and intuition-based subsystems) must be aggregated while preserving structural balance.

## Paper

Adaptive Regime Aggregation (ARA): A Design-First Architecture for Structured Balancing under Limited Information

Paper DOI: https://doi.org/10.5281/zenodo.18810210

## Repository structure

- `ara/` — core ARA implementation
- `data/` — input and experiment data
- `experiments/` — example runner and Monte Carlo experiments
- `results/` — generated outputs
- `paper/` — manuscript files

## Running the experiments

Run all CI-compatible experiments with:

```bash
python experiments/run_all_tests.py
