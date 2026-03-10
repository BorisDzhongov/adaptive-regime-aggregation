# Adaptive Regime Aggregation (ARA)

[![Paper DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18810210.svg)](https://doi.org/10.5281/zenodo.18810210)
[![Software DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18945818.svg)](https://doi.org/10.5281/zenodo.18945818)

Reference implementation of Adaptive Regime Aggregation (ARA), a formal framework for aggregating heterogeneous criteria and coordinating multiple informational subsystems under limited information.

The method is based on a strictly convex quadratic coordination functional that yields a unique global solution under explicit structural assumptions.

ARA is designed for decision systems in which multiple informational regimes (e.g. data-driven and intuition-based subsystems) must be aggregated while preserving structural balance.

## Paper

Adaptive Regime Aggregation (ARA): A Design-First Architecture for Structured Balancing under Limited Information

Paper DOI:  
https://doi.org/10.5281/zenodo.18810210

## Software archive

Zenodo software archive (reference implementation):

https://doi.org/10.5281/zenodo.18945818

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
```

## Reproducibility

The repository contains the reference implementation used to generate the experimental results reported in the ARA manuscript, including deterministic examples and Monte Carlo validation experiments.

## Citation

If you use Adaptive Regime Aggregation (ARA) in your work, please cite the paper:

```
Dzhongov, B. (2026).
Adaptive Regime Aggregation (ARA): A Design-First Architecture for Structured Balancing under Limited Information.
Zenodo.
https://doi.org/10.5281/zenodo.18810210
```

Software reference implementation:

```
Dzhongov, B. (2026).
Adaptive Regime Aggregation (ARA): Reference implementation.
Zenodo.
https://doi.org/10.5281/zenodo.18945818
```

## License

MIT License
