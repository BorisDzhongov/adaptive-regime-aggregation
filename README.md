# Adaptive Regime Aggregation (ARA)

[![Paper DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18810210.svg)](https://doi.org/10.5281/zenodo.18810210)  
[![Software DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18945818.svg)](https://doi.org/10.5281/zenodo.18945818)

Reference implementation of **Adaptive Regime Aggregation (ARA)**, a formal framework for aggregating heterogeneous criteria and coordinating multiple informational subsystems under limited information.

The method is based on a strictly convex quadratic coordination functional that yields a **unique global solution** under explicit structural assumptions.

ARA is designed for decision systems in which multiple informational regimes (e.g. data-driven and intuition-based subsystems) must be aggregated while preserving **structural balance**.

---

# Paper

Adaptive Regime Aggregation (ARA): A Design-First Architecture for Structured Balancing under Limited Information

Paper DOI  
https://doi.org/10.5281/zenodo.18810210

---

# Software archive

Zenodo software archive (reference implementation)

https://doi.org/10.5281/zenodo.18945818

---

# Repository structure

- `ara/` — core ARA implementation  
- `data/` — input and experiment data  
- `experiments/` — deterministic examples and Monte Carlo experiments  
- `extended_tests/` — additional robustness and stress tests  
- `results/` — generated outputs  
- `paper/` — manuscript files  

---

# Experimental validation

The repository includes the experimental validation used in the ARA manuscript.

The experiments examine several properties of the method:

- **Regime sensitivity** — stability of rankings across coordination regimes  
- **Noise robustness** — behaviour of ARA under stochastic perturbations  
- **Architecture comparison** — performance relative to single-source aggregation  
- **Signal divergence analysis** — controlled transitions between informational regimes  

Additional stress tests and exploratory experiments are available in the  
`extended_tests/` directory.

---

# Running the experiments

Run all CI-compatible experiments with:

```bash
python experiments/run_all_tests.py
```

Generated outputs will be written to the `results/` directory.

---

# Reproducibility

This repository contains the **reference implementation used to generate the experimental results reported in the ARA manuscript**, including:

- deterministic worked examples
- Monte Carlo validation experiments
- extended robustness tests

The code and data are provided to ensure **full reproducibility of the reported results**.

---

# Citation

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

---

# License

MIT License
