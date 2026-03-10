import subprocess
import sys

"""
Run all ARA experiments.

This script sequentially executes all experiment scripts
to reproduce the computational results.
"""

experiments = [
    "experiments/run_ara_example.py",
    "experiments/mc_prob_test_ARA_colab.py",
    "experiments/mc_test_a_colab.py",
    "experiments/mc_test_b_colab.py",
    "experiments/make_tables.py"
]

for exp in experiments:
    print(f"\nRunning {exp}...\n")

    result = subprocess.run(
        [sys.executable, exp],
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.stderr:
        print("Errors:")
        print(result.stderr)
