import subprocess
import sys

"""
Run all ARA experiments that are CI-compatible.
Colab-specific scripts are excluded from GitHub Actions.
"""

experiments = [
    "experiments/run_ara_example.py",
    "experiments/mc_prob_test_ARA_colab.py",
    "experiments/make_tables.py",
]

for exp in experiments:
    print(f"\nRunning {exp}...\n")

    result = subprocess.run(
        [sys.executable, exp],
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.returncode != 0:
        print("Errors:")
        print(result.stderr)
        raise SystemExit(result.returncode)
