import pandas as pd
import glob

files = glob.glob("*.csv")

for f in files:
    df = pd.read_csv(f)

    latex = df.to_latex(
        index=False,
        float_format="%.4f"
    )

    out = f.replace(".csv",".tex")

    with open(out,"w") as t:
        t.write(latex)

    print("Created:", out)
