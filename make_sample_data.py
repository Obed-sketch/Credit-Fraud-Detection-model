"""Generate a small SYNTHETIC creditcard.csv sample.

The real dataset for this project is the Kaggle "Credit Card Fraud Detection"
dataset (creditcard.csv), which is not committed to this repo (and, per its
licence, should be downloaded from Kaggle rather than redistributed). This
generator creates 2,000 clearly synthetic rows with the same schema
(Time, V1-V28, Amount, Class) and a mild learnable signal in a few V features,
so both scripts in this repo can be run and verified end-to-end. Metrics
produced on this file are meaningless -- it exists to exercise the code.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 2000
n_fraud = 20  # ~1% positive class

features = {f"V{i}": rng.normal(size=n) for i in range(1, 29)}
df = pd.DataFrame(features)
df.insert(0, "Time", rng.uniform(0, 172800, size=n))
df["Amount"] = np.round(rng.lognormal(mean=3.5, sigma=1.0, size=n), 2)
df["Class"] = 0

fraud_idx = rng.choice(n, size=n_fraud, replace=False)
df.loc[fraud_idx, "Class"] = 1
# Inject a mild signal so the models have something real to learn.
for col in ["V4", "V11", "V12", "V14"]:
    df.loc[fraud_idx, col] -= rng.normal(2.0, 0.5, size=n_fraud)

cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
df[cols].to_csv("creditcard.csv", index=False)
print(f"Wrote creditcard.csv ({n} synthetic rows, {n_fraud} labelled fraud)")
