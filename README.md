# Credit-Fraud-Detection-model
## Data and how to run (added during code repair, Nov 2026)

Both scripts expect **`creditcard.csv`** in this folder -- the Kaggle
"Credit Card Fraud Detection" dataset (284,807 transactions, columns
Time, V1-V28, Amount, Class). It is not committed here; download it from
Kaggle and place it next to the scripts.

To try the code without the Kaggle download, generate the small synthetic
stand-in (2,000 rows, same schema -- metrics on it are meaningless, it only
exercises the code):

```bash
pip install -r requirements.txt
python make_sample_data.py        # writes a SYNTHETIC creditcard.csv
python Simple_Fraud_detection_model.py   # trains/compares 8 classifiers
python Fraud_detection.py                # CV comparison, RF tuning, time-based
                                         # backtest, saves best_fraud_model.pkl,
                                         # then serves a Flask /predict endpoint
```

Note: on the full Kaggle dataset `Fraud_detection.py` is a long run
(cross-validation over four model families plus a random search); on the
synthetic sample it completes in a few minutes.
