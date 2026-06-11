import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json

from scripts.cpd_split_validate_impl import add_historical_event_priors

def parse_args():
    parser = argparse.ArgumentParser(description="Train Event Risk Calibrator to predict response probability.")
    parser.add_argument("--event_catalog_csv", default="output/rainfall_event_catalog/rainfall_event_catalog.csv")
    parser.add_argument("--output_model", default="output/event_calibrator.json")
    return parser.parse_args()

class EventCalibrator:
    def __init__(self, coefs, intercept, features):
        self.coefs = np.array(coefs, dtype=float)
        self.intercept = float(intercept)
        self.features = features

    def predict_proba(self, X_df):
        # Handle features mapping
        X_mat = []
        for feat in self.features:
            if feat in X_df.columns:
                X_mat.append(X_df[feat].fillna(0.0).to_numpy(dtype=float))
            else:
                X_mat.append(np.zeros(len(X_df)))
        X_mat = np.column_stack(X_mat)
        linear = np.dot(X_mat, self.coefs) + self.intercept
        return 1.0 / (1.0 + np.exp(-linear))

def load_calibrator(model_json_path):
    with open(model_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return EventCalibrator(data["coefs"], data["intercept"], data["features"])

def main():
    args = parse_args()
    df = pd.read_csv(args.event_catalog_csv)
    df = add_historical_event_priors(df)

    # Features selection
    features = [
        "historical_trigger_probability",
        "antecedent_wetness_index",
        "rain_mm",
        "spell_rain_sum",
        "spell_peak_rain",
        "spell_days",
        "pre3_rain_sum",
        "pre7_rain_sum",
        "pre15_rain_sum",
        "pre30_rain_sum"
    ]

    X = df[features].fillna(0.0)
    y = df["is_response_event_15d"].astype(int)

    # Train a logistic regression model
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, y)

    coefs = list(clf.coef_[0])
    intercept = float(clf.intercept_[0])

    # Save parameters as JSON
    model_data = {
        "coefs": coefs,
        "intercept": intercept,
        "features": features
    }
    
    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(model_data, f, indent=4)

    print(">> Event Risk Calibrator trained successfully!")
    print(f">> Saved calibrator model to: {output_path}")
    print(">> Model coefficients:")
    for feat, coef in zip(features, coefs):
        print(f"   - {feat}: {coef:.4f}")
    print(f"   - intercept: {intercept:.4f}")

    # Print training metrics
    calibrator = EventCalibrator(coefs, intercept, features)
    preds = calibrator.predict_proba(X)
    from sklearn.metrics import roc_auc_score, brier_score_loss
    auc = roc_auc_score(y, preds)
    brier = brier_score_loss(y, preds)
    print(f">> Training Metrics:")
    print(f"   - ROC-AUC: {auc:.4f}")
    print(f"   - Brier Score: {brier:.4f}")

if __name__ == "__main__":
    main()
