from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


SEED = 42


def main() -> None:
    np.random.seed(SEED)

    dataset_path = Path("data/dataset.csv")
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Dataset not found. Run `python make_dataset.py` first."
        )

    df = pd.read_csv(dataset_path)
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=SEED,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 6),
        "f1": round(f1_score(y_test, pred), 6),
        "roc_auc": round(roc_auc_score(y_test, pred_proba), 6),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "seed": SEED,
    }

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"
    metrics_path = model_dir / "metrics.json"

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

