import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


SEED = 42
EXPERIMENT_NAME = "week03-baseline"


def main() -> None:
    np.random.seed(SEED)

    dataset_path = Path("data/dataset.csv")
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Dataset not found. Run `python make_dataset.py` first."
        )

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(dataset_path)
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    params = {"n_estimators": 200, "max_depth": 8, "random_state": SEED}
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, pred_proba)),
    }

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_param("train_rows", int(len(X_train)))
        mlflow.log_param("test_rows", int(len(X_test)))
        mlflow.log_param("seed", SEED)
        mlflow.sklearn.log_model(model, artifact_path="model")

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    out = model_dir / "metrics_mlflow.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"MLflow tracking URI: {tracking_uri}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

