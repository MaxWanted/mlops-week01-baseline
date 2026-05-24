import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

LABEL_MAP = {0: "malignant", 1: "benign"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/sample_input.csv",
        help="CSV file with feature columns only.",
    )
    parser.add_argument(
        "--output",
        default="models/predictions.csv",
        help="Path to save predictions CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path("models/model.joblib")
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run `python train.py` first.")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file `{input_path}` not found. Run `python sample_input.py` first."
        )

    model = joblib.load(model_path)
    features = pd.read_csv(input_path)

    pred = model.predict(features)
    proba = model.predict_proba(features)[:, 1]

    out_df = features.copy()
    out_df["prediction"] = pred
    out_df["prediction_label"] = [LABEL_MAP[int(v)] for v in pred]
    out_df["probability"] = proba

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    summary = {
        "input_rows": int(len(features)),
        "output_path": str(output_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
