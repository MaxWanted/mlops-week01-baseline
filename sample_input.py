from pathlib import Path

import pandas as pd


def main() -> None:
    dataset_path = Path("data/dataset.csv")
    if not dataset_path.exists():
        raise FileNotFoundError("Dataset not found. Run `python make_dataset.py` first.")

    df = pd.read_csv(dataset_path)
    features = df.drop(columns=["label"]).head(10)

    out_path = Path("data/sample_input.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)
    print(f"Saved sample input to {out_path}")


if __name__ == "__main__":
    main()

