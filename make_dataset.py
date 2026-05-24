from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer


def main() -> None:
    ds = load_breast_cancer(as_frame=True)
    df = ds.frame.copy()
    df.rename(columns={"target": "label"}, inplace=True)

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved dataset to {out_path}")


if __name__ == "__main__":
    main()

