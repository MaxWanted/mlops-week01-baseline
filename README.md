# mlops-week01-baseline

Week 01 + Week 02 MLOps foundations:
- Week 01: train/evaluate/predict baseline.
- Week 02: Docker and Docker Compose reproducibility.

## Goal

Build minimal reproducible ML baseline that:

1. Creates dataset.
2. Trains model.
3. Saves metrics and artifact.
4. Runs batch prediction from CSV.

## Stack

- Python 3.10+
- scikit-learn
- pandas
- joblib

## Project layout

- `make_dataset.py` - creates `data/dataset.csv`.
- `train.py` - trains classifier and writes:
  - `models/model.joblib`
  - `models/metrics.json`
- `sample_input.py` - creates sample features CSV.
- `predict.py` - runs inference on CSV and writes `models/predictions.csv`.

## Quickstart

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
python make_dataset.py
python train.py
python sample_input.py
python predict.py --input data/sample_input.csv --output models/predictions.csv
```

## Make commands

```bash
make venv
make install
make all
```

## Week 02: Docker reproducible training

Build image:

```bash
docker build -t mlops-week01-baseline:0.2.0 .
```

Run full pipeline in containers with local artifact persistence:

```bash
docker run --rm -v "$PWD:/app" -w /app mlops-week01-baseline:0.2.0 python make_dataset.py
docker run --rm -v "$PWD:/app" -w /app mlops-week01-baseline:0.2.0 python train.py
docker run --rm -v "$PWD:/app" -w /app mlops-week01-baseline:0.2.0 python sample_input.py
docker run --rm -v "$PWD:/app" -w /app mlops-week01-baseline:0.2.0 python predict.py --input data/sample_input.csv --output models/predictions.csv
```

Same flow through `make`:

```bash
make docker-build
make docker-all
```

Run the full flow with Docker Compose:

```bash
docker compose up --build --abort-on-container-exit --exit-code-from predict
```

Or via `make`:

```bash
make compose-up
```

## Expected output

After run:

- model artifact at `models/model.joblib`
- training metrics at `models/metrics.json`
- predictions at `models/predictions.csv`

## Notes

- Seed fixed to `42` for reproducibility.
- Dataset used: scikit-learn breast cancer dataset.
- Docker image pins dependency versions through `requirements.txt`.

## Milestones

- `v0.1-week01-baseline` - pure Python baseline.
- `v0.2-week02-docker` - Dockerfile + docker run + docker compose flow.
