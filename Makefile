PYTHON := python
VENV := .venv
IMAGE := mlops-week01-baseline:0.2.0

.PHONY: venv install dataset train sample predict all clean docker-build docker-all docker-train docker-predict compose-up compose-down

venv:
	$(PYTHON) -m venv $(VENV)

install:
	$(VENV)/Scripts/pip install -r requirements.txt

dataset:
	$(VENV)/Scripts/python make_dataset.py

train:
	$(VENV)/Scripts/python train.py

sample:
	$(VENV)/Scripts/python sample_input.py

predict:
	$(VENV)/Scripts/python predict.py --input data/sample_input.csv --output models/predictions.csv

all: dataset train sample predict

clean:
	rmdir /s /q models data 2>NUL || exit 0

docker-build:
	docker build -t $(IMAGE) .

docker-all:
	docker run --rm -v "$(CURDIR):/app" -w /app $(IMAGE) python make_dataset.py
	docker run --rm -v "$(CURDIR):/app" -w /app $(IMAGE) python train.py
	docker run --rm -v "$(CURDIR):/app" -w /app $(IMAGE) python sample_input.py
	docker run --rm -v "$(CURDIR):/app" -w /app $(IMAGE) python predict.py --input data/sample_input.csv --output models/predictions.csv

docker-train:
	docker run --rm -v "$(CURDIR):/app" -w /app $(IMAGE) python train.py

docker-predict:
	docker run --rm -v "$(CURDIR):/app" -w /app $(IMAGE) python predict.py --input data/sample_input.csv --output models/predictions.csv

compose-up:
	docker compose up --build --abort-on-container-exit --exit-code-from predict

compose-down:
	docker compose down --remove-orphans
