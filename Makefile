PYTHON := python
VENV := .venv

.PHONY: venv install dataset train sample predict all clean

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
