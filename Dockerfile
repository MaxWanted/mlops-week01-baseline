FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN useradd --create-home --shell /bin/bash appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY make_dataset.py train.py sample_input.py predict.py ./

RUN chown -R appuser:appuser /app
USER appuser

CMD ["python", "train.py"]

