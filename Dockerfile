# Use official Python image
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8000

# Adjust 'app:app' if your entrypoint is different
CMD ["gunicorn", "model.app:app", "--chdir", "/app", "--bind", "0.0.0.0:8000"]