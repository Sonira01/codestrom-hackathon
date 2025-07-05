# Use official slim Python base image
FROM python:3.10-slim

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt && pip cache purge

# Copy all project files
COPY . .

# Expose the port your app runs on
EXPOSE 8000

# Start the Flask app with Gunicorn
CMD ["gunicorn", "model.app:app", "--chdir", "/app", "--bind", "0.0.0.0:8000"]
