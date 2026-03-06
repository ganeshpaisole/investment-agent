FROM python:3.10-slim

WORKDIR /app

# Install build deps (if any) and cleanup
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests from repo root (workflow builds with repo root context)
COPY requirements.txt requirements.lock ./
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt || true

# Copy the full repo into the image
COPY . /app

ENV PYTHONUNBUFFERED=1

CMD ["python", "nse_agent/main.py"]
