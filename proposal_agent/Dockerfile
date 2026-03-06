FROM python:3.10-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Install repo-level requirements by default; proposal_agent may include extras
COPY requirements.txt requirements.lock ./
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt || true

COPY . /app

ENV PYTHONUNBUFFERED=1

CMD ["python", "proposal_agent/cli.py"]
