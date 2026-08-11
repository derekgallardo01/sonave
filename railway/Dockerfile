# Sonave capture service — Railway deploy with full build control.
# Uses python:3.11-slim base; installs torch CPU + speechbrain explicitly.

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch CPU first (large wheels, separate layer for caching)
RUN pip install --no-cache-dir torch torchaudio \
        --index-url https://download.pytorch.org/whl/cpu

# Install the rest
COPY railway/requirements.txt /app/railway/requirements.txt
RUN pip install --no-cache-dir -r railway/requirements.txt

# Copy the whole repo (Railway needs both railway/ and service/)
COPY . /app/

# Enrollment + model cache on persistent volume
ENV SONAVE_ENROLL_DIR=/data/enrollments
ENV SONAVE_MODEL_CACHE=/data/models/ecapa
ENV SONAVE_DATA_DIR=/data/captured

WORKDIR /app/railway
EXPOSE 8000
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
