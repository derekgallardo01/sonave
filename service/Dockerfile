# Sonave detection microservice.
# Base image ships CUDA 12.8 runtime so the cu128 torch wheels get the GPU.
# (For a CPU-only deploy, swap to python:3.11-slim and the CPU torch index.)
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3-pip ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch first, from the cu128 index (matches the training env).
RUN pip3 install --no-cache-dir torch torchaudio \
        --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt service/requirements.txt
RUN pip3 install --no-cache-dir \
        transformers>=4.44 huggingface_hub librosa soundfile numpy scikit-learn && \
    pip3 install --no-cache-dir -r service/requirements.txt

# App code + model. Keep only the deployment model to slim the image.
COPY src/ /app/src/
COPY config.py /app/config.py
COPY service/ /app/service/
COPY models/sonave_xlsr_rw/ /app/models/sonave_xlsr_rw/

ENV SONAVE_MODEL=/app/models/sonave_xlsr_rw
EXPOSE 8000
# detector.py adds src/ to path; run from /app so imports resolve.
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
