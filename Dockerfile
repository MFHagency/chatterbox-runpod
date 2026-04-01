FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y python3.11 python3-pip git ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Chatterbox (brings torch 2.6.0, transformers 5.2.0, etc.)
RUN pip3 install --no-cache-dir git+https://github.com/resemble-ai/chatterbox.git

# Install runpod SDK
RUN pip3 install --no-cache-dir runpod

# Pre-download model at build time
RUN python3 -c "from chatterbox.tts import ChatterboxTTS; ChatterboxTTS.from_pretrained(device='cpu')" || true

COPY handler.py /app/handler.py

CMD ["python3", "-u", "handler.py"]
