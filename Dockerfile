FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN pip install --no-cache-dir --upgrade transformers>=4.45.0

RUN pip install --no-cache-dir runpod

RUN pip install --no-cache-dir git+https://github.com/resemble-ai/chatterbox.git

# Pre-download model at build time so cold starts are fast
RUN python3 -c "from chatterbox.tts import ChatterboxTTS; ChatterboxTTS.from_pretrained(device='cpu')" || true

COPY handler.py /app/handler.py

CMD ["python3", "-u", "handler.py"]
