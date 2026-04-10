FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir runpod && \
    pip install --no-cache-dir chatterbox-tts --no-deps && \
    pip install --no-cache-dir transformers safetensors conformer diffusers==0.29.0 \
    einops huggingface-hub resemble-perth s3tokenizer soundfile audioread \
    librosa omegaconf cfgv pydantic click colorama pyloudnorm numpy==1.26.4 \
    accelerate sentencepiece protobuf

RUN python -c "from chatterbox.tts_turbo import ChatterboxTurboTTS; ChatterboxTurboTTS.from_pretrained(device='cpu')"

COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
