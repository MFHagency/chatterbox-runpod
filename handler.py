import runpod
import torch
import torchaudio
import base64
import io
import tempfile
import os

from chatterbox.tts import ChatterboxTTS

print("Loading Chatterbox model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChatterboxTTS.from_pretrained(device=device)
print(f"Model loaded on {device}")

def handler(event):
    try:
        input_data = event["input"]
        text = input_data.get("text", "Hello, this is a test.")
        exaggeration = float(input_data.get("exaggeration", 0.5))
        cfg_weight = float(input_data.get("cfg_weight", 0.5))

        audio_prompt_path = None
        if "audio_prompt" in input_data:
            audio_bytes = base64.b64decode(input_data["audio_prompt"])
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(audio_bytes)
            tmp.close()
            audio_prompt_path = tmp.name

        wav = model.generate(
            text,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )

        if audio_prompt_path:
            os.unlink(audio_prompt_path)

        buf = io.BytesIO()
        torchaudio.save(buf, wav, model.sr, format="wav")
        buf.seek(0)
        audio_b64 = base64.b64encode(buf.read()).decode()
        duration_s = round(wav.shape[1] / model.sr, 2)

        return {
            "audio_base64": audio_b64,
            "sample_rate": model.sr,
            "duration_s": duration_s,
        }
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
