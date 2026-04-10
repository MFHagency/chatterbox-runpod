"""RunPod Serverless handler for Chatterbox TTS — Turbo + Multilingual with URL-based voice cloning."""
import runpod
import torch
import torchaudio as ta
import tempfile
import os
import subprocess
import base64
import urllib.request

from chatterbox.tts_turbo import ChatterboxTurboTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("Loading Turbo model...")
turbo_model = ChatterboxTurboTTS.from_pretrained(device="cuda")
print("Loading Multilingual model...")
multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
print("Both models loaded.")

def download_file(url, dest):
    urllib.request.urlretrieve(url, dest)
    print(f"Downloaded: {dest} ({os.path.getsize(dest)} bytes)")

def concatenate_audio_files(file_paths, output_path):
    if len(file_paths) == 1:
        subprocess.run(["ffmpeg","-y","-i",file_paths[0],"-ar","24000","-ac","1","-f","wav",output_path], capture_output=True, check=True)
        return
    filter_parts = []
    inputs = []
    for i, fp in enumerate(file_paths):
        inputs.extend(["-i", fp])
        filter_parts.append(f"[{i}:a]")
    filter_str = "".join(filter_parts) + f"concat=n={len(file_paths)}:v=0:a=1[out]"
    cmd = ["ffmpeg","-y"] + inputs + ["-filter_complex",filter_str,"-map","[out]","-ar","24000","-ac","1","-f","wav",output_path]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr.decode()}")
        raise Exception(f"ffmpeg failed: {result.stderr.decode()[:200]}")
    print(f"Concatenated {len(file_paths)} files -> {output_path} ({os.path.getsize(output_path)} bytes)")

def handler(event):
    inp = event.get("input", {})
    text = inp.get("text", "Hello world")
    model_type = inp.get("model", "turbo")
    audio_prompt_urls = inp.get("audio_prompt_urls", [])
    audio_prompt_base64 = inp.get("audio_prompt", None)
    exaggeration = inp.get("exaggeration", 0.5)
    cfg_weight = inp.get("cfg_weight", 0.5)
    language_id = inp.get("language_id", "en")

    print(f"Request: model={model_type}, text={text[:50]}..., urls={len(audio_prompt_urls)}, lang={language_id}")

    with tempfile.TemporaryDirectory() as tmpdir:
        ref_path = None

        if audio_prompt_urls:
            downloaded = []
            for i, url in enumerate(audio_prompt_urls):
                ext = ".m4a" if ".m4a" in url else ".wav" if ".wav" in url else ".mp3"
                local_path = os.path.join(tmpdir, f"sample_{i}{ext}")
                try:
                    download_file(url, local_path)
                    downloaded.append(local_path)
                except Exception as e:
                    print(f"Failed to download URL {i}: {e}")
            if downloaded:
                ref_path = os.path.join(tmpdir, "combined_reference.wav")
                try:
                    concatenate_audio_files(downloaded, ref_path)
                except Exception as e:
                    print(f"Concat failed: {e}, using first file")
                    ref_path = downloaded[0]
        elif audio_prompt_base64:
            ref_path = os.path.join(tmpdir, "reference.wav")
            with open(ref_path, "wb") as f:
                f.write(base64.b64decode(audio_prompt_base64))
            print(f"Base64 reference: {os.path.getsize(ref_path)} bytes")

        print(f"Reference: {ref_path}, exists={ref_path and os.path.exists(ref_path)}")

        if model_type == "multilingual":
            wav = multilingual_model.generate(
                text,
                audio_prompt_path=ref_path,
                language_id=language_id,
            )
        else:
            wav = turbo_model.generate(
                text,
                audio_prompt_path=ref_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )

        output_path = os.path.join(tmpdir, "output.wav")
        ta.save(output_path, wav, turbo_model.sr)
        with open(output_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        duration_s = wav.shape[-1] / turbo_model.sr
        print(f"Generated: {duration_s:.1f}s, model={model_type}")

        return {
            "audio_base64": audio_base64,
            "duration_s": round(float(duration_s), 2),
            "sample_rate": turbo_model.sr,
            "model_used": model_type,
        }

runpod.serverless.start({"handler": handler})
