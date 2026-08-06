import asyncio
import time
import torch
import librosa
from transformers import pipeline
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_HUB_OFFLINE"] = "1"

# 1. Model Configuration
MODEL_ID = "NCAIR1/Yoruba-ASR"
AUDIO_FILE = "recording3.wav"  # Replace with a path to a Yoruba audio file

# NCAIR1/Yoruba-ASR is a GATED repo: you must (1) accept its terms while logged
# in at https://huggingface.co/NCAIR1/Yoruba-ASR, and (2) authenticate with a
# token, or the Hub API silently returns an empty file list and transformers
# throws "no file named pytorch_model.bin or model.safetensors" even though
# the files exist. Get a token at https://huggingface.co/settings/tokens
HF_TOKEN = os.environ.get("HF_TOKEN")

def load_stt_pipeline():
    print(f"Loading {MODEL_ID}...")

    # Fixed: correct priority order so CUDA isn't silently overridden by MPS
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=MODEL_ID,
        device=device,
        token=HF_TOKEN,  # required for gated repos like this one
        # Removed use_safetensors=False: that flag forces the loader to look
        # ONLY for pytorch_model.bin. This repo ships model.safetensors instead,
        # so forcing False caused "no file named pytorch_model.bin" errors.
        # Let transformers auto-detect the correct weight format.
        #
        # chunk_length_s/stride_length_s are only needed for audio longer than
        # Whisper's native 30s window. Omitted here since the test clip is ~4s;
        # add them back (chunk_length_s=30, stride_length_s=5) if you later
        # test with longer recordings.
    )
    print(f"Model loaded successfully on {device}.")
    return asr_pipe


def prepare_audio(file_path):
    """Loads and resamples audio to 16kHz (Whisper's requirement)."""
    print(f"Loading audio from {file_path}...")
    audio, _ = librosa.load(file_path, sr=16000)
    return audio


async def transcribe_audio_async(asr_pipe, audio_array):
    """Runs the transcription asynchronously to prevent blocking."""
    print("Transcribing...")
    loop = asyncio.get_running_loop()

    start_time = time.time()
    # Fixed: explicitly pass sampling_rate so the pipeline never has to guess it
    result = await loop.run_in_executor(
    None,
    lambda: asr_pipe(
        {"array": audio_array, "sampling_rate": 16000},
        generate_kwargs={"num_beams": 1, "language": "yoruba", "task": "transcribe"},
    ),
)
    end_time = time.time()

    transcription_time = end_time - start_time

    return result["text"], transcription_time


async def main():
    try:
        asr_pipe = load_stt_pipeline()
        audio_array = prepare_audio(AUDIO_FILE)

        text, latency = await transcribe_audio_async(asr_pipe, audio_array)

        print("\n--- Transcription Results ---")
        print(f"Text: {text}")
        print(f"Latency: {latency:.2f} seconds")
        print("---------------------------\n")

    except FileNotFoundError:
        print(f"Error: Could not find the audio file at '{AUDIO_FILE}'. Please ensure the path is correct.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())