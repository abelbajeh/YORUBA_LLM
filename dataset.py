# """
# Extracts the first N clips (audio + ground-truth transcript) from a locally
# downloaded parquet file (e.g. test.parquet, already pulled via curl).

# No network access needed — everything here reads from disk.

# Setup:
#     pip install pandas soundfile

# Run:
#     python extract_from_parquet.py
# """

# import os
# import io
# import pandas as pd
# import soundfile as sf

# # ---- config ----
# PARQUET_PATH = "test.parquet"   # the file you already downloaded via curl
# NUM_CLIPS = 100
# OUTPUT_DIR = "yoruba_first_10_clips"


# def main():
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     print(f"Reading {PARQUET_PATH}...")
#     df = pd.read_parquet(PARQUET_PATH)
#     print(f"Columns: {list(df.columns)}")
#     print(f"Total rows: {len(df)}\n")

#     n = min(NUM_CLIPS, len(df))

#     for i in range(n):
#         row = df.iloc[i]

#         audio_bytes = row["audio"]["bytes"]  # raw MP3 bytes embedded in the parquet
#         ground_truth = row["sentence"].strip()

#         # Save the raw audio bytes as an .mp3 first (fastest, no re-encoding)
#         mp3_path = os.path.join(OUTPUT_DIR, f"clip_{i}.mp3")
#         with open(mp3_path, "wb") as f:
#             f.write(audio_bytes)

#         # Also decode + re-save as .wav for scripts that expect wav directly
#         # (soundfile needs a modern libsndfile with mp3 support; if this
#         # errors, just use the .mp3 file with librosa.load(..., sr=16000)
#         # instead, same as your earlier m4a/mp3 handling)
#         try:
#             data, samplerate = sf.read(io.BytesIO(audio_bytes))
#             wav_path = os.path.join(OUTPUT_DIR, f"clip_{i}.wav")
#             sf.write(wav_path, data, samplerate)
#             duration = len(data) / samplerate
#         except Exception as e:
#             wav_path = None
#             duration = None
#             print(f"  (Could not decode to wav directly: {e} — .mp3 saved, convert with ffmpeg if needed)")

#         txt_path = os.path.join(OUTPUT_DIR, f"clip_{i}.txt")
#         with open(txt_path, "w", encoding="utf-8") as f:
#             f.write(ground_truth)

#         dur_str = f"{duration:.2f}s" if duration else "unknown duration"
#         print(f"clip_{i}  ({dur_str})")
#         print(f"  Ground truth: {ground_truth}")
#         print(f"  Saved: {mp3_path}" + (f" and {wav_path}" if wav_path else ""))
#         print()

#     print(f"Done — {n} clips extracted to {OUTPUT_DIR}/")


# if __name__ == "__main__":
#     main()

from datasets import load_dataset

ds = load_dataset("google/fleurs", "yo_ng", split="test", streaming=True)

rows = []
for i, ex in enumerate(ds):
    if i >= 150:
        break
    rows.append(ex)

import pandas as pd
pd.DataFrame(rows).to_parquet("test.parquet")