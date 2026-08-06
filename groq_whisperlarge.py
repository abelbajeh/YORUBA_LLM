import os
import glob
try:
    from groq import Groq
except ImportError:
    print("Missing dependency! Please run: pip install groq")
    exit(1)

# This automatically looks for the GROQ_API_KEY variable in your OS environment.
# Make sure you have run: export GROQ_API_KEY="your_actual_key_here" in your terminal.
api_key = os.environ.get("GROQ_TOKEN")
if not api_key:
    print("❌ Error: GROQ_API_KEY environment variable not found!")
    print("Please set it using: export GROQ_API_KEY='your_api_key'")
    exit(1)

client = Groq(api_key=api_key)

# Change this to the exact folder where your clip_0.mp3 to clip_99.mp3 files are stored.
AUDIO_DIRECTORY = "yoruba_first_10_clips" 

def transcribe_audio_with_groq(file_path):
    """
    Sends an audio file to Groq's API and returns the Yoruba transcription.
    """
    try:
        with open(file_path, "rb") as file:
            # We explicitly tell Whisper it is listening to Yoruba ("yo") 
            # to prevent it from hallucinating English or other languages.
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(file_path), file.read()),
                model="whisper-large-v3",
                language="yo", 
                temperature=0.0 # Keep temperature at 0 for the most deterministic, accurate transcription
            )
            return transcription.text
    except Exception as e:
        return f"Error transcribing {file_path}: {str(e)}"

def main():
    print(f"🔍 Searching for audio files in: {AUDIO_DIRECTORY}")
    
    # Grab all .mp3 files in the directory and sort them so they process in order (clip_0, clip_1, etc.)
    audio_files = sorted(glob.glob(os.path.join(AUDIO_DIRECTORY, "*.mp3")))
    
    if not audio_files:
        print("⚠️ No .mp3 files found in the specified directory.")
        return

    print(f"🚀 Found {len(audio_files)} files. Starting Groq transcription...\n")

    for file_path in audio_files:
        filename = os.path.basename(file_path)
        print(f"--- {filename} ---")
        
        predicted_text = transcribe_audio_with_groq(file_path)
        print(f"Predicted : {predicted_text}\n")

if __name__ == "__main__":
    main()