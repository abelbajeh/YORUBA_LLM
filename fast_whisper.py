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

# Update this to your exact absolute path
AUDIO_DIRECTORY = "/home/abelbajeh/yoruba_llm/yoruba_first_10_clips" 
NUM_CLIPS = 100

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
    print(f"🔍 Checking for audio clips in: {AUDIO_DIRECTORY}")
    print(f"🚀 Starting Groq transcription...\n")

    for i in range(NUM_CLIPS):
        audio_path = os.path.join(AUDIO_DIRECTORY, f"clip_{i}.mp3")
        txt_path = os.path.join(AUDIO_DIRECTORY, f"clip_{i}.txt")
        
        if not os.path.exists(audio_path):
            print(f"clip_{i}: audio file not found, skipping\n")
            continue

        # Load ground truth if available
        ground_truth = ""
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                ground_truth = f.read().strip()
                
        predicted_text = transcribe_audio_with_groq(audio_path)

        print(f"--- clip_{i} ---")
        print(f"Ground truth: {ground_truth}")
        print(f"Predicted   : {predicted_text}")
        print()

if __name__ == "__main__":
    main()