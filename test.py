import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ---- config ----
MODEL = "llama-3.3-70b-versatile"   # swap this to test other models
SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and concisely."

key = os.environ.get("GROQ_TOKEN")
client = Groq(api_key=key)


def ask(prompt: str, model: str = MODEL) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[  
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(f"Testing model: {MODEL}\n(type 'exit' to quit)\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        try:
            reply = ask(user_input)
            print(f"\n{MODEL}: {reply}\n")
        except Exception as e:
            print(f"Error: {e}\n")