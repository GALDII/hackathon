import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

print("Attempting to call Groq API...")

try:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": "Explain the importance of low latency LLMs"}],
        model="llama-3.1-8b-instant",
    )

    print("\n✅ API Call Successful!")
    print(chat_completion.choices[0].message.content)

except Exception as e:
    print("\n❌ API Call Failed!")
    print(f"Error: {e}")