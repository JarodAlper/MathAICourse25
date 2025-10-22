from google import genai
import random

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

sequence = [random.randint(0, 20) for _ in range(6)]
print("Sequence:", sequence)

prompt = f"Given the sequence {sequence}, what is the next number?  Give only the number."

print("Prompt:", prompt)

response = client.models.generate_content(
    model="gemini-2.0-flash", contents=prompt
)
print(response.text)

