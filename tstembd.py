

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client()

try:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents="Python interview preparation"
    )
    print("SUCCESS: Gemini embedding API works.")
    print("Vector dimensions:", len(response.embeddings[0].values))
except Exception as e:
    print(type(e).__name__)
    print(str(e))

