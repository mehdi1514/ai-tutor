from dotenv import load_dotenv
import os
from google import genai

# 1️⃣ load .env into os.environ
load_dotenv()

# 2️⃣ pick Gemini 2.5 Flash-Lite (cheapest, fastest)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash-lite"   # <— new model