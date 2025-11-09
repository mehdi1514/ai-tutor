from fastapi import FastAPI, HTTPException
from app.config import client, MODEL
from pydantic import BaseModel
from app.prompts import HINT_PROMPT
from google import genai
from app.cache import get_cached_hint, store_hint

app = FastAPI(title="TripleTen AI Tutor")

class HintRequest(BaseModel):
    question: str

class HintResponse(BaseModel):
    hint: str

@app.get("/")
def root():
    return {"msg": "hello future AI tutor"}

@app.post("/generate_hint", response_model=HintResponse)
def generate_hint(body: HintRequest):
    # 1️⃣ try cache first
    if (cached := get_cached_hint(body.question)):
        return HintResponse(hint=cached)

    # 2️⃣ otherwise call Gemini and store the hint with its quesiton in cache
    try:
        prompt = HINT_PROMPT.format(question=body.question)
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=80
            )
        )
        hint = response.text.strip()
        # 3️⃣ save for next time
        store_hint(body.question, hint)
        return HintResponse(hint=hint)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))