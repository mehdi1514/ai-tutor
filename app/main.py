from fastapi import FastAPI, HTTPException
from app.config import client, MODEL
from pydantic import BaseModel
from app.prompts import HINT_PROMPT
from google import genai

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
        return HintResponse(hint=hint)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))