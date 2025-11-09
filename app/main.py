from fastapi import FastAPI

app = FastAPI(title="TripleTen AI Tutor")

@app.get("/")
def root():
    return {"msg": "hello future AI tutor"}