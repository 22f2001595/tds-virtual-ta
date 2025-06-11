from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List
import base64

app = FastAPI()
@app.get("/")
def read_root():
    return {"message": "Welcome to the TDS Virtual TA!"}


class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None  # base64 encoded image string

@app.post("/api/")
async def answer_question(req: QuestionRequest):
    # Dummy response for now
    return {
        "answer": "This is a placeholder answer. Real logic will be added later.",
        "links": [
            {
                "url": "https://example.com",
                "text": "Example discussion"
            }
        ]
    }

