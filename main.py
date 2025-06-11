from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List
import base64

app = FastAPI()

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

