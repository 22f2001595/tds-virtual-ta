from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import base64
import easyocr
from io import BytesIO
from PIL import Image
import numpy as np

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the TDS Virtual TA!"}

class QuestionRequest(BaseModel):
    question: Optional[str] = None
    image: Optional[str] = None  # base64 encoded image string

@app.post("/api/")
async def answer_question(req: QuestionRequest):
    extracted_text = None

    if req.image:
        try:
            image_data = base64.b64decode(req.image)
            image = Image.open(BytesIO(image_data)).convert("RGB")
            image_np = np.array(image)
            reader = easyocr.Reader(['en'])
            result = reader.readtext(image_np)
            extracted_text = " ".join([res[1] for res in result])
        except Exception as e:
            extracted_text = f"Error in image processing: {str(e)}"

    return {
        "original_question": req.question,
        "image_text": extracted_text,
        "message": "Processed successfully!"
    }
