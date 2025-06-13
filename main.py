import os
import json
import logging
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv

# === Constants ===
DISK_PATH = "/var/data"  # ✅ Used for Render persistent disk
NOTES_INDEX_FILE = os.path.join(DISK_PATH, "notes_faiss.index")
NOTES_METADATA_FILE = os.path.join(DISK_PATH, "notes_embeddings_metadata.json")
NOTES_DIR = "markdown_files"
UPLOAD_IMAGE_PATH = os.path.join(DISK_PATH, "uploaded_image.png")

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Lazy-loaded variables ===
model = None
discourse_index = None
notes_index = None
metadata_list = []
notes_metadata = []

# === Request schema ===
class QuestionRequest(BaseModel):
    question: str
    source: str = "all"  # discourse / note / all
    image: str | None = None  # Optional base64-encoded image

# === Helper ===
def load_model_and_indexes():
    global model, discourse_index, notes_index, metadata_list, notes_metadata

    if model is None:
        logger.info("🔁 Loading SentenceTransformer model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")

    if not metadata_list:
        try:
            with open("discourse_posts.json", "r", encoding="utf-8") as f:
                discourse_posts = json.load(f)
        except FileNotFoundError:
            logger.warning("❌ discourse_posts.json not found.")
            discourse_posts = []

        corpus_embeddings = []
        for post in discourse_posts:
            content = post.get("content", "")
            if content:
                embedding = model.encode(content)
                corpus_embeddings.append(embedding)
                metadata_list.append({
                    "title": post.get("topic_title", ""),
                    "url": post.get("url", ""),
                    "content": content,
                    "source": "discourse"
                })

        if corpus_embeddings:
            dimension = len(corpus_embeddings[0])
            discourse_index = faiss.IndexFlatL2(dimension)
            discourse_index.add(np.array(corpus_embeddings))
        else:
            discourse_index = None

    if notes_index is None:
        try:
            with open(NOTES_METADATA_FILE, "r", encoding="utf-8") as f:
                notes_metadata = json.load(f)
            notes_index = faiss.read_index(NOTES_INDEX_FILE)
        except FileNotFoundError:
            logger.warning("❌ Notes index or metadata not found.")
            notes_metadata = []
            notes_index = None

def semantic_search_index(index, metadata, query, top_k):
    query_vector = model.encode(query)
    query_vector = np.array([query_vector])
    D, I = index.search(query_vector, top_k)
    return [metadata[i] for i in I[0]]

def read_note_content(filename):
    path = os.path.join(NOTES_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return "".join(lines[4:12]).strip()
    return "(Note content unavailable)"

def semantic_search(query: str, top_k: int = 3, source: str = "all"):
    load_model_and_indexes()
    results = []

    if source in ("discourse", "all") and discourse_index:
        results.extend(semantic_search_index(discourse_index, metadata_list, query, top_k))

    if source in ("note", "all") and notes_index:
        note_results = semantic_search_index(notes_index, notes_metadata, query, top_k)
        for note in note_results:
            note_result = note.copy()
            note_result["content"] = read_note_content(note["filename"])
            note_result["url"] = ""
            note_result["source"] = "note"
            results.append(note_result)

    return results[:top_k]

@app.post("/api/")
async def answer_question(req: QuestionRequest):
    try:
        answers = semantic_search(req.question, top_k=6, source=req.source)

        if req.image:
            try:
                image_data = base64.b64decode(req.image)
                with open(UPLOAD_IMAGE_PATH, "wb") as f:
                    f.write(image_data)
                logger.info("🖼️ Image saved to uploaded_image.png")
            except Exception as e:
                logger.warning(f"⚠️ Failed to decode base64 image: {e}")

        return {"answers": answers}

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def home():
    return {"message": "TDS Virtual TA is running."}
