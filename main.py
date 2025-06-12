import os
import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv

# === Constants ===
NOTES_INDEX_FILE = "notes_faiss.index"
NOTES_METADATA_FILE = "notes_embeddings_metadata.json"
NOTES_DIR = "markdown_files"

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load sentence-transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load discourse posts
try:
    with open("discourse_posts.json", "r", encoding="utf-8") as f:
        discourse_posts = json.load(f)
except FileNotFoundError:
    logger.warning("❌ discourse_posts.json file not found.")
    discourse_posts = []

# Load notes metadata and index
try:
    with open(NOTES_METADATA_FILE, "r", encoding="utf-8") as f:
        notes_metadata = json.load(f)
    notes_index = faiss.read_index(NOTES_INDEX_FILE)
except FileNotFoundError:
    logger.warning("❌ Notes FAISS index or metadata not found.")
    notes_metadata = []
    notes_index = None

# Prepare discourse index
corpus_embeddings = []
metadata_list = []
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

# Request schema
class QuestionRequest(BaseModel):
    question: str
    source: str = "all"  # "discourse", "note", or "all"

# Search helpers
def semantic_search_index(index, metadata_list, query, top_k):
    query_vector = model.encode(query)
    query_vector = np.array([query_vector])
    D, I = index.search(query_vector, top_k)
    return [metadata_list[i] for i in I[0]]

def read_note_content(filename):
    path = os.path.join(NOTES_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return "".join(lines[4:12]).strip()  # Skip frontmatter, return first ~8 lines
    return "(Note content unavailable)"

# Main search logic
def semantic_search(query: str, top_k: int = 3, source: str = "all"):
    results = []

    if source in ("discourse", "all") and discourse_index:
        discourse_results = semantic_search_index(discourse_index, metadata_list, query, top_k)
        results.extend(discourse_results)

    if source in ("note", "all") and notes_index:
        notes_results = semantic_search_index(notes_index, notes_metadata, query, top_k)
        for note in notes_results:
            note_result = note.copy()
            note_result["content"] = read_note_content(note["filename"])
            note_result["url"] = ""
            note_result["source"] = "note"
            results.append(note_result)

    return results[:top_k]

# API endpoint
@app.post("/api/")
async def answer_question(req: QuestionRequest):
    try:
        results = semantic_search(req.question, top_k=6, source=req.source)
        return {"answers": results}
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
