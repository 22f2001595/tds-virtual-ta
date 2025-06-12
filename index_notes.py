import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# === CONFIG ===
NOTES_DIR = "markdown_files"
METADATA_FILE = "metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_FILE = "notes_faiss.index"
EMBEDDINGS_METADATA_FILE = "notes_embeddings_metadata.json"

# === Load sentence transformer ===
model = SentenceTransformer(MODEL_NAME)

# === Load metadata ===
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata_entries = json.load(f)

corpus_embeddings = []
metadata_list = []

for entry in metadata_entries:
    filename = entry["filename"]
    filepath = os.path.join(NOTES_DIR, filename)

    if not os.path.exists(filepath):
        continue

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Skip YAML frontmatter (first 5–6 lines)
    try:
        start = lines.index('---\n')
        end = lines.index('---\n', start + 1)
        content = "".join(lines[end+1:]).strip()
    except ValueError:
        content = "".join(lines).strip()

    if not content:
        continue

    embedding = model.encode(content)
    corpus_embeddings.append(embedding)
    metadata_list.append({
        "title": entry["title"],
        "url": entry["original_url"],
        "filename": filename,
        "content": content[:500]  # truncate preview content
    })

# === Create FAISS index ===
if corpus_embeddings:
    dim = len(corpus_embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(corpus_embeddings))
    faiss.write_index(index, INDEX_FILE)

    with open(EMBEDDINGS_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2)

    print(f"✅ Saved FAISS index with {len(metadata_list)} notes.")
else:
    print("⚠️ No embeddings found.")
