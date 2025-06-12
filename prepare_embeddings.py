import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Sample Discourse posts or questions (for now hardcoded)
texts = [
    "What is the deadline for assignment submission?",
    "How do I join the Zoom session?",
    "What are the prerequisites for the exam?",
    "The website is not loading properly.",
    "Where can I find the lecture notes?"
]

# 2. Load local sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Convert texts to embeddings
embeddings = model.encode(texts, show_progress_bar=True)
embeddings_np = np.array(embeddings).astype("float32")

# 4. Save embeddings into FAISS index
dimension = embeddings_np.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_np)

# 5. Save index and original texts to files
os.makedirs("embeddings", exist_ok=True)
faiss.write_index(index, "embeddings/faiss_index.bin")

with open("embeddings/documents.pkl", "wb") as f:
    pickle.dump(texts, f)

with open("embeddings/embeddings.pkl", "wb") as f:
    pickle.dump(embeddings_np, f)

print("✅ Embeddings and FAISS index saved.")
