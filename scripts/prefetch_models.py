"""Download embed + rerank models at build time so runtime is fully offline."""
import os
from sentence_transformers import SentenceTransformer, CrossEncoder

SentenceTransformer(os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
CrossEncoder(os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
print("models cached")
