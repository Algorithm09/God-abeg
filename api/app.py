import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import hashlib
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# --- Hugging Face API ---
api_key = os.getenv("huggingface_api_key")
client = InferenceClient(api_key=api_key)
model = "openai/gpt-oss-20b"

# --- Lazy-loaded globals ---
embedding_model = None
chroma_client = None
collection = None
embedding_cache = {}
session_queries = {}

MAX_QUERIES = 5
base_prompt = "You are a helpful customer service bot that answers based on the provided context."

# --- FastAPI app ---
app = FastAPI(title="HelpGenie Chatbot API")

# --- Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper: Lazy-load models ---
def load_models():
    global embedding_model, chroma_client, collection
    if embedding_model is None:
        from sentence_transformers import SentenceTransformer
        import chromadb
        from chromadb.config import Settings

        # Load embedding model
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Try to load persistent DB, fallback to in-memory
        try:
            chroma_client = chromadb.PersistentClient(path="./faq_chromadb")
        except Exception:
            chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=None))

        collection = chroma_client.get_or_create_collection("faq_collection")

# --- Helper: Cache embeddings ---
def get_cached_embedding(text):
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if key in embedding_cache:
        return embedding_cache[key]
    emb = embedding_model.encode([text])[0].tolist()
    embedding_cache[key] = emb
    return emb

# --- RAG chat function ---
def chat(user_query: str):
    load_models()  # lazy-load on first request
    try:
        query_emb = get_cached_embedding(user_query)
        results = collection.query(query_embeddings=[query_emb], n_results=2)
        retrieved_docs = results.get("documents", [[]])[0]
        context = "\n\n".join(retrieved_docs) if retrieved_docs else "No relevant information found."

        rag_prompt = f"Context:\n{context}\n\nQuestion:\n{user_query}\n\nAnswer based on the above context. If not sure, politely say you don't have enough information."

        messages = [
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": rag_prompt},
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.choices[0].message["content"]

    except Exception as e:
        return f"[Error] Unexpected issue: {e}"

# --- Request schema ---
class ChatRequest(BaseModel):
    session_id: str
    query: str

# --- Chat endpoint ---
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    user_query = request.query.strip()

    count = session_queries.get(session_id, 0)
    if count >= MAX_QUERIES:
        raise HTTPException(status_code=403, detail="Max queries reached.")

    session_queries[session_id] = count + 1
    answer = chat(user_query)

    return {
        "answer": answer,
        "queries_used": session_queries[session_id],
        "queries_remaining": MAX_QUERIES - session_queries[session_id]
    }
