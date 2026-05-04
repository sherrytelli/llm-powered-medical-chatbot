import os
import shutil
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from model import MedicalRAG

# Global state for the RAG instance
rag_bot: Optional[MedicalRAG] = None

def init_rag(kb_path: str = "knowledge/medquad.csv", index_path: str = "knowledge/faiss.index", chunks_path: str = "knowledge/chunks.json", embed_model: str = "nomic-embed-text:v1.5", chat_model: str = "phi3.5:3.8b", k: int = 3):
    global rag_bot
    rag_bot = MedicalRAG(
        kb_path=kb_path,
        index_path=index_path,
        chunks_path=chunks_path,
        embed_model=embed_model,
        chat_model=chat_model,
        k=k
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize RAG with default paths
    init_rag()
    yield
    # Shutdown: Cleanup if needed
    global rag_bot
    rag_bot = None

app = FastAPI(title="Medical RAG API", lifespan=lifespan)

# CORS for Next.js frontend (usually localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
def chat(query: str = Form(...), history: str = Form(default="[]")):
    """
    Handle chat inference.
    history is passed as a JSON string to keep it simple with Form data.
    """
    import json
    if not rag_bot:
        return {"response": "System not initialized.", "accepted": False}
    
    try:
        history_list = json.loads(history)
    except:
        history_list = []

    response = rag_bot.generate(query, history=history_list)
    return response

@app.post("/admin/rebuild-index")
def rebuild_index(file: UploadFile = File(...), kb_path: str = Form("knowledge/medquad.csv"), index_path: str = Form("knowledge/faiss.index"), chunks_path: str = Form("knowledge/chunks.json"), embed_model: str = Form("nomic-embed-text:v1.5"), chat_model: str = Form("phi3.5:3.8b"), k: int = Form(3)
):
    """
    Admin endpoint to upload a new CSV and rebuild the FAISS index.
    """
    global rag_bot
    
    # Save uploaded file to kb_path
    os.makedirs(os.path.dirname(kb_path) or ".", exist_ok=True)
    with open(kb_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    print(f"📥 Uploaded new dataset to {kb_path}")

    # Re-initialize RAG with new paths/config
    init_rag(
        kb_path=kb_path,
        index_path=index_path,
        chunks_path=chunks_path,
        embed_model=embed_model,
        chat_model=chat_model,
        k=int(k)
    )
    
    return {
        "status": "success",
        "message": f"Index rebuilt with {len(rag_bot.chunks)} chunks using model {rag_bot.chat_model}"
    }

@app.get("/admin/config")
def get_config():
    """Return current configuration."""
    global rag_bot
    if not rag_bot:
        return {"error": "System not initialized"}
    
    return {
        "kb_path": rag_bot.kb_path,
        "index_path": rag_bot.index_path,
        "chunks_path": rag_bot.chunks_path,
        "embed_model": rag_bot.embed_model,
        "chat_model": rag_bot.chat_model,
        "k": rag_bot.k
    }

@app.post("/admin/update-config")
def update_config(kb_path: str = Form("knowledge/medquad.csv"), index_path: str = Form("knowledge/faiss.index"), chunks_path: str = Form("knowledge/chunks.json"), embed_model: str = Form("nomic-embed-text:v1.5"), chat_model: str = Form("phi3.5:3.8b"), k: int = Form(3)
):
    """
    Update configuration without rebuilding index (if index already exists).
    """
    global rag_bot
    init_rag(
        kb_path=kb_path,
        index_path=index_path,
        chunks_path=chunks_path,
        embed_model=embed_model,
        chat_model=chat_model,
        k=int(k)
    )
    return {"status": "success", "message": "Configuration updated"}

