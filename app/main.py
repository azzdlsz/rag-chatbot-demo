"""FastAPI entrypoint for the RAG Chatbot demo."""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app import rag  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ALLOWED = {".pdf", ".txt", ".md"}
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", 10)) * 1024 * 1024
PUBLIC_DEMO = os.getenv("PUBLIC_DEMO", "").lower() in ("1", "true", "yes")
MAX_DOCS = int(os.getenv("MAX_DOCS", 20))
MAX_CHARS = int(os.getenv("MAX_DOC_CHARS", 200_000))
SAMPLES = set()

app = FastAPI(title="RAG Chatbot Demo", version="1.0.0")
store = rag.VectorStore()


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[dict] = []


@app.get("/health")
def health():
    return {"status": "ok", "public_demo": PUBLIC_DEMO, "max_upload_mb": MAX_BYTES // 1048576,
            "max_docs": MAX_DOCS, "mode": "mock" if rag.MOCK else "openai",
            "documents": len(store.sources()), "chunks": len(store.chunks)}


@app.get("/api/documents")
def documents():
    return [{"name": n, "chunks": c} for n, c in store.sources().items()]


@app.post("/api/documents")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"Unsupported file type {ext}. Allowed: {sorted(ALLOWED)}")
    if len(store.sources()) >= MAX_DOCS:
        raise HTTPException(429, f"Demo limit reached ({MAX_DOCS} documents). Delete one first.")
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_BYTES // 1048576} MB)")
    text = rag.extract_text(file.filename, data)[:MAX_CHARS]
    n = store.add(file.filename, rag.chunk_text(text))
    if n == 0:
        raise HTTPException(422, "No extractable text found in file")
    return {"name": file.filename, "chunks": n}


@app.delete("/api/documents/{name}")
def delete(name: str):
    if name not in store.sources():
        raise HTTPException(404, "Document not found")
    if PUBLIC_DEMO and name in SAMPLES:
        raise HTTPException(403, "Sample documents can't be deleted in the public demo")
    with store.lock:
        store.remove(name)
    return {"deleted": name}


@app.post("/api/chat")
def chat(req: ChatRequest):
    hits = store.search(req.question)
    return {"answer": rag.answer(req.question, hits, req.history),
            "sources": [{"n": i, "source": c.source, "score": round(s, 3), "text": c.text[:300]}
                        for i, (c, s) in enumerate(hits, 1)]}


@app.on_event("startup")
def seed():
    """Index bundled sample docs on first run so the demo works out of the box."""
    SAMPLES.update(p.name for p in (ROOT / "sample_docs").glob("*"))
    if store.chunks or os.getenv("SKIP_SEED"):
        return
    for p in sorted((ROOT / "sample_docs").glob("*")):
        if p.suffix.lower() in ALLOWED:
            SAMPLES.add(p.name)
            store.add(p.name, rag.chunk_text(rag.extract_text(p.name, p.read_bytes())))


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")
