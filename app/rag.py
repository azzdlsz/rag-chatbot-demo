"""Core RAG logic: loading, chunking, embedding, vector search, answering."""
from __future__ import annotations
import hashlib, io, json, os, re, threading
from dataclasses import dataclass, asdict
import numpy as np
from pypdf import PdfReader

API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
BASE_URL = os.getenv("OPENAI_BASE_URL") or None
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
TOP_K = int(os.getenv("TOP_K", 4))
DATA_DIR = os.getenv("DATA_DIR", "./data")
MOCK = not API_KEY
MOCK_DIM = 512

_client = None
if not MOCK:
    from openai import OpenAI
    _client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


@dataclass
class Chunk:
    id: str
    source: str
    text: str


def extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    return data.decode("utf-8", errors="ignore")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):  # prefer to cut at sentence boundary
            cut = text.rfind(". ", start + size // 2, end)
            if cut != -1:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def embed(texts: list[str]) -> np.ndarray:
    if MOCK:  # deterministic hashing-trick bag of words, good enough for a demo
        vecs = np.zeros((len(texts), MOCK_DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in _tokens(t):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                vecs[i, h % MOCK_DIM] += 1.0
    else:
        resp = _client.embeddings.create(model=EMBED_MODEL, input=texts)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1, norms)


class VectorStore:
    """Tiny persistent cosine-similarity store backed by numpy."""

    def __init__(self, path: str = DATA_DIR):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.lock = threading.Lock()
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self._load()

    def _load(self):
        meta, vec = os.path.join(self.path, "chunks.json"), os.path.join(self.path, "vectors.npy")
        if os.path.exists(meta) and os.path.exists(vec):
            self.chunks = [Chunk(**c) for c in json.load(open(meta))]
            self.vectors = np.load(vec)

    def _save(self):
        json.dump([asdict(c) for c in self.chunks], open(os.path.join(self.path, "chunks.json"), "w"))
        np.save(os.path.join(self.path, "vectors.npy"), self.vectors)

    def add(self, source: str, texts: list[str]) -> int:
        if not texts:
            return 0
        vecs = embed(texts)
        with self.lock:
            self.remove(source, save=False)
            new = [Chunk(f"{source}#{i}", source, t) for i, t in enumerate(texts)]
            self.chunks.extend(new)
            self.vectors = vecs if self.vectors.size == 0 else np.vstack([self.vectors, vecs])
            self._save()
        return len(new)

    def remove(self, source: str, save: bool = True):
        keep = [i for i, c in enumerate(self.chunks) if c.source != source]
        self.chunks = [self.chunks[i] for i in keep]
        self.vectors = self.vectors[keep] if self.vectors.size else self.vectors
        if save:
            self._save()

    def sources(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.chunks:
            out[c.source] = out.get(c.source, 0) + 1
        return out

    def search(self, query: str, k: int = TOP_K) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        scores = self.vectors @ embed([query])[0]
        idx = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in idx]


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers ONLY from the provided context. "
    "Cite sources inline like [1], [2]. If the answer is not in the context, say you don't know."
)


def answer(question: str, hits: list[tuple[Chunk, float]], history: list[dict] | None = None) -> str:
    if not hits:
        return "I don't have any documents yet. Upload a PDF or text file to get started."
    if MOCK:
        q = set(_tokens(question))
        best = []
        for n, (c, _) in enumerate(hits[:2], 1):
            sents = re.split(r"(?<=[.!?])\s+", c.text)
            s = max(sents, key=lambda s: len(q & set(_tokens(s))))
            best.append(f"{s.strip()} [{n}]")
        return "(mock mode – no OPENAI_API_KEY set) Most relevant passages:\n\n" + "\n\n".join(best)
    context = "\n\n".join(f"[{i}] ({c.source})\n{c.text}" for i, (c, _) in enumerate(hits, 1))
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, *(history or [])[-6:],
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]
    resp = _client.chat.completions.create(model=CHAT_MODEL, messages=msgs, temperature=0.2)
    return resp.choices[0].message.content
