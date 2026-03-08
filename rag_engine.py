"""
RAG Engine for the Facts-Only MF Assistant.

Embeddings: uses sentence-transformers (all-MiniLM-L6-v2) when available,
with automatic fallback to TF-IDF cosine similarity for offline environments.
Vector search: FAISS (inner-product / cosine) or numpy dot-product fallback.
Generation: OpenAI GPT-3.5-turbo.
Corpus: official AMC, SEBI, and AMFI text documents.
"""

import logging
import os
import re
from pathlib import Path

import numpy as np
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Try to import optional heavy dependencies
try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

# Keywords that suggest the user wants investment advice (not facts)
ADVICE_PATTERNS = [
    r'\bshould i\b',
    r'\bshould i buy\b',
    r'\bshould i sell\b',
    r'\bshould i invest\b',
    r'\bshould i redeem\b',
    r'\brecommend\b',
    r'\bbest fund\b',
    r'\bwhich fund (is|to|should)\b',
    r'\bportfolio\b',
    r'\bbetter (to|than|fund)\b',
    r'\bworth investing\b',
    r'\bgood investment\b',
    r'\bis it good\b',
    r'\bwill.*give.*returns?\b',
    r'\bwill.*perform\b',
    r'\bpredict\b',
    r'\bforecast\b',
    r'\bfuture returns?\b',
    r'\bexpected returns?\b',
    r'\bgood returns?\b',
]

EDUCATIONAL_REFUSAL_LINK = (
    "https://www.amfiindia.com/investor-corner/knowledge-center/how-to-invest.html"
)

SYSTEM_PROMPT = """You are a facts-only mutual fund information assistant.

STRICT RULES:
1. Answer ONLY factual questions using the provided context from official sources.
2. Keep every answer to 3 sentences or fewer.
3. Do NOT provide investment advice, recommendations, performance predictions, or opinions.
4. If the context does not contain enough information to answer, respond with exactly:
   "I don't have that specific information in my sources. Please visit the official HDFC Fund website at https://www.hdfcfund.com or AMFI at https://www.amfiindia.com for the latest data."
5. Do not compute or compare past/future returns.
6. Do not make up facts or hallucinate data.
7. Always be concise and factual."""


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


class RAGEngine:
    """Retrieval-Augmented Generation engine for MF FAQ.

    Embedding strategy (chosen at init time):
    - If sentence-transformers loads successfully → dense embeddings + FAISS
    - Otherwise → TF-IDF cosine similarity (no internet required)
    """

    def __init__(self, corpus_dir: str = "data/corpus"):
        self.corpus_dir = Path(corpus_dir)
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        self.chunks: list[str] = []
        self.metadata: list[dict] = []

        # Dense embedding state
        self._embedding_model = None
        self._faiss_index = None
        self._dense_embeddings: np.ndarray | None = None

        # Sparse (TF-IDF) fallback state
        self._tfidf: TfidfVectorizer | None = None
        self._tfidf_matrix = None
        self._use_dense = False

        self._load_corpus()
        self._build_index()

    # ── Corpus Loading ─────────────────────────────────────────────────────────

    def _load_corpus(self) -> None:
        """Read all .txt files from the corpus directory into chunks + metadata."""
        txt_files = sorted(self.corpus_dir.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(
                f"No .txt files found in corpus directory: {self.corpus_dir}"
            )

        for filepath in txt_files:
            content = filepath.read_text(encoding="utf-8")
            lines = content.strip().splitlines()

            source_url = ""
            source_name = ""
            body_start = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("SOURCE_URL:"):
                    source_url = stripped.replace("SOURCE_URL:", "").strip()
                    body_start = i + 1
                elif stripped.startswith("SOURCE_NAME:"):
                    source_name = stripped.replace("SOURCE_NAME:", "").strip()
                    body_start = i + 1
                elif stripped.startswith("LAST_ACCESSED:"):
                    body_start = i + 1

            body = "\n".join(lines[body_start:]).strip()
            for chunk in _chunk_text(body):
                self.chunks.append(chunk)
                self.metadata.append(
                    {
                        "source_url": source_url,
                        "source_name": source_name,
                        "file": filepath.name,
                    }
                )

        if not self.chunks:
            raise ValueError("Corpus loaded but no text chunks were produced.")

    # ── Index Building ─────────────────────────────────────────────────────────

    def _build_index(self) -> None:
        """Try dense embeddings; fall back to TF-IDF if unavailable."""
        if _ST_AVAILABLE:
            try:
                self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                embeddings = self._embedding_model.encode(
                    self.chunks,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).astype(np.float32)
                if _FAISS_AVAILABLE:
                    dim = embeddings.shape[1]
                    self._faiss_index = faiss.IndexFlatIP(dim)
                    self._faiss_index.add(embeddings)
                else:
                    self._dense_embeddings = embeddings
                self._use_dense = True
                logger.info("Using dense embeddings (sentence-transformers).")
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Dense embeddings unavailable (%s); falling back to TF-IDF.", exc)

        # Fallback: TF-IDF
        self._tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20_000,
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._tfidf.fit_transform(self.chunks)
        logger.info("Using TF-IDF retrieval.")

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def _retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, dict]]:
        """Return the top_k most relevant (chunk, metadata) pairs for a query."""
        if self._use_dense:
            return self._retrieve_dense(query, top_k)
        return self._retrieve_tfidf(query, top_k)

    def _retrieve_dense(self, query: str, top_k: int) -> list[tuple[str, dict]]:
        q_emb = self._embedding_model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)
        if self._faiss_index is not None:
            _, indices = self._faiss_index.search(q_emb, top_k)
            idx_list = indices[0].tolist()
        else:
            sims = (self._dense_embeddings @ q_emb.T).squeeze()
            idx_list = np.argsort(-sims)[:top_k].tolist()
        return [(self.chunks[i], self.metadata[i]) for i in idx_list if 0 <= i < len(self.chunks)]

    def _retrieve_tfidf(self, query: str, top_k: int) -> list[tuple[str, dict]]:
        q_vec = self._tfidf.transform([query])
        sims = cosine_similarity(q_vec, self._tfidf_matrix).flatten()
        top_indices = np.argsort(-sims)[:top_k]
        return [(self.chunks[i], self.metadata[i]) for i in top_indices]

    # ── Advice Detection ───────────────────────────────────────────────────────

    def _is_advice_query(self, query: str) -> bool:
        """Return True if the query asks for investment advice or predictions."""
        lower = query.lower()
        return any(re.search(pattern, lower) for pattern in ADVICE_PATTERNS)

    # ── Generation ─────────────────────────────────────────────────────────────
        """Call OpenAI to generate a factual answer from the provided context."""
        user_message = (
            f"Context from official AMC/SEBI/AMFI sources:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer factually in 3 sentences or fewer:"
        )
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=250,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    def answer(self, query: str) -> dict:
        """
        Process a user query and return a structured result dict.

        Returns a dict with:
          - type: "answer" | "refusal" | "error"
          - answer: (answer type only) the factual answer string
          - source_url: (answer type only) primary source URL
          - source_name: (answer type only) primary source name
          - message: (refusal/error type) user-facing message
          - link: (refusal type) educational link
        """
        query = query.strip()
        if not query:
            return {"type": "error", "message": "Please enter a question."}

        if self._is_advice_query(query):
            return {
                "type": "refusal",
                "message": (
                    "I'm a facts-only assistant and cannot provide investment advice, "
                    "recommendations, or return predictions. For personalized guidance, "
                    "please consult a SEBI-registered investment advisor."
                ),
                "link": EDUCATIONAL_REFUSAL_LINK,
            }

        retrieved = self._retrieve(query)
        if not retrieved:
            return {
                "type": "error",
                "message": "Could not retrieve relevant information. Please try rephrasing your question.",
            }

        context = "\n\n---\n\n".join(chunk for chunk, _ in retrieved)
        primary_meta = retrieved[0][1]

        if not self.openai_client.api_key:
            return {
                "type": "error",
                "message": (
                    "OpenAI API key is not configured. "
                    "Please set the OPENAI_API_KEY environment variable and restart the app."
                ),
            }

        try:
            answer_text = self._generate_answer(query, context)
        except Exception as exc:  # noqa: BLE001
            return {
                "type": "error",
                "message": f"Failed to generate answer: {exc}",
            }

        return {
            "type": "answer",
            "answer": answer_text,
            "source_url": primary_meta["source_url"],
            "source_name": primary_meta["source_name"],
        }
