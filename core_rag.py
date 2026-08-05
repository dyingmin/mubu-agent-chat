from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Sequence


import chromadb
import numpy as np
import requests
from rank_bm25 import BM25Okapi
from http import HTTPStatus

try:
    import dashscope  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    dashscope = None
from config import (
    AppPaths,
    BM25_WEIGHT,
    CHUNK_MAX_CHARS,
    CHUNK_OVERLAP_CHARS,
    DOCX_ENABLE_SEMANTIC_CHUNKING,
    EMBEDDING_API_KEY,
    EMBEDDING_MODEL,
    HYBRID_TOP_K,
    MIN_CONTEXT_SCORE,
    QWEN_API_KEY,
    QWEN_BASE_URL,
    QWEN_MODEL,
    QWEN_TIMEOUT,
    RERANK_TOP_K,
    SEMANTIC_CHUNK_BREAKPOINT_PERCENTILE,
    SEMANTIC_CHUNK_BUFFER_SIZE,
    SEMANTIC_CHUNK_MAX_CHARS,
    SEMANTIC_CHUNK_MIN_CHARS,
    SYSTEM_PROMPT,
    TOP_K_BM25,
    TOP_K_CONTEXT,
    TOP_K_VECTOR,
    VECTOR_WEIGHT,
    ensure_dirs,
)

try:
    from langchain_community.document_loaders import Docx2txtLoader  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    Docx2txtLoader = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    RecursiveCharacterTextSplitter = None

try:
    from pypdf import PdfReader  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    PdfReader = None

TITLE_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;\.])\s*")
WORD_RE = re.compile(r"[\w\u4e00-\u9fff]+")
STOPWORDS = {"的", "了", "和", "是", "在", "请问", "什么", "如何", "怎么", "为什么", "吗", "呢", "吧"}
TERM_TITLE_RE = re.compile(r"^(?:\d+[\.)、]|[（(]?\d+[）)]?\s*)?(.{2,40}?)(?:[:：\-—]\s*(.*))?$")
DOCX_CHUNK_MAX_CHARS = 700
DOCX_CHUNK_OVERLAP = 100
DOCX_RECURSIVE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""]
TERM_MERGE_MIN_CHARS = 40


@dataclass
class SearchHit:
    chunk: DocumentChunk
    bm25_score: float = 0.0
    vector_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0


@dataclass
class DocumentChunk:
    chunk_id: str
    source_file: str
    title_path: str
    content: str
    heading_level: int
    order: int

    def metadata(self) -> dict:
        return asdict(self)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = WORD_RE.findall(text)
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def stable_hash_token(token: str) -> int:
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)


def split_by_title(text: str, source_file: str) -> List[DocumentChunk]:
    lines = normalize_text(text).split("\n")
    chunks: List[DocumentChunk] = []
    title_stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    order = 0

    def current_title_path() -> str:
        return " > ".join(title for _, title in title_stack) if title_stack else Path(source_file).stem

    def flush() -> None:
        nonlocal buffer, order
        content = normalize_text("\n".join(buffer))
        if content:
            order += 1
            chunks.append(
                DocumentChunk(
                    chunk_id=make_chunk_id(source_file, order, 0, content),
                    source_file=source_file,
                    title_path=current_title_path(),
                    content=content,
                    heading_level=title_stack[-1][0] if title_stack else 0,
                    order=order,
                )
            )
        buffer = []

    for line in lines:
        match = TITLE_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            while title_stack and title_stack[-1][0] >= level:
                title_stack.pop()
            title_stack.append((level, title))
        else:
            buffer.append(line)
    flush()
    return chunks


def make_chunk_id(source_file: str, order: int, part: int = 0, content: str = "") -> str:
    """生成稳定且尽量避免碰撞的 chunk ID。"""
    normalized_source = str(Path(source_file).resolve())
    payload = f"{normalized_source}:{order}:{part}:{normalize_text(content)}"
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
    return f"chunk_{digest[:16]}"


def further_chunk_long_text(content: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> List[str]:
    content = normalize_text(content)
    if len(content) <= max_chars:
        return [content]
    sentences = SENTENCE_SPLIT_RE.split(content)
    parts: List[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        candidate = (current + " " + sentence).strip() if current else sentence.strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                parts.append(current)
            current = sentence.strip()
    if current:
        parts.append(current)
    if not parts:
        step = max(1, max_chars - overlap)
        parts = [content[i : i + max_chars] for i in range(0, len(content), step)]
    return parts


def read_text_file(file_path: Path) -> str:
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return file_path.read_text(encoding="utf-8", errors="ignore")


def read_docx_file(file_path: Path) -> str:
    if Docx2txtLoader is None:
        raise ImportError("缺少 langchain_community 依赖，无法解析 Word 文档。")
    loader = Docx2txtLoader(str(file_path))
    documents = loader.load()
    text = "\n".join(doc.page_content for doc in documents if getattr(doc, "page_content", "").strip())
    return normalize_text(text)


def recursive_split_text(text: str, chunk_size: int = DOCX_CHUNK_MAX_CHARS, chunk_overlap: int = DOCX_CHUNK_OVERLAP) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=DOCX_RECURSIVE_SEPARATORS,
        )
        return splitter.split_text(text)
    return further_chunk_long_text(text, max_chars=chunk_size, overlap=chunk_overlap)


def chunk_title_from_text(text: str, fallback: str, index: int) -> str:
    cleaned = normalize_text(text)
    if not cleaned:
        return f"{fallback}-{index}"
    first_line = cleaned.split("\n", 1)[0].strip()
    first_line = re.sub(r"^[\W_\d]+", "", first_line)
    if len(first_line) > 24:
        first_line = first_line[:24].rstrip()
    return first_line or f"{fallback}-{index}"


def split_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(text) if sentence and sentence.strip()]
    return sentences


def cosine_distance(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    similarity = float(np.dot(a, b) / denom)
    similarity = max(min(similarity, 1.0), -1.0)
    return 1.0 - similarity


def semantic_split_text(
    text: str,
    embedder: "EmbeddingClient",
    max_chars: int = SEMANTIC_CHUNK_MAX_CHARS,
    min_chars: int = SEMANTIC_CHUNK_MIN_CHARS,
    buffer_size: int = SEMANTIC_CHUNK_BUFFER_SIZE,
    breakpoint_percentile_threshold: float = SEMANTIC_CHUNK_BREAKPOINT_PERCENTILE,
) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return [text]
    if len(sentences) <= buffer_size:
        return [text]

    window_texts: List[str] = []
    half_window = max(1, buffer_size // 2)
    for idx in range(len(sentences)):
        start = max(0, idx - half_window)
        end = min(len(sentences), idx + half_window + 1)
        window_texts.append(" ".join(sentences[start:end]))

    embeddings = embedder.embed(window_texts)
    if len(embeddings) < 2:
        return [text]

    distances = [cosine_distance(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]
    threshold = float(np.percentile(distances, breakpoint_percentile_threshold)) if distances else 1.0

    chunks: List[str] = []
    current: List[str] = []
    for idx, sentence in enumerate(sentences):
        current.append(sentence)
        current_text = normalize_text(" ".join(current))
        should_break = idx < len(distances) and distances[idx] >= threshold and len(current_text) >= min_chars
        if should_break:
            chunks.append(current_text)
            current = []
    if current:
        chunks.append(normalize_text(" ".join(current)))

    final_chunks: List[str] = []
    for chunk in chunks:
        chunk = normalize_text(chunk)
        if not chunk:
            continue
        if len(chunk) > max_chars:
            final_chunks.extend(recursive_split_text(chunk, chunk_size=max_chars, chunk_overlap=DOCX_CHUNK_OVERLAP))
            continue
        if final_chunks and len(chunk) < min_chars:
            merged = normalize_text(f"{final_chunks[-1]}\n{chunk}")
            if len(merged) <= max_chars:
                final_chunks[-1] = merged
                continue
        final_chunks.append(chunk)

    return [chunk for chunk in final_chunks if chunk.strip()]


def semantic_split_docx_chunks(file_path: Path, embedder: "EmbeddingClient") -> List[DocumentChunk]:
    text = read_docx_file(file_path)
    pieces = recursive_split_text(text, chunk_size=max(DOCX_CHUNK_MAX_CHARS, SEMANTIC_CHUNK_MAX_CHARS), chunk_overlap=DOCX_CHUNK_OVERLAP)
    semantic_pieces: List[str] = []
    if DOCX_ENABLE_SEMANTIC_CHUNKING:
        for piece in pieces:
            semantic_pieces.extend(semantic_split_text(piece, embedder=embedder))
    else:
        semantic_pieces = pieces

    chunks: list[DocumentChunk] = []
    for order, piece in enumerate(semantic_pieces, start=1):
        piece = normalize_text(piece)
        if not piece:
            continue
        title = chunk_title_from_text(piece, file_path.stem, order)
        chunks.append(
            DocumentChunk(
                chunk_id=make_chunk_id(str(file_path), order, 0, f"{title}\n{piece}"),
                source_file=str(file_path),
                title_path=title,
                content=piece,
                heading_level=1,
                order=order,
            )
        )
    return chunks


def read_pdf_file(file_path: Path) -> str:
    if PdfReader is None:
        raise ImportError("缺少 pypdf 依赖，无法解析 PDF 文档。")
    reader = PdfReader(str(file_path))
    pages: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append(text)
    return "\n".join(pages)


def load_file_content(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        return read_text_file(file_path)
    if suffix == ".docx":
        return read_docx_file(file_path)
    if suffix == ".pdf":
        return read_pdf_file(file_path)
    return ""


def load_documents(doc_dir: Path, embedder: "EmbeddingClient | None" = None) -> List[DocumentChunk]:
    documents: List[DocumentChunk] = []
    supported_suffixes = {".md", ".markdown", ".txt", ".docx", ".pdf"}
    embedder = embedder or EmbeddingClient()
    for file_path in sorted(doc_dir.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in supported_suffixes:
            continue
        if file_path.suffix.lower() == ".docx":
            base_chunks = semantic_split_docx_chunks(file_path, embedder=embedder)
            documents.extend(base_chunks)
            continue

        text = load_file_content(file_path)
        if not text.strip():
            continue
        base_chunks = split_by_title(text, str(file_path))
        if not base_chunks:
            base_chunks = [
                DocumentChunk(
                    chunk_id=make_chunk_id(str(file_path), 1, 0, text),
                    source_file=str(file_path),
                    title_path=file_path.stem,
                    content=normalize_text(text),
                    heading_level=0,
                    order=1,
                )
            ]
        for chunk in base_chunks:
            sub_chunks = further_chunk_long_text(chunk.content)
            if len(sub_chunks) == 1:
                documents.append(chunk)
            else:
                for idx, sub in enumerate(sub_chunks, start=1):
                    documents.append(
                        DocumentChunk(
                            chunk_id=make_chunk_id(chunk.source_file, chunk.order, idx, sub),
                            source_file=chunk.source_file,
                            title_path=chunk.title_path,
                            content=sub,
                            heading_level=chunk.heading_level,
                            order=chunk.order * 100 + idx,
                        )
                    )
    return documents


class EmbeddingClient:
    """使用 DashScope SDK 的 embedding 客户端。

    优先调用 qwen3.7-text-embedding；如果未安装 dashscope 或未配置 API Key，则回退到本地词袋向量。
    """

    MAX_BATCH_SIZE = 20

    def __init__(self) -> None:
        self.model = EMBEDDING_MODEL
        self.api_key = EMBEDDING_API_KEY or QWEN_API_KEY
        self.dimension = int(os.getenv("EMBEDDING_DIM", "1024"))
        self._sdk_available = dashscope is not None and bool(self.api_key)
        if self._sdk_available:
            dashscope.api_key = self.api_key

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if self._sdk_available:
            return self._embed_via_dashscope(texts)
        return self._fallback_embed(texts)

    def _embed_via_dashscope(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        for start in range(0, len(texts), self.MAX_BATCH_SIZE):
            batch = list(texts[start:start + self.MAX_BATCH_SIZE])
            batch_embeddings = self._call_dashscope_batch(batch)
            if len(batch_embeddings) != len(batch):
                raise ValueError(
                    f"DashScope embedding 返回数量不匹配，输入 {len(batch)} 条，返回 {len(batch_embeddings)} 条"
                )
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def _call_dashscope_batch(self, texts: Sequence[str]) -> List[List[float]]:
        if len(texts) == 1:
            input_payload: str | List[str] = texts[0]
        else:
            input_payload = list(texts)

        resp = dashscope.TextEmbedding.call(  # type: ignore[union-attr]
            model=self.model,
            input=input_payload,
        )
        if getattr(resp, "status_code", None) not in (None, 200, HTTPStatus.OK):
            raise RuntimeError(f"DashScope embedding 请求失败: {resp}")

        output = getattr(resp, "output", None)
        embeddings = self._extract_embeddings(output)
        if not embeddings:
            raise ValueError(f"DashScope embedding 返回结果为空: {resp}")
        return embeddings

    def _extract_embeddings(self, output: object) -> List[List[float]]:
        if not isinstance(output, dict):
            return []

        for key in ("embeddings", "data", "output"):
            embeddings = self._normalize_embedding_container(output.get(key))
            if embeddings:
                return embeddings
        return []

    def _normalize_embedding_container(self, candidate: object) -> List[List[float]]:
        if candidate is None:
            return []

        if isinstance(candidate, dict):
            candidate = [candidate]

        if isinstance(candidate, list):
            normalized: List[List[float]] = []
            for item in candidate:
                vector = self._normalize_embedding_item(item)
                if vector is None:
                    continue
                normalized.append(vector)
            return normalized

        vector = self._normalize_embedding_item(candidate)
        return [vector] if vector is not None else []

    def _normalize_embedding_item(self, item: object) -> List[float] | None:
        if isinstance(item, dict):
            for key in ("embedding", "vector", "embeddings"):
                value = item.get(key)
                if isinstance(value, list):
                    try:
                        return [float(x) for x in value]
                    except (TypeError, ValueError):
                        return None
            return None

        if isinstance(item, (list, tuple)):
            try:
                return [float(x) for x in item]
            except (TypeError, ValueError):
                return None

        return None

    def _fallback_embed(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            for token in tokenize(text):
                idx = stable_hash_token(token) % self.dimension
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec.tolist())
        return vectors


class RAGIndex:
    def __init__(self, paths: AppPaths | None = None) -> None:
        ensure_dirs()
        self.paths = paths or AppPaths()
        self.embedder = EmbeddingClient()
        self.chroma_client = chromadb.PersistentClient(path=str(self.paths.chroma_dir))
        self.collection = self.chroma_client.get_or_create_collection(name="rag_knowledge_base")
        self.chunks: List[DocumentChunk] = []
        self.bm25: BM25Okapi | None = None
        self._load_cache()

    def _cache_path(self) -> Path:
        return self.paths.cache_dir / "chunks.json"

    def _load_cache(self) -> None:
        cache_path = self._cache_path()
        if cache_path.exists():
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            self.chunks = [DocumentChunk(**item) for item in raw]
            self._build_bm25()
            self._restore_collection_from_cache()

    def _restore_collection_from_cache(self) -> None:
        if not self.chunks:
            return
        try:
            existing = self.collection.get(include=[])
            if existing.get("ids"):
                return
        except Exception:
            pass
        embeddings = self.embedder.embed([f"{c.title_path}\n{c.content}" for c in self.chunks])
        self.collection.add(
            ids=[c.chunk_id for c in self.chunks],
            documents=[c.content for c in self.chunks],
            embeddings=embeddings,
            metadatas=[c.metadata() for c in self.chunks],
        )

    def _reset_collection(self) -> None:
        try:
            self.chroma_client.delete_collection(name=self.collection.name)
        except Exception:
            pass
        self.collection = self.chroma_client.get_or_create_collection(name="rag_knowledge_base")

    def _save_cache(self) -> None:
        self._cache_path().write_text(
            json.dumps([asdict(chunk) for chunk in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _build_bm25(self) -> None:
        tokenized = [tokenize(f"{c.title_path} {c.content}") for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def build(self, doc_dir: Path | None = None) -> int:
        doc_dir = doc_dir or self.paths.doc_dir
        docs = load_documents(doc_dir, embedder=self.embedder)
        self.chunks = docs
        self._build_bm25()
        self._reset_collection()
        if docs:
            embeddings = self.embedder.embed([f"{c.title_path}\n{c.content}" for c in docs])
            self.collection.add(
                ids=[c.chunk_id for c in docs],
                documents=[c.content for c in docs],
                embeddings=embeddings,
                metadatas=[c.metadata() for c in docs],
            )
        self._save_cache()
        return len(docs)

    def _bm25_search(self, query: str, top_k: int = TOP_K_BM25) -> list[tuple[float, DocumentChunk]]:
        if not self.bm25 or not self.chunks:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(scores, self.chunks), key=lambda x: x[0], reverse=True)
        return [(float(score), chunk) for score, chunk in ranked[:top_k] if score > 0]

    def _vector_search(self, query: str, top_k: int = TOP_K_VECTOR) -> list[tuple[float, DocumentChunk]]:
        if not self.chunks:
            return []
        embedding = self.embedder.embed([query])[0]
        result = self.collection.query(query_embeddings=[embedding], n_results=top_k)
        scores: list[tuple[float, DocumentChunk]] = []
        ids = result.get("ids", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distance_list = result.get("distances", [[]])[0]
        for idx, _chunk_id in enumerate(ids):
            if idx >= len(metadatas):
                continue
            meta = metadatas[idx]
            if not meta:
                continue
            chunk = DocumentChunk(**meta)
            dist = float(distance_list[idx]) if idx < len(distance_list) else 0.0
            score = 1.0 / (1.0 + max(dist, 0.0))
            scores.append((score, chunk))
        return scores

    def _build_hybrid_candidates(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchHit]:
        bm25_hits = self._bm25_search(query, top_k=TOP_K_BM25)
        vec_hits = self._vector_search(query, top_k=TOP_K_VECTOR)

        merged: dict[str, SearchHit] = {}
        for score, chunk in bm25_hits:
            hit = merged.setdefault(chunk.chunk_id, SearchHit(chunk=chunk))
            hit.bm25_score = score
        for score, chunk in vec_hits:
            hit = merged.setdefault(chunk.chunk_id, SearchHit(chunk=chunk))
            hit.vector_score = score

        results: list[SearchHit] = []
        for hit in merged.values():
            hit.hybrid_score = BM25_WEIGHT * hit.bm25_score + VECTOR_WEIGHT * hit.vector_score
            if hit.hybrid_score >= MIN_CONTEXT_SCORE:
                results.append(hit)
        results.sort(key=lambda x: x.hybrid_score, reverse=True)
        return results[:top_k]

    def _rerank_hits(self, query: str, hits: list[SearchHit], top_k: int = RERANK_TOP_K) -> list[SearchHit]:
        if not hits:
            return []

        query_tokens = set(tokenize(query))
        if not query_tokens:
            query_tokens = set(tokenize(query.lower()))

        for hit in hits:
            title_tokens = set(tokenize(hit.chunk.title_path))
            content_tokens = set(tokenize(hit.chunk.content))
            overlap = len(query_tokens & (title_tokens | content_tokens))
            title_overlap = len(query_tokens & title_tokens)
            length_bonus = min(len(hit.chunk.content) / 2000.0, 1.0) * 0.05
            source_bonus = 0.12 if Path(hit.chunk.source_file).suffix.lower() in {'.md', '.markdown'} else 0.0
            hit.rerank_score = (
                hit.hybrid_score * 0.55
                + overlap * 0.20
                + title_overlap * 0.16
                + length_bonus
                + source_bonus
            )

        hits.sort(key=lambda x: x.rerank_score, reverse=True)
        return hits[:top_k]

    def hybrid_search(self, query: str, top_k: int = TOP_K_CONTEXT) -> list[dict]:
        candidates = self._build_hybrid_candidates(query, top_k=HYBRID_TOP_K)
        reranked = self._rerank_hits(query, candidates, top_k=top_k)
        return [
            {
                "chunk_id": hit.chunk.chunk_id,
                "title_path": hit.chunk.title_path,
                "source_file": hit.chunk.source_file,
                "content": hit.chunk.content,
                "score": float(hit.rerank_score),
                "hybrid_score": float(hit.hybrid_score),
                "bm25_score": float(hit.bm25_score),
                "vector_score": float(hit.vector_score),
            }
            for hit in reranked
        ]

    def answer(self, question: str) -> dict:
        contexts = self.hybrid_search(question)
        if not contexts:
            return {
                "answer": "知识库中未找到足够相关的内容，请尝试更具体的问题，或先执行知识库重建。",
                "source": "knowledge_base_miss",
                "contexts": [],
            }
        prompt_context = self._build_context(contexts)
        answer = call_qwen_api(question, prompt_context)
        return {"answer": answer, "source": "knowledge_base", "contexts": contexts}

    def _build_context(self, contexts: list[dict]) -> str:
        blocks = []
        for i, ctx in enumerate(contexts, start=1):
            blocks.append(f"[文档{i}]\n标题路径：{ctx['title_path']}\n来源：{ctx['source_file']}\n内容：{ctx['content']}")
        return "\n\n".join(blocks)


def call_qwen_api(question: str, context: str) -> str:
    if not QWEN_API_KEY:
        return (
            "当前未配置 QWEN_API_KEY，以下为基于检索上下文的离线回答。\n\n"
            f"问题：{question}\n\n"
            f"检索到的上下文：\n{context[:2000]}\n\n"
            "请在环境变量中配置千问 API Key 后启用真实大模型回答。"
        )

    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"问题：{question}\n\n可用知识库上下文：\n{context}\n\n请基于上下文回答。",
            },
        ],
        "temperature": 0.2,
    }
    base_url = QWEN_BASE_URL.rstrip('/')
    if base_url.endswith('/chat/completions'):
        url = base_url
    else:
        url = f"{base_url}/chat/completions"
    resp = requests.post(url, json=payload, headers=headers, timeout=QWEN_TIMEOUT)
    if not resp.ok:
        raise RuntimeError(f"QWEN API 请求失败: {resp.status_code} {resp.text[:1000]}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]
