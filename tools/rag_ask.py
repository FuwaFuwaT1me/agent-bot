#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Chunk:
    id: str
    source_path: str
    chunk_index: int
    text: str


def load_index_from_dir(index_dir: Path) -> tuple[dict, List[Chunk], np.ndarray]:
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    chunks: List[Chunk] = []
    with (index_dir / "chunks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            chunks.append(
                Chunk(
                    id=obj["id"],
                    source_path=obj["source_path"],
                    chunk_index=int(obj["chunk_index"]),
                    text=obj["text"],
                )
            )
    emb = np.load(index_dir / "embeddings.npy").astype(np.float32, copy=False)
    if emb.shape[0] != len(chunks):
        raise ValueError(f"Embeddings rows ({emb.shape[0]}) != chunks ({len(chunks)})")
    return meta, chunks, emb


def load_index_from_json(path: Path) -> tuple[dict, List[Chunk], np.ndarray]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    meta = obj["meta"]
    items = obj["items"]
    chunks: List[Chunk] = []
    vectors: List[np.ndarray] = []
    for it in items:
        chunks.append(
            Chunk(
                id=it["id"],
                source_path=it["source_path"],
                chunk_index=int(it["chunk_index"]),
                text=it["text"],
            )
        )
        vectors.append(np.asarray(it["embedding"], dtype=np.float32))
    emb = np.stack(vectors, axis=0) if vectors else np.zeros((0, int(meta.get("dim", 0))), dtype=np.float32)
    return meta, chunks, emb


def load_index_from_sqlite(path: Path) -> tuple[dict, List[Chunk], np.ndarray]:
    con = sqlite3.connect(str(path))
    try:
        meta_rows = con.execute("SELECT key, value FROM meta").fetchall()
        meta = {k: json.loads(v) for (k, v) in meta_rows}

        rows = con.execute(
            """
            SELECT c.id, c.source_path, c.chunk_index, c.text, e.dim, e.vector
            FROM chunks c
            JOIN embeddings e ON e.id = c.id
            ORDER BY c.rowid ASC
            """
        ).fetchall()

        chunks: List[Chunk] = []
        vectors: List[np.ndarray] = []
        for (cid, src, cidx, text, dim, blob) in rows:
            chunks.append(Chunk(id=cid, source_path=src, chunk_index=int(cidx), text=text))
            v = np.frombuffer(blob, dtype=np.float32)
            if int(dim) != v.shape[0]:
                raise ValueError(f"Bad vector dim for id={cid}: expected {dim}, got {v.shape[0]}")
            vectors.append(v)

        emb = np.stack(vectors, axis=0) if vectors else np.zeros((0, int(meta.get("dim", 0))), dtype=np.float32)
        return meta, chunks, emb
    finally:
        con.close()


def load_index(index_path: Path) -> tuple[dict, List[Chunk], np.ndarray]:
    if index_path.is_dir():
        return load_index_from_dir(index_path)
    suffix = index_path.suffix.lower()
    if suffix == ".json":
        return load_index_from_json(index_path)
    if suffix in {".sqlite", ".db"}:
        return load_index_from_sqlite(index_path)
    raise ValueError("Unsupported index format. Use a directory, .json or .sqlite/.db file.")


def embed_query(query: str, model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    v = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    return v.astype(np.float32, copy=False)[0]


def topk_cosine(query_vec: np.ndarray, emb: np.ndarray, k: int) -> np.ndarray:
    scores = emb @ query_vec
    if scores.shape[0] == 0:
        return np.array([], dtype=np.int64)
    if k >= scores.shape[0]:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx


def build_context(chunks: List[Chunk], emb: np.ndarray, q: np.ndarray, idxs: np.ndarray, max_chars: int) -> str:
    parts: List[str] = []
    used = 0
    for rank, i in enumerate(idxs.tolist(), 1):
        c = chunks[i]
        score = float(emb[i] @ q)
        block = f"[{rank}] score={score:.4f}\nsource={c.source_path} chunk={c.chunk_index}\n{c.text}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts).strip()


def try_answer_with_yandex(question: str, context: str) -> Optional[str]:
    folder_id = os.getenv("YANDEX_FOLDER_ID")
    auth = os.getenv("YANDEX_AUTH")
    if not folder_id or not auth:
        return None
    try:
        from yandex_cloud_ml_sdk import YCloudML
    except Exception:
        return None

    yandex = YCloudML(folder_id=folder_id, auth=auth)
    system = (
        "Ты помощник. Отвечай на вопрос, используя ТОЛЬКО контекст ниже. "
        "Если в контексте нет ответа — честно скажи, что в базе знаний этого нет."
    )
    prompt = f"КОНТЕКСТ:\n{context}\n\nВОПРОС:\n{question}"
    messages = [
        {"role": "system", "text": system},
        {"role": "user", "text": prompt},
    ]
    result = yandex.models.completions("yandexgpt").configure(temperature=0.2, max_tokens=800).run(messages)
    for alt in result:
        if hasattr(alt, "text"):
            return (alt.text or "").strip()
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG demo: retrieve top chunks and (optionally) ask LLM with context")
    ap.add_argument("--index", default="doc_index", help="Index path: directory, .json or .sqlite/.db")
    ap.add_argument("--question", required=True, help="User question")
    ap.add_argument("--top-k", type=int, default=5, help="How many chunks to retrieve")
    ap.add_argument("--max-context-chars", type=int, default=6000, help="Max context size (chars)")
    ap.add_argument("--no-llm", action="store_true", help="Only retrieve context, do not call an LLM")
    args = ap.parse_args()

    index_path = Path(args.index).expanduser().resolve()
    meta, chunks, emb = load_index(index_path)
    q = embed_query(args.question, model_name=meta["model"])
    idxs = topk_cosine(q, emb, k=args.top_k)
    context = build_context(chunks, emb, q, idxs, max_chars=args.max_context_chars)

    print("=== Retrieved context ===")
    print(context if context else "(empty)")
    print()

    if args.no_llm:
        return

    answer = try_answer_with_yandex(args.question, context)
    if answer is None:
        print("=== Answer ===")
        print("LLM not configured (set YANDEX_FOLDER_ID + YANDEX_AUTH and install yandex-cloud-ml-sdk), showing context only.")
        return

    print("=== Answer ===")
    print(answer)


if __name__ == "__main__":
    main()


