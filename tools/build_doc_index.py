#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


def _is_probably_text_file(path: Path) -> bool:
    # Heuristic: skip common binary/media formats
    lower = path.name.lower()
    binary_exts = {
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff",
        ".mp4", ".mov", ".avi", ".mkv", ".mp3", ".wav",
        ".zip", ".gz", ".tar", ".7z", ".rar",
        ".exe", ".dll", ".dylib", ".so", ".class", ".jar",
        ".pyc",
    }
    if path.suffix.lower() in binary_exts:
        return False
    return True


def _read_text(path: Path) -> str:
    # Try utf-8, fall back to "replace"
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: List[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n".join(parts)


def load_document_text(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    if not _is_probably_text_file(path) and path.suffix.lower() != ".pdf":
        return None

    if path.suffix.lower() == ".pdf":
        try:
            text = _read_pdf_text(path)
        except Exception:
            return None
    else:
        text = _read_text(path)

    # Basic cleanup
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return None
    return text


def iter_input_files(
    inputs: Sequence[str],
    input_dirs: Sequence[str],
    exts: Sequence[str],
    ignore_dirs: Sequence[str],
) -> Iterator[Path]:
    seen: set[Path] = set()

    def add(p: Path) -> None:
        p = p.resolve()
        if p in seen:
            return
        seen.add(p)
        yield p

    # Direct files
    for s in inputs:
        p = Path(s).expanduser()
        if p.exists() and p.is_file():
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                yield rp

    # Directories (recursive)
    ignore_set = {d.strip() for d in ignore_dirs if d.strip()}
    normalized_exts = [e if e.startswith(".") else f".{e}" for e in exts]
    normalized_exts = [e.lower() for e in normalized_exts]

    for d in input_dirs:
        root = Path(d).expanduser()
        if not root.exists() or not root.is_dir():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if any(part in ignore_set for part in p.parts):
                continue
            if normalized_exts:
                if p.suffix.lower() not in normalized_exts:
                    continue
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            yield rp


def chunk_text(
    text: str,
    chunk_chars: int,
    overlap_chars: int,
) -> List[str]:
    """
    Простая и стабильная разбивка: сначала по пустым строкам (абзацы),
    затем если абзац слишком большой — режем скользящим окном по символам.
    """
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be > 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")
    if overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must be < chunk_chars")

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush_buf() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        combined = "\n\n".join(buf).strip()
        if combined:
            chunks.extend(_chunk_by_window(combined, chunk_chars, overlap_chars))
        buf = []
        buf_len = 0

    for para in paragraphs:
        if len(para) > chunk_chars:
            flush_buf()
            chunks.extend(_chunk_by_window(para, chunk_chars, overlap_chars))
            continue
        # try to pack paragraphs into ~chunk size
        extra = len(para) + (2 if buf else 0)
        if buf_len + extra <= chunk_chars:
            buf.append(para)
            buf_len += extra
        else:
            flush_buf()
            buf.append(para)
            buf_len = len(para)

    flush_buf()
    return chunks


def _chunk_by_window(text: str, chunk_chars: int, overlap_chars: int) -> List[str]:
    if len(text) <= chunk_chars:
        return [text.strip()]
    out: List[str] = []
    step = chunk_chars - overlap_chars
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        start += step
    return out


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    source_path: str
    chunk_index: int
    text: str


def stable_chunk_id(source_path: str, chunk_index: int, text: str) -> str:
    h = hashlib.sha256()
    h.update(source_path.encode("utf-8", errors="ignore"))
    h.update(b"\n")
    h.update(str(chunk_index).encode("utf-8"))
    h.update(b"\n")
    h.update(text.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:24]


def build_records(paths: Iterable[Path], chunk_chars: int, overlap_chars: int) -> List[ChunkRecord]:
    records: List[ChunkRecord] = []
    for p in paths:
        text = load_document_text(p)
        if not text:
            continue
        chunks = chunk_text(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
        source = str(p)
        for i, ch in enumerate(chunks):
            rid = stable_chunk_id(source, i, ch)
            records.append(ChunkRecord(id=rid, source_path=source, chunk_index=i, text=ch))
    return records


def embed_texts(texts: Sequence[str], model_name: str, batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    # normalize_embeddings=True => cosine similarity is just dot-product
    emb = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32, copy=False)


def save_dir_npy_jsonl(out_dir: Path, meta: dict, records: Sequence[ChunkRecord], embeddings: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", embeddings)
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(
                json.dumps(
                    {
                        "id": r.id,
                        "source_path": r.source_path,
                        "chunk_index": r.chunk_index,
                        "text": r.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def save_json(path: Path, meta: dict, records: Sequence[ChunkRecord], embeddings: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    items = []
    for r, v in zip(records, embeddings):
        items.append(
            {
                "id": r.id,
                "source_path": r.source_path,
                "chunk_index": r.chunk_index,
                "text": r.text,
                "embedding": v.astype(float).tolist(),
            }
        )
    payload = {"meta": meta, "items": items}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def save_sqlite(path: Path, meta: dict, records: Sequence[ChunkRecord], embeddings: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    con = sqlite3.connect(str(path))
    try:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")

        con.execute(
            """
            CREATE TABLE meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            """
        )
        con.execute(
            """
            CREATE TABLE chunks (
              id TEXT PRIMARY KEY,
              source_path TEXT NOT NULL,
              chunk_index INTEGER NOT NULL,
              text TEXT NOT NULL
            );
            """
        )
        con.execute(
            """
            CREATE TABLE embeddings (
              id TEXT PRIMARY KEY,
              dim INTEGER NOT NULL,
              vector BLOB NOT NULL,
              FOREIGN KEY(id) REFERENCES chunks(id)
            );
            """
        )
        con.execute("CREATE INDEX idx_chunks_source ON chunks(source_path);")

        for k, v in meta.items():
            con.execute("INSERT INTO meta(key, value) VALUES (?, ?)", (str(k), json.dumps(v, ensure_ascii=False)))

        con.executemany(
            "INSERT INTO chunks(id, source_path, chunk_index, text) VALUES (?, ?, ?, ?)",
            [(r.id, r.source_path, int(r.chunk_index), r.text) for r in records],
        )

        dim = int(embeddings.shape[1])
        con.executemany(
            "INSERT INTO embeddings(id, dim, vector) VALUES (?, ?, ?)",
            [
                (r.id, dim, sqlite3.Binary(embeddings[i].astype(np.float32, copy=False).tobytes()))
                for i, r in enumerate(records)
            ],
        )
        con.commit()
    finally:
        con.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build local document index: chunks + embeddings")
    ap.add_argument("--input", action="append", default=[], help="Input file path (repeatable)")
    ap.add_argument("--input-dir", action="append", default=[], help="Input directory path (repeatable, recursive)")
    ap.add_argument("--ext", action="append", default=[], help="File extension filter for --input-dir, e.g. md or .py (repeatable)")
    ap.add_argument("--ignore-dir", action="append", default=[".git", "node_modules", "build", "__pycache__", ".venv", "new-env"], help="Directory names to ignore (repeatable)")
    ap.add_argument("--chunk-chars", type=int, default=1200, help="Chunk size in characters")
    ap.add_argument("--overlap-chars", type=int, default=150, help="Overlap size in characters")
    ap.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help="SentenceTransformers model name")
    ap.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    ap.add_argument(
        "--store",
        choices=["dir", "json", "sqlite"],
        default="dir",
        help="Store format: dir (meta.json + chunks.jsonl + embeddings.npy), json (single file), sqlite (single .sqlite file)",
    )
    ap.add_argument("--out", default="doc_index", help="Output path (dir for --store dir, file path for --store json/sqlite)")
    args = ap.parse_args()

    out_path = Path(args.out).expanduser().resolve()
    if args.store == "dir":
        out_path.mkdir(parents=True, exist_ok=True)

    files = list(
        iter_input_files(
            inputs=args.input,
            input_dirs=args.input_dir,
            exts=args.ext,
            ignore_dirs=args.ignore_dir,
        )
    )
    if not files:
        raise SystemExit("No input files found. Use --input and/or --input-dir")

    records = build_records(files, chunk_chars=args.chunk_chars, overlap_chars=args.overlap_chars)
    if not records:
        raise SystemExit("No text extracted from inputs (empty or unsupported files).")

    embeddings = embed_texts([r.text for r in records], model_name=args.model, batch_size=args.batch_size)

    meta = {
        "model": args.model,
        "chunk_chars": args.chunk_chars,
        "overlap_chars": args.overlap_chars,
        "count_chunks": len(records),
        "dim": int(embeddings.shape[1]),
        "sources": sorted({r.source_path for r in records}),
    }

    if args.store == "dir":
        save_dir_npy_jsonl(out_path, meta, records, embeddings)
        print(f"✓ Saved index to directory: {out_path}")
        print(f"  - {out_path / 'meta.json'}")
        print(f"  - {out_path / 'chunks.jsonl'}")
        print(f"  - {out_path / 'embeddings.npy'}  shape={embeddings.shape} dtype={embeddings.dtype}")
    elif args.store == "json":
        if out_path.suffix.lower() != ".json":
            out_path = out_path.with_suffix(".json")
        save_json(out_path, meta, records, embeddings)
        print(f"✓ Saved index to JSON: {out_path}")
        print(f"  - chunks: {meta['count_chunks']}  dim: {meta['dim']}")
    else:
        if out_path.suffix.lower() not in {'.sqlite', '.db'}:
            out_path = out_path.with_suffix(".sqlite")
        save_sqlite(out_path, meta, records, embeddings)
        print(f"✓ Saved index to SQLite: {out_path}")
        print(f"  - chunks: {meta['count_chunks']}  dim: {meta['dim']}")


if __name__ == "__main__":
    main()


