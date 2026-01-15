#!/usr/bin/env python3
"""
Простой Telegram-бот на базе YandexGPT и DeepSeek.
Отвечает на вопросы пользователей.
"""

import os
import time
import json
import httpx
import asyncio
import shlex
import io
from datetime import time as dt_time, datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv
import shutil
import platform
from openai import OpenAI
from yandex_cloud_ml_sdk import YCloudML
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from history_compressor import check_and_compress_history
from local_storage import get_combined_summary, clear_summaries, get_summary_count
from mobile_mcp import MobileMcpService, pick_tool_name, parse_kv_args, extract_images_from_mcp_result, extract_text_from_mcp_result, safe_call

# Загружаем переменные окружения
load_dotenv()

YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
YANDEX_AUTH = os.getenv("YANDEX_AUTH")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

if not YANDEX_FOLDER_ID or not YANDEX_AUTH or not TELEGRAM_BOT_TOKEN:
    raise ValueError("Установите YANDEX_FOLDER_ID, YANDEX_AUTH и TELEGRAM_BOT_TOKEN в .env файле!")

# MCP Server URLs (Kotlin MCP Servers)
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8080/mcp")  # Calendar MCP
MCP_EVENTS_URL = os.getenv("MCP_EVENTS_URL", "http://localhost:8081/mcp")  # KudaGo Events MCP

# Mobile MCP (stdio, Node)
MOBILE_MCP_COMMAND = os.getenv("MOBILE_MCP_COMMAND", "npx -y @mobilenext/mobile-mcp@latest")

# Daily reminder settings
DAILY_REMINDER_HOUR = int(os.getenv("DAILY_REMINDER_HOUR", "9"))  # Default: 9:00 AM
DAILY_REMINDER_MINUTE = int(os.getenv("DAILY_REMINDER_MINUTE", "0"))  # Default: 0 minutes
DAILY_REMINDER_CHAT_ID = os.getenv("DAILY_REMINDER_CHAT_ID")  # Your Telegram chat ID
DAILY_REMINDER_TIMEZONE_OFFSET = int(os.getenv("DAILY_REMINDER_TIMEZONE_OFFSET", "3"))  # Default: Moscow (UTC+3)

# === RAG / KB (single local document) ===
# Source text file you edit:
#   kb/bookechi_docs.md
# Index is created by tools/build_doc_index.py into SQLite:
#   doc_index/bookechi.sqlite
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KB_SOURCE_PATH = os.getenv("KB_SOURCE_PATH", os.path.join(REPO_ROOT, "kb", "bookechi_docs.md"))
KB_INDEX_PATH = os.getenv("KB_INDEX_PATH", os.path.join(REPO_ROOT, "doc_index", "bookechi.sqlite"))
KB_TOP_K = int(os.getenv("KB_TOP_K", "5"))
KB_MAX_CONTEXT_CHARS = int(os.getenv("KB_MAX_CONTEXT_CHARS", "6000"))
KB_MIN_SCORE_DEFAULT = float(os.getenv("KB_MIN_SCORE", "0.0"))

# === Bookechi Git Repository ===
BOOKECHI_REPO_PATH = os.getenv("BOOKECHI_REPO_PATH", os.path.join(REPO_ROOT, "bookechi_repo"))

# === Support Ticket System ===
TICKETS_FILE = os.path.join(os.path.dirname(__file__), "tickets.json")

# Per-user support mode toggle
user_support_mode: Dict[int, bool] = {}  # True = support mode enabled
user_current_ticket: Dict[int, Optional[int]] = {}  # user_id -> ticket_id

def load_tickets() -> dict:
    """Load tickets from JSON file."""
    if os.path.exists(TICKETS_FILE):
        with open(TICKETS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"tickets": [], "next_id": 1, "categories": ["bug", "question", "feature", "other"], 
            "priorities": ["low", "medium", "high", "critical"], "statuses": ["open", "in_progress", "waiting", "closed"]}

def save_tickets(data: dict) -> None:
    """Save tickets to JSON file."""
    with open(TICKETS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_ticket_by_id(ticket_id: int) -> Optional[dict]:
    """Get ticket by ID."""
    data = load_tickets()
    for ticket in data["tickets"]:
        if ticket["id"] == ticket_id:
            return ticket
    return None

def get_user_tickets(user_id: int) -> List[dict]:
    """Get all tickets for a user."""
    data = load_tickets()
    return [t for t in data["tickets"] if t["user_id"] == user_id]

def get_open_tickets() -> List[dict]:
    """Get all open tickets."""
    data = load_tickets()
    return [t for t in data["tickets"] if t["status"] in ("open", "in_progress", "waiting")]

def create_ticket(user_id: int, user_name: str, subject: str, description: str, category: str = "question") -> dict:
    """Create a new ticket."""
    data = load_tickets()
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    ticket = {
        "id": data["next_id"],
        "user_id": user_id,
        "user_name": user_name,
        "status": "open",
        "priority": "medium",
        "category": category,
        "subject": subject,
        "description": description,
        "created_at": now,
        "updated_at": now,
        "messages": [
            {"from": "user", "text": description, "timestamp": now}
        ]
    }
    data["tickets"].append(ticket)
    data["next_id"] += 1
    save_tickets(data)
    return ticket

def add_message_to_ticket(ticket_id: int, from_who: str, text: str) -> bool:
    """Add message to ticket."""
    data = load_tickets()
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    for ticket in data["tickets"]:
        if ticket["id"] == ticket_id:
            ticket["messages"].append({"from": from_who, "text": text, "timestamp": now})
            ticket["updated_at"] = now
            save_tickets(data)
            return True
    return False

def update_ticket_status(ticket_id: int, status: str) -> bool:
    """Update ticket status."""
    data = load_tickets()
    for ticket in data["tickets"]:
        if ticket["id"] == ticket_id:
            ticket["status"] = status
            ticket["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            save_tickets(data)
            return True
    return False

def format_ticket_summary(ticket: dict) -> str:
    """Format ticket for display."""
    status_emoji = {"open": "🔴", "in_progress": "🟡", "waiting": "🟠", "closed": "🟢"}.get(ticket["status"], "⚪")
    priority_emoji = {"low": "🔵", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(ticket["priority"], "⚪")
    return (
        f"{status_emoji} *Тикет #{ticket['id']}*\n"
        f"📋 {ticket['subject']}\n"
        f"👤 {ticket['user_name']}\n"
        f"📁 {ticket['category']} | {priority_emoji} {ticket['priority']}\n"
        f"📅 {ticket['created_at'][:10]}"
    )

def format_ticket_full(ticket: dict) -> str:
    """Format ticket with messages."""
    status_emoji = {"open": "🔴", "in_progress": "🟡", "waiting": "🟠", "closed": "🟢"}.get(ticket["status"], "⚪")
    
    header = (
        f"{status_emoji} *Тикет #{ticket['id']}* — {ticket['status']}\n"
        f"📋 *{ticket['subject']}*\n"
        f"👤 {ticket['user_name']} (ID: {ticket['user_id']})\n"
        f"📁 Категория: {ticket['category']} | Приоритет: {ticket['priority']}\n"
        f"📅 Создан: {ticket['created_at'][:16].replace('T', ' ')}\n"
        f"─────────────────────\n"
    )
    
    messages_text = ""
    for msg in ticket["messages"][-10:]:  # Last 10 messages
        sender = "👤 Пользователь" if msg["from"] == "user" else "🤖 Поддержка"
        time_str = msg["timestamp"][11:16] if "T" in msg["timestamp"] else ""
        messages_text += f"{sender} ({time_str}):\n{msg['text']}\n\n"
    
    return header + messages_text

# Per-user toggle: whether to inject KB context into regular chat messages.
user_kb_enabled: Dict[int, bool] = {}
user_kb_min_score: Dict[int, float] = {}  # per-user threshold for cosine similarity (0..1)

# Smart RAG routing: agent decides automatically whether to use RAG
user_kb_auto_enabled: Dict[int, bool] = {}  # auto-routing mode (default: False)

# Description of knowledge base for the router (what topics it contains)
KB_DESCRIPTION = """База знаний содержит информацию о проекте Bookechi:
- Обзор проекта: Android-приложение для отслеживания чтения книг
- Технологический стек: Kotlin, Jetpack Compose, Room, Koin, Navigation Compose
- Архитектура MVI: BaseViewModel, Actions, State, Model
- Модели данных: Book, ReadingSession, ReadingStatus, ActivityIntensity
- UI компоненты: чарты активности (как на GitHub), обложки книг, текстовые поля
- База данных Room: BookDao, ReadingSessionDao, AppDatabase
- Dependency Injection: Koin модули, DatabaseModule
- Навигация: NavigationHost, экраны (BookList, AddBook, BookDetails, Stats, Goals, Settings)
- Правила стиля кода Kotlin/Compose
- Исходный код всех классов проекта
"""

# In-memory cache of KB index for fast retrieval.
_kb_cache: Dict[str, Any] = {"path": None, "mtime": None, "meta": None, "chunks": None, "emb": None, "model": None}
_kb_st_model = None  # SentenceTransformer


@dataclass(frozen=True)
class KbChunk:
    id: str
    source_path: str
    chunk_index: int
    text: str


def _kb_load_sqlite(index_path: str) -> tuple[dict, List[KbChunk], "np.ndarray"]:
    import sqlite3
    import numpy as np

    con = sqlite3.connect(index_path)
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
        chunks: List[KbChunk] = []
        vectors: List[np.ndarray] = []
        for (cid, src, cidx, text, dim, blob) in rows:
            chunks.append(KbChunk(id=cid, source_path=src, chunk_index=int(cidx), text=text))
            v = np.frombuffer(blob, dtype=np.float32)
            if int(dim) != v.shape[0]:
                raise ValueError(f"Bad vector dim for id={cid}: expected {dim}, got {v.shape[0]}")
            vectors.append(v)
        emb = np.stack(vectors, axis=0) if vectors else np.zeros((0, int(meta.get("dim", 0))), dtype=np.float32)
        return meta, chunks, emb.astype(np.float32, copy=False)
    finally:
        con.close()


def kb_load_index() -> tuple[dict, List[KbChunk], "np.ndarray", str]:
    import numpy as np

    index_path = os.path.abspath(KB_INDEX_PATH)
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"KB index not found: {index_path}. Run /kb_reindex first (it builds from {KB_SOURCE_PATH})."
        )
    mtime = os.path.getmtime(index_path)
    if _kb_cache["path"] == index_path and _kb_cache["mtime"] == mtime:
        return _kb_cache["meta"], _kb_cache["chunks"], _kb_cache["emb"], _kb_cache["model"]

    meta, chunks, emb = _kb_load_sqlite(index_path)
    model_name = meta.get("model") or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    _kb_cache.update({"path": index_path, "mtime": mtime, "meta": meta, "chunks": chunks, "emb": emb, "model": model_name})
    return meta, chunks, emb, model_name


def kb_embed_query(query: str, model_name: str) -> "np.ndarray":
    import numpy as np

    global _kb_st_model
    if _kb_st_model is None or getattr(_kb_st_model, "model_card", None) is None:
        from sentence_transformers import SentenceTransformer
        _kb_st_model = SentenceTransformer(model_name)
    v = _kb_st_model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    return v.astype(np.float32, copy=False)[0]


def kb_topk_cosine(query_vec: "np.ndarray", emb: "np.ndarray", k: int) -> "np.ndarray":
    import numpy as np

    if emb.shape[0] == 0:
        return np.array([], dtype=np.int64)
    scores = emb @ query_vec
    if k >= scores.shape[0]:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx


def kb_build_context(chunks: List[KbChunk], emb: "np.ndarray", q: "np.ndarray", idxs: "np.ndarray", max_chars: int) -> str:
    used = 0
    out: List[str] = []
    for rank, i in enumerate(idxs.tolist(), 1):
        c = chunks[i]
        score = float(emb[i] @ q)
        block = f"[{rank}] score={score:.4f}\n{c.text}\n"
        if used + len(block) > max_chars:
            break
        out.append(block)
        used += len(block)
    return "\n".join(out).strip()


def kb_retrieve(
    question: str,
    top_k: int = None,
    min_score: Optional[float] = None,
    allow_fallback: bool = True,
) -> tuple[str, dict]:
    """Returns (context_text, debug_meta). Applies optional min_score threshold."""
    top_k = top_k or KB_TOP_K
    if min_score is None:
        min_score = KB_MIN_SCORE_DEFAULT
    meta, chunks, emb, model_name = kb_load_index()
    q = kb_embed_query(question, model_name=model_name)
    idxs = kb_topk_cosine(q, emb, k=top_k)
    scores = [(int(i), float(emb[int(i)] @ q)) for i in idxs.tolist()]
    kept = [(i, s) for (i, s) in scores if s >= float(min_score)]

    # Fallback: if threshold too strict, keep best-1 (so RAG doesn't become empty unless retrieval is empty).
    fallback_used = False
    if allow_fallback and (not kept) and scores:
        kept = [scores[0]]
        fallback_used = True

    kept_idxs = [i for (i, _s) in kept]
    import numpy as np
    kept_np = np.asarray(kept_idxs, dtype=np.int64) if kept_idxs else np.array([], dtype=np.int64)

    ctx = kb_build_context(chunks, emb, q, kept_np, max_chars=KB_MAX_CONTEXT_CHARS)
    best_score = kept[0][1] if kept else (scores[0][1] if scores else None)
    dbg = {
        "index": os.path.abspath(KB_INDEX_PATH),
        "model": model_name,
        "chunks": len(chunks),
        "top_k": top_k,
        "min_score": float(min_score),
        "kept": len(kept),
        "retrieved": len(scores),
        "best_score": best_score,
        "fallback_used": fallback_used,
        "top_scores": [round(s, 4) for (_i, s) in scores[: min(5, len(scores))]],
        "kept_scores": [round(s, 4) for (_i, s) in kept[: min(5, len(kept))]],
        "context_chars": len(ctx) if ctx else 0,
    }
    return ctx, dbg


async def cmd_kb_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    enabled = user_kb_enabled.get(user_id, False)
    auto_enabled = user_kb_auto_enabled.get(user_id, False)
    min_score = user_kb_min_score.get(user_id, KB_MIN_SCORE_DEFAULT)
    src = os.path.abspath(KB_SOURCE_PATH)
    idx = os.path.abspath(KB_INDEX_PATH)
    idx_exists = os.path.exists(idx)
    
    # Определяем текущий режим
    if auto_enabled:
        mode = "🧭 Умный (агент решает сам)"
    elif enabled:
        mode = "✅ Принудительный (всегда RAG)"
    else:
        mode = "❌ Выключен"
    
    msg = (
        "📚 KB (RAG) status\n\n"
        f"Режим: {mode}\n"
        f"Min score (threshold): {min_score:.3f}\n"
        f"KB source: {src}\n"
        f"KB index:  {idx}\n"
        f"Index exists: {'YES' if idx_exists else 'NO'}\n\n"
        "Команды:\n"
        "/kb_auto_on — 🧭 умный режим (агент решает сам)\n"
        "/kb_auto_off — выключить умный режим\n"
        "/kb_on — принудительный RAG (всегда)\n"
        "/kb_off — выключить RAG\n"
        "/kb_reindex — пересобрать индекс\n"
        "/kb_ask <вопрос> — спросить по базе\n"
        "/kb_compare <вопрос> — сравнить ответы: без RAG vs с RAG\n"
        "/kb_threshold [0.0-1.0] — порог релевантности\n"
        "/kb_debug <вопрос> — диагностика retrieval"
    )
    await update.message.reply_text(msg)


async def cmd_kb_threshold(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show or set per-user KB min_score threshold (cosine similarity)."""
    user_id = update.effective_user.id
    if not context.args:
        cur = user_kb_min_score.get(user_id, KB_MIN_SCORE_DEFAULT)
        await update.message.reply_text(
            f"📏 KB threshold\n\nCurrent min_score: {cur:.3f}\n\n"
            "Использование:\n"
            "/kb_threshold 0.28"
        )
        return

    raw = context.args[0].strip().replace(",", ".")
    try:
        v = float(raw)
        if not (0.0 <= v <= 1.0):
            raise ValueError("out of range")
    except Exception:
        await update.message.reply_text("❌ Неверное значение. Укажи число от 0.0 до 1.0, например: /kb_threshold 0.28")
        return

    user_kb_min_score[user_id] = v
    await update.message.reply_text(f"✅ KB min_score установлен: {v:.3f}")


async def cmd_kb_debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show retrieved chunks with scores (for debugging retrieval/threshold)."""
    if not context.args:
        await update.message.reply_text("Использование: /kb_debug <вопрос>")
        return

    question = " ".join(context.args).strip()
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        # Use min_score=0.0 to show raw retrieval distribution.
        ctx, dbg = kb_retrieve(question, min_score=0.0, allow_fallback=False)
    except Exception as e:
        await update.message.reply_text(f"❌ KB error: {e}\n\nСначала сделай /kb_reindex")
        return

    preview = (ctx or "").strip()
    if len(preview) > 1600:
        preview = preview[:1550].rstrip() + "\n…(truncated)…"

    msg = (
        "🧩 KB debug (retrieval)\n\n"
        f"Вопрос:\n{question}\n\n"
        f"top_k={dbg.get('top_k')} retrieved={dbg.get('retrieved')} best={dbg.get('best_score')} ctx_chars={dbg.get('context_chars')}\n"
        f"top_scores={dbg.get('top_scores')} kept_scores={dbg.get('kept_scores')} fallback_used={dbg.get('fallback_used')}\n\n"
        "Контекст (превью):\n"
        f"{preview or '(empty)'}"
    )
    await update.message.reply_text(msg)


async def cmd_kb_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_kb_enabled[user_id] = True
    await update.message.reply_text("✅ KB (RAG) включён: буду подмешивать контекст из knowledge_base в обычные ответы.")


async def cmd_kb_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_kb_enabled[user_id] = False
    user_kb_auto_enabled[user_id] = False  # Выключаем и умный режим
    await update.message.reply_text("❌ KB (RAG) выключен: обычные ответы без базы знаний.")


async def cmd_kb_auto_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Включить умный режим RAG — агент сам решает, нужна ли база знаний."""
    user_id = update.effective_user.id
    user_kb_auto_enabled[user_id] = True
    user_kb_enabled[user_id] = False  # Выключаем принудительный режим
    await update.message.reply_text(
        "🧭 *Умный режим KB включён!*\n\n"
        "Теперь агент сам решает, обращаться ли к базе знаний:\n"
        "• Если вопрос о продуктах, ценах, контактах → RAG\n"
        "• Если вопрос общий или продолжение темы → без RAG\n\n"
        "В статистике будет видно решение роутера.",
        parse_mode="Markdown"
    )


async def cmd_kb_auto_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Выключить умный режим RAG."""
    user_id = update.effective_user.id
    user_kb_auto_enabled[user_id] = False
    await update.message.reply_text(
        "❌ Умный режим KB выключен.\n\n"
        "Используй /kb_on для принудительного RAG или /kb_auto_on для умного режима."
    )


# === GitHub Code Review Functions ===
import re
from urllib.parse import urlparse

def parse_github_url(url: str) -> Optional[dict]:
    """
    Parse GitHub URL to extract owner, repo, type (commit/pull), and id.
    Supports:
    - https://github.com/owner/repo/commit/sha
    - https://github.com/owner/repo/pull/123
    - https://github.com/owner/repo/pull/123/commits/sha
    """
    patterns = [
        # Commit URL
        r'github\.com/([^/]+)/([^/]+)/commit/([a-f0-9]+)',
        # PR URL
        r'github\.com/([^/]+)/([^/]+)/pull/(\d+)(?:/.*)?',
    ]
    
    for i, pattern in enumerate(patterns):
        match = re.search(pattern, url)
        if match:
            groups = match.groups()
            if i == 0:  # Commit
                return {
                    "owner": groups[0],
                    "repo": groups[1],
                    "type": "commit",
                    "id": groups[2]
                }
            else:  # PR
                return {
                    "owner": groups[0],
                    "repo": groups[1],
                    "type": "pull",
                    "id": groups[2]
                }
    return None


async def fetch_github_diff(url: str) -> tuple[Optional[str], Optional[str]]:
    """
    Fetch diff from GitHub URL.
    Returns (diff_content, error_message).
    """
    parsed = parse_github_url(url)
    if not parsed:
        return None, "Не удалось распознать URL. Поддерживаются ссылки на коммиты и PR."
    
    # Construct diff URL
    if parsed["type"] == "commit":
        diff_url = f"https://github.com/{parsed['owner']}/{parsed['repo']}/commit/{parsed['id']}.diff"
    else:  # pull
        diff_url = f"https://github.com/{parsed['owner']}/{parsed['repo']}/pull/{parsed['id']}.diff"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(diff_url, follow_redirects=True)
            if response.status_code == 200:
                return response.text, None
            else:
                return None, f"GitHub вернул статус {response.status_code}"
    except Exception as e:
        return None, f"Ошибка при получении diff: {e}"


async def fetch_github_commit_info(url: str) -> tuple[Optional[dict], Optional[str]]:
    """
    Fetch commit/PR info from GitHub API.
    Returns (info_dict, error_message).
    """
    parsed = parse_github_url(url)
    if not parsed:
        return None, "Не удалось распознать URL"
    
    # GitHub API URLs
    if parsed["type"] == "commit":
        api_url = f"https://api.github.com/repos/{parsed['owner']}/{parsed['repo']}/commits/{parsed['id']}"
    else:
        api_url = f"https://api.github.com/repos/{parsed['owner']}/{parsed['repo']}/pulls/{parsed['id']}"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {"Accept": "application/vnd.github.v3+json"}
            response = await client.get(api_url, headers=headers, follow_redirects=True)
            if response.status_code == 200:
                return response.json(), None
            else:
                return None, f"GitHub API вернул статус {response.status_code}"
    except Exception as e:
        return None, f"Ошибка API: {e}"


def extract_changed_files_from_diff(diff: str) -> List[str]:
    """Extract list of changed files from diff."""
    files = []
    for line in diff.split("\n"):
        if line.startswith("diff --git"):
            # Extract file path from "diff --git a/path b/path"
            parts = line.split(" ")
            if len(parts) >= 4:
                file_path = parts[2][2:]  # Remove "a/" prefix
                files.append(file_path)
    return files


def truncate_diff(diff: str, max_chars: int = 8000) -> str:
    """Truncate diff if too long, keeping file headers."""
    if len(diff) <= max_chars:
        return diff
    
    lines = diff.split("\n")
    result = []
    current_len = 0
    
    for line in lines:
        if current_len + len(line) + 1 > max_chars - 200:
            result.append("\n... (diff обрезан, слишком длинный) ...")
            break
        result.append(line)
        current_len += len(line) + 1
    
    return "\n".join(result)


async def cmd_review(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /review <github_url> - Analyze commit or PR and provide code review.
    Uses RAG for project context and GitHub API for diff.
    """
    if not context.args:
        await update.message.reply_text(
            "📝 *Ревью кода*\n\n"
            "Использование: `/review <github_url>`\n\n"
            "Примеры:\n"
            "• `/review https://github.com/owner/repo/commit/sha`\n"
            "• `/review https://github.com/owner/repo/pull/123`\n\n"
            "Бот получит diff, проанализирует изменения с учётом контекста проекта "
            "и выдаст ревью с замечаниями.",
            parse_mode="Markdown"
        )
        return
    
    url = context.args[0].strip()
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    # Parse URL
    parsed = parse_github_url(url)
    if not parsed:
        await update.message.reply_text(
            "❌ Не удалось распознать URL.\n\n"
            "Поддерживаются:\n"
            "• `https://github.com/owner/repo/commit/sha`\n"
            "• `https://github.com/owner/repo/pull/123`",
            parse_mode="Markdown"
        )
        return
    
    # Send status message
    status_msg = await update.message.reply_text(
        f"🔍 Получаю diff для {parsed['type']} `{parsed['id'][:8] if parsed['type'] == 'commit' else '#' + parsed['id']}`...",
        parse_mode="Markdown"
    )
    
    # Fetch diff
    diff, error = await fetch_github_diff(url)
    if error:
        await status_msg.edit_text(f"❌ {error}")
        return
    
    if not diff or len(diff.strip()) == 0:
        await status_msg.edit_text("❌ Diff пустой или не найден")
        return
    
    # Extract changed files
    changed_files = extract_changed_files_from_diff(diff)
    
    await status_msg.edit_text(
        f"📄 Получен diff ({len(diff)} символов, {len(changed_files)} файлов)\n"
        f"🔎 Ищу контекст в базе знаний..."
    )
    
    # Get RAG context based on changed files and diff content
    rag_context = ""
    try:
        # Build query from file names and first part of diff
        query_parts = []
        for f in changed_files[:5]:
            # Extract class/file name
            name = os.path.basename(f).replace(".kt", "").replace(".java", "")
            query_parts.append(name)
        
        query = " ".join(query_parts) if query_parts else "Kotlin Android code review"
        
        min_score = user_kb_min_score.get(update.effective_user.id, KB_MIN_SCORE_DEFAULT)
        rag_context, dbg = kb_retrieve(query, top_k=5, min_score=min_score)
    except Exception as e:
        rag_context = ""
    
    await status_msg.edit_text(
        f"📄 Diff: {len(diff)} символов, {len(changed_files)} файлов\n"
        f"📚 Контекст: {len(rag_context)} символов\n"
        f"🤖 Генерирую ревью..."
    )
    
    # Truncate diff if too long
    diff_for_review = truncate_diff(diff, max_chars=10000)
    
    # Build review prompt
    system_prompt = """Ты опытный код-ревьюер для Android-проекта на Kotlin с Jetpack Compose.

Твоя задача — проанализировать diff и дать конструктивное ревью:

1. **Общая оценка** — кратко опиши, что делает этот коммит/PR
2. **Позитивные моменты** — что сделано хорошо
3. **Замечания и предложения** — конкретные проблемы с указанием файла и строки:
   - Потенциальные баги
   - Нарушения архитектуры (MVI pattern)
   - Проблемы производительности
   - Нарушения стиля кода
   - Отсутствие обработки ошибок
   - Проблемы с именованием
4. **Рекомендации** — общие советы по улучшению

Отвечай на русском языке. Будь конкретным и конструктивным.
Если замечаний нет — так и скажи. Не выдумывай проблемы."""

    # Add project context
    context_info = ""
    if rag_context:
        context_info = f"""
КОНТЕКСТ ПРОЕКТА (из документации и кода):
{rag_context}

"""

    user_prompt = f"""{context_info}ИЗМЕНЁННЫЕ ФАЙЛЫ:
{chr(10).join('• ' + f for f in changed_files[:20])}

DIFF:
```diff
{diff_for_review}
```

Проведи ревью этого кода."""

    # Generate review
    try:
        model = get_model(update.effective_user.id)
        if model == "deepseek":
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            completion = hf_client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3",
                messages=messages,
                temperature=0.3,
                max_tokens=2000,
            )
            review = (completion.choices[0].message.content or "").strip()
        else:
            messages = [
                {"role": "system", "text": system_prompt},
                {"role": "user", "text": user_prompt}
            ]
            result = yandex_sdk.models.completions("yandexgpt").configure(
                temperature=0.3,
                max_tokens=2000,
            ).run(messages)
            review = ""
            for alt in result:
                if hasattr(alt, "text"):
                    review = (alt.text or "").strip()
                    break
        
        # Format response
        header = (
            f"📝 *Code Review*\n"
            f"🔗 [{parsed['owner']}/{parsed['repo']}]({url})\n"
            f"📦 {parsed['type'].upper()}: `{parsed['id'][:8] if parsed['type'] == 'commit' else '#' + parsed['id']}`\n"
            f"📄 Файлов изменено: {len(changed_files)}\n\n"
        )
        
        # Delete status message and send review
        await status_msg.delete()
        
        # Split long messages
        full_response = header + review
        if len(full_response) > 4000:
            # Send in parts
            await update.message.reply_text(header, parse_mode="Markdown", disable_web_page_preview=True)
            
            # Split review into chunks
            chunks = [review[i:i+3900] for i in range(0, len(review), 3900)]
            for chunk in chunks:
                await update.message.reply_text(chunk)
        else:
            await update.message.reply_text(full_response, parse_mode="Markdown", disable_web_page_preview=True)
        
    except Exception as e:
        await status_msg.edit_text(f"❌ Ошибка генерации ревью: {e}")


# === Support Mode Commands ===

async def cmd_support(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Toggle support mode or show status."""
    user_id = update.effective_user.id
    is_enabled = user_support_mode.get(user_id, False)
    current_ticket_id = user_current_ticket.get(user_id)
    current_ticket = get_ticket_by_id(current_ticket_id) if current_ticket_id else None
    
    ticket_info = ""
    if current_ticket:
        ticket_info = f"\n📋 Активный тикет: #{current_ticket['id']} — {current_ticket['subject']}"
    
    status = "✅ Включён" if is_enabled else "❌ Выключен"
    
    await update.message.reply_text(
        f"🎧 *Режим поддержки Bookechi*\n\n"
        f"Статус: {status}{ticket_info}\n\n"
        f"*Команды:*\n"
        f"`/support_on` — включить режим поддержки\n"
        f"`/support_off` — выключить\n"
        f"`/ticket_new <тема>` — создать тикет\n"
        f"`/ticket_list` — мои тикеты\n"
        f"`/ticket_view <id>` — просмотр тикета\n"
        f"`/ticket_select <id>` — выбрать тикет для контекста\n"
        f"`/ticket_close <id>` — закрыть тикет\n"
        f"`/tickets_all` — все открытые тикеты (админ)\n\n"
        f"В режиме поддержки бот отвечает на вопросы о Bookechi,\n"
        f"используя FAQ, документацию и контекст вашего тикета.",
        parse_mode="Markdown"
    )


async def cmd_support_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enable support mode."""
    user_id = update.effective_user.id
    user_support_mode[user_id] = True
    
    # Check if user has open tickets
    tickets = get_user_tickets(user_id)
    open_tickets = [t for t in tickets if t["status"] in ("open", "in_progress", "waiting")]
    
    ticket_msg = ""
    if open_tickets:
        ticket_msg = f"\n\n📋 У вас {len(open_tickets)} открытых тикетов.\nИспользуйте `/ticket_select <id>` чтобы выбрать тикет для контекста."
    else:
        ticket_msg = "\n\n💡 Создайте тикет командой `/ticket_new <тема>` для более точной помощи."
    
    await update.message.reply_text(
        f"🎧 *Режим поддержки включён!*\n\n"
        f"Теперь я буду отвечать на ваши вопросы о приложении Bookechi.\n"
        f"Используется FAQ, документация и контекст вашего тикета.{ticket_msg}",
        parse_mode="Markdown"
    )


async def cmd_support_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Disable support mode."""
    user_id = update.effective_user.id
    user_support_mode[user_id] = False
    user_current_ticket[user_id] = None
    
    await update.message.reply_text("❌ Режим поддержки выключен. Теперь бот работает в обычном режиме.")


async def cmd_ticket_new(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Create a new support ticket."""
    if not context.args:
        await update.message.reply_text(
            "📝 *Создание тикета*\n\n"
            "Использование: `/ticket_new <тема вашего вопроса>`\n\n"
            "Пример: `/ticket_new Приложение вылетает при добавлении книги`",
            parse_mode="Markdown"
        )
        return
    
    user_id = update.effective_user.id
    user_name = update.effective_user.full_name or f"User {user_id}"
    subject = " ".join(context.args).strip()
    
    # Determine category from subject
    subject_lower = subject.lower()
    if any(word in subject_lower for word in ["вылетает", "ошибка", "баг", "не работает", "crash"]):
        category = "bug"
    elif any(word in subject_lower for word in ["как", "почему", "где", "?"]):
        category = "question"
    elif any(word in subject_lower for word in ["хочу", "добавьте", "предлагаю", "feature"]):
        category = "feature"
    else:
        category = "other"
    
    ticket = create_ticket(user_id, user_name, subject, subject, category)
    user_current_ticket[user_id] = ticket["id"]
    user_support_mode[user_id] = True
    
    await update.message.reply_text(
        f"✅ *Тикет #{ticket['id']} создан!*\n\n"
        f"📋 {subject}\n"
        f"📁 Категория: {category}\n\n"
        f"Режим поддержки включён автоматически.\n"
        f"Опишите вашу проблему подробнее, и я постараюсь помочь!",
        parse_mode="Markdown"
    )


async def cmd_ticket_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """List user's tickets."""
    user_id = update.effective_user.id
    tickets = get_user_tickets(user_id)
    
    if not tickets:
        await update.message.reply_text(
            "📭 У вас пока нет тикетов.\n\n"
            "Создайте новый: `/ticket_new <тема>`",
            parse_mode="Markdown"
        )
        return
    
    lines = ["📋 *Ваши тикеты:*\n"]
    for ticket in sorted(tickets, key=lambda t: t["created_at"], reverse=True):
        lines.append(format_ticket_summary(ticket))
        lines.append("")
    
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_ticket_view(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """View ticket details."""
    if not context.args:
        await update.message.reply_text("Использование: `/ticket_view <id>`", parse_mode="Markdown")
        return
    
    try:
        ticket_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("❌ ID тикета должен быть числом")
        return
    
    ticket = get_ticket_by_id(ticket_id)
    if not ticket:
        await update.message.reply_text(f"❌ Тикет #{ticket_id} не найден")
        return
    
    # Check access (user can see own tickets, or it's a public demo)
    user_id = update.effective_user.id
    if ticket["user_id"] != user_id:
        # Allow viewing for demo purposes
        pass
    
    await update.message.reply_text(format_ticket_full(ticket), parse_mode="Markdown")


async def cmd_ticket_select(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Select a ticket for context in support mode."""
    if not context.args:
        await update.message.reply_text("Использование: `/ticket_select <id>`", parse_mode="Markdown")
        return
    
    try:
        ticket_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("❌ ID тикета должен быть числом")
        return
    
    ticket = get_ticket_by_id(ticket_id)
    if not ticket:
        await update.message.reply_text(f"❌ Тикет #{ticket_id} не найден")
        return
    
    user_id = update.effective_user.id
    user_current_ticket[user_id] = ticket_id
    user_support_mode[user_id] = True
    
    await update.message.reply_text(
        f"✅ Тикет #{ticket_id} выбран для контекста!\n\n"
        f"📋 {ticket['subject']}\n\n"
        f"Теперь ваши вопросы будут учитывать историю этого тикета.",
        parse_mode="Markdown"
    )


async def cmd_ticket_close(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Close a ticket."""
    if not context.args:
        await update.message.reply_text("Использование: `/ticket_close <id>`", parse_mode="Markdown")
        return
    
    try:
        ticket_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("❌ ID тикета должен быть числом")
        return
    
    ticket = get_ticket_by_id(ticket_id)
    if not ticket:
        await update.message.reply_text(f"❌ Тикет #{ticket_id} не найден")
        return
    
    if update_ticket_status(ticket_id, "closed"):
        user_id = update.effective_user.id
        if user_current_ticket.get(user_id) == ticket_id:
            user_current_ticket[user_id] = None
        
        await update.message.reply_text(f"🟢 Тикет #{ticket_id} закрыт!")
    else:
        await update.message.reply_text("❌ Ошибка при закрытии тикета")


async def cmd_tickets_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """List all open tickets (admin view)."""
    tickets = get_open_tickets()
    
    if not tickets:
        await update.message.reply_text("✅ Нет открытых тикетов!")
        return
    
    lines = [f"📋 *Открытые тикеты ({len(tickets)}):*\n"]
    for ticket in sorted(tickets, key=lambda t: t["created_at"], reverse=True):
        lines.append(format_ticket_summary(ticket))
        lines.append("")
    
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def handle_support_message(update: Update, user_id: int, message_text: str) -> Optional[str]:
    """
    Handle message in support mode. Returns response text or None if not in support mode.
    Uses RAG (FAQ + docs) and ticket context.
    """
    if not user_support_mode.get(user_id, False):
        return None
    
    await update.message.chat.send_action("typing")
    
    # Get current ticket context
    ticket_context = ""
    current_ticket_id = user_current_ticket.get(user_id)
    if current_ticket_id:
        ticket = get_ticket_by_id(current_ticket_id)
        if ticket:
            ticket_context = (
                f"\n\nКОНТЕКСТ ТИКЕТА #{ticket['id']}:\n"
                f"Тема: {ticket['subject']}\n"
                f"Категория: {ticket['category']}\n"
                f"Статус: {ticket['status']}\n"
                f"История переписки:\n"
            )
            for msg in ticket["messages"][-5:]:
                sender = "Пользователь" if msg["from"] == "user" else "Поддержка"
                ticket_context += f"- {sender}: {msg['text'][:200]}\n"
            
            # Add user message to ticket
            add_message_to_ticket(current_ticket_id, "user", message_text)
    
    # Get RAG context from FAQ and docs
    # Use more chunks and no min_score filter for support mode
    rag_context = ""
    dbg = {}
    try:
        rag_context, dbg = kb_retrieve(message_text, top_k=8, min_score=0.0, allow_fallback=True)
    except Exception as e:
        rag_context = ""
        dbg = {"error": str(e)}
    
    # Build system prompt for support agent
    system_prompt = """Ты агент поддержки и эксперт по проекту Bookechi — Android-приложению для отслеживания чтения книг.

Bookechi — это:
- Android-приложение на Kotlin с Jetpack Compose
- Архитектура MVI (Model-View-Intent)
- Room для базы данных, Koin для DI
- UI с чартами активности (как GitHub contributions)

Структура проекта:
- ui/feature/ — экраны (book_list, add_book, book_details, reading_stats, settings)
- data/model/ — модели (Book, ReadingSession, ReadingStatus)
- data/local/ — Room DAO (BookDao, ReadingSessionDao)
- data/repository/ — репозитории
- mvi/ — базовые классы MVI (BaseViewModel, State, Action)
- base/ui/ — общие UI компоненты (чарты, обложки книг)

Твоя задача:
1. Отвечай на вопросы о коде и архитектуре проекта
2. ОБЯЗАТЕЛЬНО используй предоставленный КОНТЕКСТ — там фрагменты кода и документации
3. Приводи примеры кода из контекста
4. Если вопрос о пользовательских функциях — используй FAQ
5. Отвечай структурированно и по делу

Отвечай на русском языке."""

    prompt = f"""КОНТЕКСТ ИЗ FAQ И ДОКУМЕНТАЦИИ:
{rag_context if rag_context else "(нет релевантного контекста)"}
{ticket_context}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{message_text}

Дай полезный ответ:"""

    # Generate response
    try:
        model = get_model(user_id)
        if model == "deepseek":
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            completion = hf_client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3",
                messages=messages,
                temperature=0.4,
                max_tokens=1000,
            )
            response = (completion.choices[0].message.content or "").strip()
        else:
            messages = [
                {"role": "system", "text": system_prompt},
                {"role": "user", "text": prompt}
            ]
            result = yandex_sdk.models.completions("yandexgpt").configure(
                temperature=0.4,
                max_tokens=1000,
            ).run(messages)
            response = ""
            for alt in result:
                if hasattr(alt, "text"):
                    response = (alt.text or "").strip()
                    break
        
        # Add response to ticket if exists
        if current_ticket_id and response:
            add_message_to_ticket(current_ticket_id, "support", response)
        
        # Add footer with debug info
        footer = "\n\n─────────────────────\n🎧 _Режим поддержки Bookechi_"
        if current_ticket_id:
            footer += f" | 📋 _Тикет #{current_ticket_id}_"
        
        # Debug: show RAG stats
        if dbg:
            chunks = dbg.get('kept', dbg.get('retrieved', 0))
            best = dbg.get('best_score', 0)
            footer += f"\n📚 _RAG: {chunks} фрагментов, score={best:.2f}_" if best else ""
        
        return response + footer
    
    except Exception as e:
        return f"❌ Ошибка генерации ответа: {e}"


# === Git Integration Functions ===
def git_get_current_branch(repo_path: str = None) -> str:
    """Get current git branch name."""
    repo_path = repo_path or BOOKECHI_REPO_PATH
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def git_get_recent_commits(repo_path: str = None, count: int = 5) -> List[dict]:
    """Get recent git commits."""
    repo_path = repo_path or BOOKECHI_REPO_PATH
    import subprocess
    try:
        result = subprocess.run(
            ["git", "log", f"-{count}", "--pretty=format:%H|%an|%ar|%s"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            return []
        commits = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|", 3)
            if len(parts) >= 4:
                commits.append({
                    "hash": parts[0][:8],
                    "author": parts[1],
                    "date": parts[2],
                    "message": parts[3]
                })
        return commits
    except Exception:
        return []


def git_get_changed_files(repo_path: str = None) -> dict:
    """Get list of changed files (staged, unstaged, untracked)."""
    repo_path = repo_path or BOOKECHI_REPO_PATH
    import subprocess
    result = {"staged": [], "modified": [], "untracked": []}
    try:
        # Staged files
        proc = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5
        )
        if proc.returncode == 0:
            result["staged"] = [f for f in proc.stdout.strip().split("\n") if f]
        
        # Modified files (not staged)
        proc = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5
        )
        if proc.returncode == 0:
            result["modified"] = [f for f in proc.stdout.strip().split("\n") if f]
        
        # Untracked files
        proc = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5
        )
        if proc.returncode == 0:
            result["untracked"] = [f for f in proc.stdout.strip().split("\n") if f]
        
        return result
    except Exception:
        return result


def git_read_file(file_path: str, repo_path: str = None) -> Optional[str]:
    """Read file content from the repository."""
    repo_path = repo_path or BOOKECHI_REPO_PATH
    full_path = os.path.join(repo_path, file_path)
    try:
        if os.path.exists(full_path) and os.path.isfile(full_path):
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
    except Exception:
        pass
    return None


def git_list_files(directory: str = "", repo_path: str = None, extension: str = None) -> List[str]:
    """List files in the repository directory."""
    repo_path = repo_path or BOOKECHI_REPO_PATH
    target_dir = os.path.join(repo_path, directory) if directory else repo_path
    files = []
    try:
        if os.path.isdir(target_dir):
            for root, _, filenames in os.walk(target_dir):
                for filename in filenames:
                    if extension and not filename.endswith(extension):
                        continue
                    rel_path = os.path.relpath(os.path.join(root, filename), repo_path)
                    files.append(rel_path)
    except Exception:
        pass
    return files[:100]  # Limit to 100 files


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /help command - answers questions about the Bookechi project.
    Uses RAG for documentation and Git integration for current repo state.
    """
    if not context.args:
        # Show help menu
        branch = git_get_current_branch()
        commits = git_get_recent_commits(count=3)
        
        commits_text = ""
        if commits:
            commits_text = "\n📝 Последние коммиты:\n"
            for c in commits:
                commits_text += f"  • `{c['hash']}` {c['message'][:40]}... ({c['date']})\n"
        
        help_text = (
            "📚 *Bookechi Project Assistant*\n\n"
            f"🌿 Текущая ветка: `{branch}`\n"
            f"{commits_text}\n"
            "Я могу помочь с вопросами о проекте Bookechi:\n\n"
            "*Примеры вопросов:*\n"
            "• `/help как добавить новую книгу?`\n"
            "• `/help какая архитектура используется?`\n"
            "• `/help покажи пример ViewModel`\n"
            "• `/help какие зависимости в проекте?`\n"
            "• `/help как работает Room в проекте?`\n"
            "• `/help покажи структуру модели Book`\n\n"
            "*Команды Git:*\n"
            "• `/git_status` — статус репозитория\n"
            "• `/git_branch` — текущая ветка\n"
            "• `/git_log` — последние коммиты\n"
            "• `/git_files [путь]` — файлы в директории\n"
            "• `/git_show <файл>` — показать содержимое файла\n"
        )
        await update.message.reply_text(help_text, parse_mode="Markdown")
        return
    
    question = " ".join(context.args).strip()
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    # Get git context
    branch = git_get_current_branch()
    changed = git_get_changed_files()
    
    git_context = f"Текущая ветка: {branch}\n"
    if changed["staged"]:
        git_context += f"Staged файлы: {', '.join(changed['staged'][:5])}\n"
    if changed["modified"]:
        git_context += f"Измененные файлы: {', '.join(changed['modified'][:5])}\n"
    
    # Get RAG context
    try:
        min_score = user_kb_min_score.get(update.effective_user.id, KB_MIN_SCORE_DEFAULT)
        rag_context, dbg = kb_retrieve(question, top_k=7, min_score=min_score)
    except Exception as e:
        await update.message.reply_text(
            f"❌ Ошибка базы знаний: {e}\n\n"
            "Попробуй сначала выполнить `/kb_reindex` для индексации документации."
        )
        return
    
    if not rag_context:
        await update.message.reply_text(
            "ℹ️ Не нашёл релевантной информации в документации проекта.\n\n"
            "Попробуй переформулировать вопрос или спроси что-то конкретное о:\n"
            "• Архитектуре MVI\n"
            "• Моделях данных (Book, ReadingSession)\n"
            "• UI компонентах\n"
            "• Room базе данных\n"
            "• Навигации"
        )
        return
    
    # Build system prompt for project assistant
    system = (
        "Ты ассистент по проекту Bookechi — Android-приложению для отслеживания чтения книг.\n"
        "Проект написан на Kotlin с использованием Jetpack Compose, архитектуры MVI, Room и Koin.\n\n"
        "Твоя задача — отвечать на вопросы о проекте, используя предоставленный контекст.\n"
        "Если вопрос о коде — приводи примеры из контекста.\n"
        "Если вопрос о правилах стиля — объясняй паттерны, используемые в проекте.\n"
        "Отвечай на русском языке, структурировано и по делу.\n"
        "Если в контексте нет информации — честно скажи об этом."
    )
    
    prompt = (
        f"GIT СТАТУС:\n{git_context}\n\n"
        f"КОНТЕКСТ ИЗ ДОКУМЕНТАЦИИ И КОДА:\n{rag_context}\n\n"
        f"ВОПРОС ПОЛЬЗОВАТЕЛЯ:\n{question}"
    )
    
    try:
        model = get_model(update.effective_user.id)
        if model == "deepseek":
            messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
            completion = hf_client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3",
                messages=messages,
                temperature=0.3,
                max_tokens=1500,
            )
            answer = (completion.choices[0].message.content or "").strip()
        else:
            messages = [{"role": "system", "text": system}, {"role": "user", "text": prompt}]
            result = yandex_sdk.models.completions("yandexgpt").configure(
                temperature=0.3,
                max_tokens=1500,
            ).run(messages)
            answer = ""
            for alt in result:
                if hasattr(alt, "text"):
                    answer = (alt.text or "").strip()
                    break
        
        footer = (
            f"\n\n---\n"
            f"📚 Источники: {dbg.get('kept', 0)} фрагментов | "
            f"🌿 Ветка: {branch}"
        )
        await update.message.reply_text((answer or "❌ Пустой ответ модели") + footer)
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка LLM: {e}")


async def cmd_git_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show git repository status."""
    branch = git_get_current_branch()
    changed = git_get_changed_files()
    
    status_lines = [
        f"🌿 *Ветка:* `{branch}`\n"
    ]
    
    if changed["staged"]:
        status_lines.append("📦 *Staged:*")
        for f in changed["staged"][:10]:
            status_lines.append(f"  • `{f}`")
    
    if changed["modified"]:
        status_lines.append("\n✏️ *Изменённые:*")
        for f in changed["modified"][:10]:
            status_lines.append(f"  • `{f}`")
    
    if changed["untracked"]:
        status_lines.append("\n❓ *Untracked:*")
        for f in changed["untracked"][:10]:
            status_lines.append(f"  • `{f}`")
    
    if not any([changed["staged"], changed["modified"], changed["untracked"]]):
        status_lines.append("\n✅ Рабочая директория чистая")
    
    await update.message.reply_text("\n".join(status_lines), parse_mode="Markdown")


async def cmd_git_branch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show current git branch."""
    branch = git_get_current_branch()
    await update.message.reply_text(f"🌿 Текущая ветка: `{branch}`", parse_mode="Markdown")


async def cmd_git_log(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show recent git commits."""
    count = 10
    if context.args:
        try:
            count = int(context.args[0])
            count = min(max(count, 1), 30)
        except ValueError:
            pass
    
    commits = git_get_recent_commits(count=count)
    if not commits:
        await update.message.reply_text("❌ Не удалось получить коммиты")
        return
    
    lines = ["📝 *Последние коммиты:*\n"]
    for c in commits:
        lines.append(f"`{c['hash']}` {c['message'][:50]}")
        lines.append(f"  👤 {c['author']} • {c['date']}\n")
    
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_git_files(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """List files in repository directory."""
    directory = context.args[0] if context.args else ""
    extension = None
    
    # Check for extension filter
    if len(context.args) > 1:
        ext_arg = context.args[1]
        if ext_arg.startswith("."):
            extension = ext_arg
        elif ext_arg.startswith("*."):
            extension = ext_arg[1:]
    
    files = git_list_files(directory=directory, extension=extension)
    if not files:
        await update.message.reply_text(f"📁 Директория `{directory or '/'}` пуста или не найдена")
        return
    
    # Group by directory
    lines = [f"📁 *Файлы в `{directory or '/'}`*:\n"]
    shown = files[:30]
    for f in shown:
        lines.append(f"  • `{f}`")
    
    if len(files) > 30:
        lines.append(f"\n... и ещё {len(files) - 30} файлов")
    
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_git_show(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show file content from repository."""
    if not context.args:
        await update.message.reply_text(
            "Использование: `/git_show <путь_к_файлу>`\n\n"
            "Пример: `/git_show app/src/main/java/fuwafuwa/time/bookechi/data/model/Book.kt`",
            parse_mode="Markdown"
        )
        return
    
    file_path = " ".join(context.args).strip()
    content = git_read_file(file_path)
    
    if content is None:
        await update.message.reply_text(f"❌ Файл не найден: `{file_path}`", parse_mode="Markdown")
        return
    
    # Detect language for syntax highlighting
    ext = os.path.splitext(file_path)[1].lower()
    lang = {
        ".kt": "kotlin",
        ".kts": "kotlin",
        ".java": "java",
        ".xml": "xml",
        ".json": "json",
        ".md": "markdown",
        ".py": "python",
    }.get(ext, "")
    
    # Truncate if too long
    if len(content) > 3500:
        content = content[:3500] + "\n\n... (файл обрезан, слишком длинный)"
    
    msg = f"📄 `{file_path}`\n\n```{lang}\n{content}\n```"
    
    try:
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception:
        # Fallback without formatting
        await update.message.reply_text(f"📄 {file_path}\n\n{content[:3500]}")


async def cmd_kb_reindex(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    src = os.path.abspath(KB_SOURCE_PATH)
    idx = os.path.abspath(KB_INDEX_PATH)
    tools_script = os.path.join(REPO_ROOT, "tools", "build_doc_index.py")

    if not os.path.exists(src):
        await update.message.reply_text(f"❌ KB source not found: {src}")
        return

    os.makedirs(os.path.dirname(idx), exist_ok=True)

    # Run build_doc_index.py as a subprocess (non-blocking).
    cmd = [os.environ.get("PYTHON", "python3"), tools_script, "--input", src, "--store", "sqlite", "--out", idx]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=REPO_ROOT,
    )
    out_b, err_b = await proc.communicate()
    out = (out_b or b"").decode("utf-8", errors="replace").strip()
    err = (err_b or b"").decode("utf-8", errors="replace").strip()

    if proc.returncode != 0:
        await update.message.reply_text(f"❌ KB reindex failed (code={proc.returncode})\n\n{err or out}")
        return

    # Invalidate cache so next query reloads updated index.
    _kb_cache.update({"path": None, "mtime": None, "meta": None, "chunks": None, "emb": None, "model": None})
    await update.message.reply_text(f"✅ KB index rebuilt.\n\n{out or 'ok'}")


async def cmd_kb_ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Использование: /kb_ask <вопрос>")
        return

    question = " ".join(context.args).strip()
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        min_score = user_kb_min_score.get(update.effective_user.id, KB_MIN_SCORE_DEFAULT)
        ctx_text, dbg = kb_retrieve(question, min_score=min_score)
    except Exception as e:
        await update.message.reply_text(f"❌ KB error: {e}\n\nСначала сделай /kb_reindex")
        return

    if not ctx_text:
        await update.message.reply_text("ℹ️ В базе знаний не нашёл релевантных фрагментов (контекст пуст).")
        return

    # Ask chosen model with the retrieved context.
    model = get_model(update.effective_user.id)
    system = (
        "Ты помощник. Отвечай на вопрос, используя ТОЛЬКО контекст из базы знаний ниже. "
        "Если в контексте нет ответа — скажи, что в базе знаний этого нет."
    )
    prompt = f"КОНТЕКСТ (из базы знаний):\n{ctx_text}\n\nВОПРОС:\n{question}"

    try:
        if model == "deepseek":
            messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
            completion = hf_client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3",
                messages=messages,
                temperature=get_temperature(update.effective_user.id),
                max_tokens=get_max_tokens(update.effective_user.id) or 800,
            )
            answer = (completion.choices[0].message.content or "").strip()
        else:
            messages = [{"role": "system", "text": system}, {"role": "user", "text": prompt}]
            result = yandex_sdk.models.completions("yandexgpt").configure(
                temperature=get_temperature(update.effective_user.id),
                max_tokens=get_max_tokens(update.effective_user.id) or 800,
            ).run(messages)
            answer = ""
            for alt in result:
                if hasattr(alt, "text"):
                    answer = (alt.text or "").strip()
                    break
        footer = (
            "\n\n---\n"
            f"KB: {dbg['chunks']} chunks | top_k={dbg['top_k']} | min_score={dbg.get('min_score', 0):.3f}\n"
            f"retrieved={dbg.get('retrieved', 0)} kept={dbg.get('kept', 0)} best={dbg.get('best_score')} ctx_chars={dbg.get('context_chars', 0)}\n"
            f"top_scores={dbg.get('top_scores')} kept_scores={dbg.get('kept_scores')} fallback_used={dbg.get('fallback_used')}"
        )
        await update.message.reply_text((answer or "❌ Пустой ответ модели") + footer)
    except Exception as e:
        await update.message.reply_text(f"❌ LLM error: {e}\n\nКонтекст:\n{ctx_text[:1500]}")


def _llm_one_shot(
    *,
    user_id: int,
    system: str,
    user_prompt: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> tuple[str, int, int]:
    """
    One-off LLM call without touching chat history.
    Returns (text, input_tokens, output_tokens).
    """
    model = get_model(user_id)
    t = get_temperature(user_id) if temperature is None else float(temperature)
    mt = get_max_tokens(user_id) if max_tokens is None else int(max_tokens)
    if not mt:
        mt = 800

    history = [{"role": "system", "text": system}, {"role": "user", "text": user_prompt}]
    if model == "deepseek":
        text, in_tok, out_tok = ask_deepseek(history, t, mt)
    else:
        text, in_tok, out_tok = ask_yandex(history, t, mt)
    return (text or "").strip(), int(in_tok or 0), int(out_tok or 0)


def should_use_rag(user_id: int, question: str, chat_history: List[dict]) -> tuple[bool, str]:
    """
    Smart router: determines if RAG (knowledge base) should be used for the question.
    
    Returns:
        (should_use: bool, reason: str)
    """
    # Build a compact history summary for context
    history_summary = ""
    recent_messages = chat_history[-6:] if len(chat_history) > 6 else chat_history  # Last 3 turns
    for msg in recent_messages:
        if msg.get("role") in ("user", "assistant"):
            text = msg.get("text", "")[:200]  # Truncate long messages
            history_summary += f"{msg['role']}: {text}\n"
    
    router_system = f"""Ты router-агент. Твоя задача — определить, нужно ли обращаться к базе знаний (RAG) для ответа на вопрос пользователя.

{KB_DESCRIPTION}

ПРАВИЛА:
1. Если вопрос касается тем из базы знаний (продукты, цены, контакты, доставка, возвраты, подписки) — ответь "RAG"
2. Если вопрос общий, философский, о погоде, новостях, программировании, математике и т.п. — ответь "NO_RAG"
3. Если вопрос продолжает предыдущую тему из истории диалога и НЕ требует новых фактов из базы — ответь "NO_RAG"  
4. Если в истории уже был контекст из базы знаний и вопрос уточняющий — ответь "NO_RAG"
5. При сомнениях — лучше ответь "RAG" (лишний контекст лучше, чем недостающий)

ВАЖНО: Отвечай ТОЛЬКО одним словом: RAG или NO_RAG"""

    router_prompt = f"""История диалога (последние сообщения):
{history_summary if history_summary else "(пусто)"}

Новый вопрос пользователя: {question}

Нужен ли RAG?"""

    try:
        response, _, _ = _llm_one_shot(
            user_id=user_id,
            system=router_system,
            user_prompt=router_prompt,
            temperature=0.1,  # Low temperature for consistent routing
            max_tokens=10
        )
        
        response_clean = response.strip().upper()
        
        if "NO_RAG" in response_clean or "NO RAG" in response_clean:
            return False, "Роутер решил: RAG не нужен"
        elif "RAG" in response_clean:
            return True, "Роутер решил: нужен RAG"
        else:
            # Default to RAG if unclear
            return True, f"Роутер неопределён ({response_clean}), используем RAG"
    except Exception as e:
        # On error, default to RAG
        return True, f"Ошибка роутера ({e}), используем RAG по умолчанию"


def _try_compare_judge(
    *,
    user_id: int,
    question: str,
    rag_context: str,
    answer_no_rag: str,
    answer_rag: str,
) -> Optional[dict]:
    """
    Uses the currently selected model to produce a strict JSON comparison.
    Returns parsed JSON or None.
    """
    system = (
        "Ты строгий reviewer качества ответов LLM. Сравни два ответа на один и тот же вопрос. "
        "Если дан КОНТЕКСТ — он является источником фактов для оценки. "
        "Верни СТРОГО валидный JSON без текста вокруг."
    )
    user = f"""ВОПРОС:
{question}

КОНТЕКСТ (RAG):
{rag_context}

ОТВЕТ A (без RAG):
{answer_no_rag}

ОТВЕТ B (с RAG):
{answer_rag}

Верни JSON со схемой:
{{
  "winner": "A" | "B" | "tie",
  "where_rag_helped": [string, ...],
  "where_rag_not_needed": [string, ...],
  "where_rag_hurt": [string, ...],
  "factuality_notes": [string, ...],
  "confidence": number
}}
"""
    try:
        judge_text, _in_tok, _out_tok = _llm_one_shot(
            user_id=user_id,
            system=system,
            user_prompt=user,
            temperature=0.1,
            max_tokens=900,
        )
        if not judge_text:
            return None
        return json.loads(judge_text)
    except Exception:
        return None


async def cmd_kb_compare(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Compare answers with RAG vs without RAG for the same question.
    Flow: question -> retrieve chunks -> build context -> ask LLM (no-RAG) + ask LLM (RAG) -> conclusion.
    """
    if not context.args:
        await update.message.reply_text("Использование: /kb_compare <вопрос>")
        return

    user_id = update.effective_user.id
    question = " ".join(context.args).strip()
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        min_score = user_kb_min_score.get(user_id, KB_MIN_SCORE_DEFAULT)
        rag_context, dbg = kb_retrieve(question, min_score=min_score)
    except Exception as e:
        await update.message.reply_text(f"❌ KB error: {e}\n\nСначала сделай /kb_reindex")
        return

    # Build prompts
    no_rag_system = (
        "Ты полезный ассистент. Отвечай по делу, без выдуманных фактов. "
        "Если информации недостаточно — честно скажи об этом."
    )
    rag_system = (
        "Ты помощник. Отвечай на вопрос, используя ТОЛЬКО контекст из базы знаний ниже. "
        "Если в контексте нет ответа — скажи, что в базе знаний этого нет."
    )
    rag_prompt = f"КОНТЕКСТ (из базы знаний):\n{rag_context}\n\nВОПРОС:\n{question}"

    try:
        ans_a, a_in, a_out = _llm_one_shot(user_id=user_id, system=no_rag_system, user_prompt=question)
        ans_b, b_in, b_out = _llm_one_shot(user_id=user_id, system=rag_system, user_prompt=rag_prompt)
    except Exception as e:
        await update.message.reply_text(f"❌ LLM error: {e}")
        return

    judge = _try_compare_judge(
        user_id=user_id,
        question=question,
        rag_context=rag_context,
        answer_no_rag=ans_a,
        answer_rag=ans_b,
    )

    # Render conclusion (prefer judge; fallback to a minimal heuristic)
    if judge:
        winner = judge.get("winner", "tie")
        helped = judge.get("where_rag_helped") or []
        not_needed = judge.get("where_rag_not_needed") or []
        hurt = judge.get("where_rag_hurt") or []
        notes = judge.get("factuality_notes") or []
        conclusion_lines = [
            f"Winner: {winner}",
            f"RAG helped: {len(helped)}",
            f"RAG not needed: {len(not_needed)}",
            f"RAG hurt: {len(hurt)}",
        ]
        if helped:
            conclusion_lines.append("Где RAG помог:")
            conclusion_lines.extend([f"- {x}" for x in helped[:6]])
        if hurt:
            conclusion_lines.append("Где RAG навредил/ухудшил:")
            conclusion_lines.extend([f"- {x}" for x in hurt[:6]])
        if not_needed:
            conclusion_lines.append("Где RAG был не нужен:")
            conclusion_lines.extend([f"- {x}" for x in not_needed[:4]])
        if notes:
            conclusion_lines.append("Заметки по фактам:")
            conclusion_lines.extend([f"- {x}" for x in notes[:6]])
        conclusion = "\n".join(conclusion_lines).strip()
    else:
        conclusion = (
            "Judge недоступен/вернул не-JSON, поэтому авто-вывод ограничен.\n"
            "Рекомендация: если ответ с RAG опирается на конкретные факты из контекста и меньше «галлюцинирует» — RAG помог."
        )

    def _truncate(s: str, limit: int) -> str:
        s = (s or "").strip()
        if len(s) <= limit:
            return s
        return s[: max(0, limit - 20)].rstrip() + "\n…(truncated)…"

    # Keep output compact for Telegram
    ctx_hint = "(контекст пуст)" if not rag_context else f"(контекст: {len(rag_context)} chars)"
    footer = (
        f"\n\n---\n"
        f"KB: {dbg['chunks']} chunks | top_k={dbg['top_k']} | min_score={dbg.get('min_score', 0):.3f} {ctx_hint}\n"
        f"retrieved={dbg.get('retrieved', 0)} kept={dbg.get('kept', 0)} best={dbg.get('best_score')} ctx_chars={dbg.get('context_chars', 0)}\n"
        f"top_scores={dbg.get('top_scores')} kept_scores={dbg.get('kept_scores')} fallback_used={dbg.get('fallback_used')}\n"
        f"A tokens: in={a_in} out={a_out} | B tokens: in={b_in} out={b_out}"
    )

    # Telegram message limit is ~4096 chars; split if needed.
    ans_a_short = _truncate(ans_a or "∅", 2500)
    ans_b_short = _truncate(ans_b or "∅", 2500)
    conclusion_short = _truncate(conclusion, 1600)

    msg = (
        "🧪 KB compare (без RAG vs с RAG)\n\n"
        f"Вопрос:\n{question}\n\n"
        "A) Ответ без RAG:\n"
        f"{ans_a_short}\n\n"
        "B) Ответ с RAG:\n"
        f"{ans_b_short}\n\n"
        "Вывод:\n"
        f"{conclusion_short}"
        f"{footer}"
    )
    if len(msg) <= 3800:
        await update.message.reply_text(msg)
        return

    # Fallback: send in 3 messages.
    await update.message.reply_text(
        "🧪 KB compare (без RAG vs с RAG)\n\n"
        f"Вопрос:\n{question}\n\n"
        "A) Ответ без RAG:\n"
        f"{ans_a_short}"
    )
    await update.message.reply_text(
        "B) Ответ с RAG:\n"
        f"{ans_b_short}"
    )
    await update.message.reply_text(
        "Вывод:\n"
        f"{conclusion_short}"
        f"{footer}"
    )


def _try_ab_judge(
    *,
    user_id: int,
    question: str,
    context_a: str,
    answer_a: str,
    context_b: str,
    answer_b: str,
) -> Optional[dict]:
    """Judge for A/B comparisons where both answers may have different contexts."""
    system = (
        "Ты строгий reviewer качества ответов LLM. Сравни два ответа A и B на один и тот же вопрос. "
        "Для каждого ответа дан свой КОНТЕКСТ — он является источником фактов. "
        "Верни СТРОГО валидный JSON без текста вокруг."
    )
    user = f"""ВОПРОС:
{question}

КОНТЕКСТ A:
{context_a}

ОТВЕТ A:
{answer_a}

КОНТЕКСТ B:
{context_b}

ОТВЕТ B:
{answer_b}

Верни JSON со схемой:
{{
  "winner": "A" | "B" | "tie",
  "why": [string, ...],
  "where_filter_helped": [string, ...],
  "where_filter_hurt": [string, ...],
  "confidence": number
}}
"""
    try:
        judge_text, _in_tok, _out_tok = _llm_one_shot(
            user_id=user_id,
            system=system,
            user_prompt=user,
            temperature=0.1,
            max_tokens=900,
        )
        if not judge_text:
            return None
        return json.loads(judge_text)
    except Exception:
        return None


async def cmd_kb_compare_filter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Compare RAG quality without filtering vs with filtering threshold.
    A: min_score=0.0 (no filter)
    B: min_score=user threshold (filter)
    """
    if not context.args:
        await update.message.reply_text("Использование: /kb_compare_filter <вопрос>")
        return

    user_id = update.effective_user.id
    question = " ".join(context.args).strip()
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    min_score_b = user_kb_min_score.get(user_id, KB_MIN_SCORE_DEFAULT)
    try:
        ctx_a, dbg_a = kb_retrieve(question, min_score=0.0, allow_fallback=False)
        # Strict filtered mode: if nothing passes the threshold, context stays empty.
        ctx_b, dbg_b = kb_retrieve(question, min_score=min_score_b, allow_fallback=False)
    except Exception as e:
        await update.message.reply_text(f"❌ KB error: {e}\n\nСначала сделай /kb_reindex")
        return

    rag_system = (
        "Ты помощник. Отвечай на вопрос, используя ТОЛЬКО контекст из базы знаний ниже. "
        "Если в контексте нет ответа — скажи, что в базе знаний этого нет."
    )
    prompt_a = f"КОНТЕКСТ (из базы знаний):\n{ctx_a}\n\nВОПРОС:\n{question}"
    prompt_b = f"КОНТЕКСТ (из базы знаний):\n{ctx_b}\n\nВОПРОС:\n{question}"

    try:
        ans_a, a_in, a_out = _llm_one_shot(user_id=user_id, system=rag_system, user_prompt=prompt_a)
        ans_b, b_in, b_out = _llm_one_shot(user_id=user_id, system=rag_system, user_prompt=prompt_b)
    except Exception as e:
        await update.message.reply_text(f"❌ LLM error: {e}")
        return

    judge = _try_ab_judge(
        user_id=user_id,
        question=question,
        context_a=ctx_a,
        answer_a=ans_a,
        context_b=ctx_b,
        answer_b=ans_b,
    )

    def _truncate(s: str, limit: int) -> str:
        s = (s or "").strip()
        if len(s) <= limit:
            return s
        return s[: max(0, limit - 20)].rstrip() + "\n…(truncated)…"

    if judge:
        winner = judge.get("winner", "tie")
        why = judge.get("why") or []
        helped = judge.get("where_filter_helped") or []
        hurt = judge.get("where_filter_hurt") or []
        conclusion_lines = [
            f"Winner: {winner}",
        ]
        if why:
            conclusion_lines.append("Почему:")
            conclusion_lines.extend([f"- {x}" for x in why[:6]])
        if helped:
            conclusion_lines.append("Где фильтр помог:")
            conclusion_lines.extend([f"- {x}" for x in helped[:6]])
        if hurt:
            conclusion_lines.append("Где фильтр ухудшил:")
            conclusion_lines.extend([f"- {x}" for x in hurt[:6]])
        conclusion = "\n".join(conclusion_lines).strip()
    else:
        conclusion = (
            "Judge недоступен/вернул не-JSON, поэтому авто-вывод ограничен.\n"
            "Ориентир: если в B меньше нерелевантных деталей и больше точных фактов из KB — фильтр помог."
        )

    footer = (
        "\n\n---\n"
        f"A (no filter): min_score=0.000 retrieved={dbg_a.get('retrieved', 0)} kept={dbg_a.get('kept', 0)} best={dbg_a.get('best_score')} ctx_chars={dbg_a.get('context_chars')} top_scores={dbg_a.get('top_scores')} kept_scores={dbg_a.get('kept_scores')}\n"
        f"B (filtered): min_score={dbg_b.get('min_score', 0):.3f} retrieved={dbg_b.get('retrieved', 0)} kept={dbg_b.get('kept', 0)} best={dbg_b.get('best_score')} ctx_chars={dbg_b.get('context_chars')} top_scores={dbg_b.get('top_scores')} kept_scores={dbg_b.get('kept_scores')}\n"
        f"A tokens: in={a_in} out={a_out} | B tokens: in={b_in} out={b_out}"
    )

    ans_a_short = _truncate(ans_a or "∅", 2200)
    ans_b_short = _truncate(ans_b or "∅", 2200)
    conclusion_short = _truncate(conclusion, 1500)

    msg = (
        "🧪 KB compare filter (RAG без порога vs с порогом)\n\n"
        f"Вопрос:\n{question}\n\n"
        "A) RAG без порога (min_score=0.0):\n"
        f"{ans_a_short}\n\n"
        f"B) RAG с порогом (min_score={min_score_b:.3f}):\n"
        f"{ans_b_short}\n\n"
        "Вывод:\n"
        f"{conclusion_short}"
        f"{footer}"
    )
    if len(msg) <= 3800:
        await update.message.reply_text(msg)
        return

    # Fallback: split into multiple messages.
    await update.message.reply_text(
        "🧪 KB compare filter (RAG без порога vs с порогом)\n\n"
        f"Вопрос:\n{question}\n\n"
        "A) RAG без порога (min_score=0.0):\n"
        f"{ans_a_short}"
    )
    await update.message.reply_text(
        f"B) RAG с порогом (min_score={min_score_b:.3f}):\n"
        f"{ans_b_short}"
    )
    await update.message.reply_text(
        "Вывод:\n"
        f"{conclusion_short}"
        f"{footer}"
    )

# === 1. СОЗДАНИЕ SDK КЛИЕНТОВ ===
# YandexGPT
yandex_sdk = YCloudML(folder_id=YANDEX_FOLDER_ID, auth=YANDEX_AUTH)

# HuggingFace (DeepSeek)
hf_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN or ""
)


# === MCP CLIENT ===
class McpClient:
    """Клиент для взаимодействия с MCP сервером."""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self._request_id = 0
    
    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id
    
    async def _send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Отправляет JSON-RPC запрос к MCP серверу."""
        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method
        }
        if params:
            request["params"] = params
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.server_url,
                json=request,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
    
    async def initialize(self) -> Dict[str, Any]:
        """Инициализирует соединение с MCP сервером."""
        result = await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {}
        })
        return result.get("result", {})
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """Получает список доступных инструментов."""
        result = await self._send_request("tools/list")
        return result.get("result", {}).get("tools", [])
    
    async def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Вызывает инструмент на MCP сервере."""
        params = {"name": tool_name}
        if arguments:
            params["arguments"] = arguments
        
        response = await self._send_request("tools/call", params)
        
        # Проверяем на ошибку JSON-RPC
        if "error" in response and response["error"]:
            error = response["error"]
            return {
                "content": [{"type": "text", "text": f"Error: {error.get('message', 'Unknown error')}"}],
                "isError": True
            }
        
        return response.get("result") or {}


# Глобальные MCP клиенты
mcp_client = McpClient(MCP_SERVER_URL)  # Calendar MCP
mcp_events = McpClient(MCP_EVENTS_URL)  # KudaGo Events MCP
mobile_mcp_service = MobileMcpService(command=shlex.split(MOBILE_MCP_COMMAND))

# Selected Mobile MCP device per chat (so /mobile_call can auto-inject {"device": "..."}).
mobile_selected_device: Dict[int, str] = {}


# Доступные модели
MODELS = {
    "yandex": "YandexGPT",
    "deepseek": "DeepSeek-V3"
}

# === 2. СИСТЕМНЫЙ ПРОМПТ ===
SYSTEM_PROMPT = """
отвечай на вопросы очень подробно
ИСПОЛЬЗУЙ ВСЕ ДОСТУПНЫЕ ТОКЕНЫ ДЛЯ ОТВЕТА
"""

# === 3. ИСТОРИЯ СООБЩЕНИЙ ДЛЯ КАЖДОГО ПОЛЬЗОВАТЕЛЯ ===
# Ключ - ID пользователя в Telegram, значение - список сообщений
user_histories: Dict[int, List[dict]] = {}

# === 4. ТЕМПЕРАТУРА ДЛЯ КАЖДОГО ПОЛЬЗОВАТЕЛЯ ===
# 0 = строгие ответы, 1 = креативные ответы
user_temperatures: Dict[int, float] = {}

# === 5. ВЫБРАННАЯ МОДЕЛЬ ДЛЯ КАЖДОГО ПОЛЬЗОВАТЕЛЯ ===
user_models: Dict[int, str] = {}  # "yandex" или "deepseek"

# === 6. ЛИМИТ ТОКЕНОВ ДЛЯ КАЖДОГО ПОЛЬЗОВАТЕЛЯ ===
user_max_tokens: Dict[int, int] = {}  # Максимум токенов в ответе (0 = без лимита)

# === 7. ПРЕДЫДУЩЕЕ ЗНАЧЕНИЕ INPUT TOKENS (для расчёта токенов текущего запроса) ===
user_prev_input_tokens: Dict[int, int] = {}

# === 8. СЖАТИЕ ИСТОРИИ ===
# Индекс последнего сообщения после последнего сжатия для каждого пользователя
user_last_compressed_idx: Dict[int, int] = {}

# Количество сообщений (user + assistant, каждое считается отдельно) до срабатывания триггера сжатия (0 = отключено)
user_compress_trigger_turns: Dict[int, int] = {}

def get_history(user_id: int) -> List[dict]:
    """
    Получает историю для пользователя. Создаёт новую, если её нет.
    При создании новой истории загружает суммаризацию из локального хранилища.
    """
    if user_id not in user_histories:
        user_histories[user_id] = [{"role": "system", "text": SYSTEM_PROMPT}]
        # Инициализируем индекс сжатия для нового пользователя
        user_last_compressed_idx[user_id] = -1
        
        # Загружаем суммаризацию из локального хранилища, если есть
        combined_summary = get_combined_summary(user_id)
        if combined_summary:
            summary_msg = {
                "role": "system",
                "name": "summary",
                "text": f"Краткий конспект всех предыдущих частей диалога:\n{combined_summary}"
            }
            user_histories[user_id].append(summary_msg)
            # Устанавливаем индекс сжатия на позицию summary
            user_last_compressed_idx[user_id] = 1
            print(f"✓ Загружена суммаризация из локального хранилища для user_id={user_id}")
    
    return user_histories[user_id]


def clear_history(user_id: int, clear_summaries_too: bool = False):
    """
    Очищает историю пользователя.
    
    Args:
        user_id: ID пользователя
        clear_summaries_too: Если True, также очищает сохраненные суммаризации
    """
    user_histories[user_id] = [{"role": "system", "text": SYSTEM_PROMPT}]
    user_prev_input_tokens[user_id] = 0  # Сбрасываем счётчик токенов
    user_last_compressed_idx[user_id] = -1  # Сбрасываем индекс сжатия
    
    if clear_summaries_too:
        clear_summaries(user_id)
        print(f"✓ Суммаризации очищены для user_id={user_id}")
    else:
        # Загружаем существующую суммаризацию
        combined_summary = get_combined_summary(user_id)
        if combined_summary:
            summary_msg = {
                "role": "system",
                "name": "summary",
                "text": f"Краткий конспект всех предыдущих частей диалога:\n{combined_summary}"
            }
            user_histories[user_id].append(summary_msg)
            user_last_compressed_idx[user_id] = 1
            print(f"✓ Загружена суммаризация при очистке истории для user_id={user_id}")

def change_system_prompt(user_id: int, prompt: str):
    """Изменяет системный промпт для пользователя."""
    user_histories[user_id].append({"role": "system", "text": prompt})


def get_temperature(user_id: int) -> float:
    """Получает температуру для пользователя. По умолчанию 0.5."""
    return user_temperatures.get(user_id, 0.5)


def set_temperature(user_id: int, temp: float):
    """Устанавливает температуру для пользователя."""
    user_temperatures[user_id] = temp


def get_model(user_id: int) -> str:
    """Получает выбранную модель для пользователя. По умолчанию yandex."""
    return user_models.get(user_id, "yandex")


def set_model(user_id: int, model: str):
    """Устанавливает модель для пользователя."""
    user_models[user_id] = model


def get_max_tokens(user_id: int) -> int:
    """Получает лимит токенов для пользователя. 0 = без лимита."""
    return user_max_tokens.get(user_id, 0)


def set_max_tokens(user_id: int, max_tokens: int):
    """Устанавливает лимит токенов для пользователя."""
    user_max_tokens[user_id] = max_tokens


def get_compress_trigger(user_id: int) -> int:
    """Получает количество сообщений для триггера сжатия. 0 = отключено."""
    return user_compress_trigger_turns.get(user_id, 10)  # По умолчанию 10 сообщений


def set_compress_trigger(user_id: int, turns: int):
    """Устанавливает количество сообщений для триггера сжатия. 0 = отключить сжатие."""
    user_compress_trigger_turns[user_id] = turns


@dataclass
class AgentResponse:
    """Результат ответа агента с метриками."""
    text: str
    input_tokens: int      # Токены всей истории (context)
    output_tokens: int     # Токены ответа
    total_tokens: int      # Всего токенов в этом запросе
    message_tokens: int    # Токены только текущего сообщения пользователя
    time_seconds: float
    cost_rub: float  # Примерная стоимость в рублях
    model: str = ""  # Название модели
    rag_used: bool = False  # Использовался ли RAG (база знаний)
    history_used: bool = False  # Использовалась ли история диалога


# Цены YandexGPT (примерные, руб за 1000 токенов)
PRICE_INPUT_PER_1K = 0.12   # входные токены
PRICE_OUTPUT_PER_1K = 0.24  # выходные токены


def ask_yandex(history: List[dict], temperature: float, max_tokens: int = 0) -> tuple:
    """Запрос к YandexGPT. max_tokens ограничивает только ответ (completion), не контекст."""
    model = yandex_sdk.models.completions("yandexgpt")
    
    # Настраиваем параметры генерации
    if max_tokens > 0:
        result = model.configure(temperature=temperature, max_tokens=max_tokens).run(history)
    else:
        result = model.configure(temperature=temperature).run(history)
    
    response_text = ""
    input_tokens = 0
    output_tokens = 0
    
    for alt in result:
        if hasattr(alt, 'text'):
            response_text = alt.text
    
    if hasattr(result, 'usage'):
        usage = result.usage
        input_tokens = getattr(usage, 'input_text_tokens', 0)
        output_tokens = getattr(usage, 'completion_tokens', 0)
    
    return response_text, input_tokens, output_tokens


def ask_deepseek(history: List[dict], temperature: float, max_tokens: int = 0) -> tuple:
    """Запрос к DeepSeek через HuggingFace."""
    # Конвертируем формат истории (text -> content)
    messages = []
    for msg in history:
        messages.append({
            "role": msg["role"],
            "content": msg.get("text", msg.get("content", ""))
        })
    
    kwargs = {
        "model": "deepseek-ai/DeepSeek-V3",
        "messages": messages,
        "temperature": temperature
    }
    if max_tokens > 0:
        kwargs["max_tokens"] = max_tokens
    
    completion = hf_client.chat.completions.create(**kwargs)
    
    response_text = completion.choices[0].message.content or ""
    input_tokens = completion.usage.prompt_tokens if completion.usage else 0
    output_tokens = completion.usage.completion_tokens if completion.usage else 0
    
    return response_text, input_tokens, output_tokens


def ask_agent(user_id: int, question: str, rag_used: bool = False) -> AgentResponse:
    """Отправляет вопрос агенту и получает ответ с метриками."""
    history = get_history(user_id)
    model = get_model(user_id)
    temperature = get_temperature(user_id)
    max_tokens = get_max_tokens(user_id)
    
    # Проверяем, использовалась ли история (до добавления нового вопроса)
    # История считается использованной, если есть сообщения кроме системного промпта
    history_used = len(history) > 1  # Больше чем только system prompt
    
    # Проверяем и сжимаем историю ПЕРЕД добавлением нового вопроса
    compressed_before = check_and_compress_history(
        user_id=user_id,
        history=history,
        last_compressed_idx=user_last_compressed_idx,
        trigger_turns=user_compress_trigger_turns,
        yandex_sdk=yandex_sdk,
        hf_client=hf_client,
        model=model
    )
    
    # Получаем предыдущее значение input_tokens для расчёта токенов сообщения
    prev_input_tokens = user_prev_input_tokens.get(user_id, 0)
    
    # Добавляем вопрос в историю
    history.append({"role": "user", "text": question})
    
    # Замеряем время
    start_time = time.time()
    
    # Запрос к выбранной модели
    if model == "deepseek":
        response_text, input_tokens, output_tokens = ask_deepseek(history, temperature, max_tokens)
    else:
        response_text, input_tokens, output_tokens = ask_yandex(history, temperature, max_tokens)
    
    elapsed_time = time.time() - start_time
    total_tokens = input_tokens + output_tokens
    
    # Рассчитываем токены только текущего сообщения
    # (разница между текущим context и предыдущим context + предыдущий ответ)
    message_tokens = input_tokens - prev_input_tokens
    if message_tokens < 0:
        message_tokens = input_tokens  # Если история была очищена
    
    # Сохраняем текущее значение для следующего запроса
    # (input_tokens + output_tokens = следующий prev_input_tokens)
    user_prev_input_tokens[user_id] = input_tokens + output_tokens
    
    # Рассчитываем стоимость (примерно)
    cost = (input_tokens / 1000 * PRICE_INPUT_PER_1K) + (output_tokens / 1000 * PRICE_OUTPUT_PER_1K)
    
    # Добавляем ответ в историю
    history.append({"role": "assistant", "text": response_text})
    
    # Проверяем и сжимаем историю ПОСЛЕ добавления ответа
    compressed_after = check_and_compress_history(
        user_id=user_id,
        history=history,
        last_compressed_idx=user_last_compressed_idx,
        trigger_turns=user_compress_trigger_turns,
        yandex_sdk=yandex_sdk,
        hf_client=hf_client,
        model=model
    )
    
    return AgentResponse(
        text=response_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        message_tokens=message_tokens,
        time_seconds=elapsed_time,
        cost_rub=cost,
        model=MODELS[model],
        rag_used=rag_used,
        history_used=history_used
    )


# === ОБРАБОТЧИКИ КОМАНД ===

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    user_id = update.effective_user.id
    clear_history(user_id)
    
    summary_count = get_summary_count(user_id)
    summary_info = f"\n📦 Загружено суммаризаций из памяти: {summary_count}" if summary_count > 0 else ""
    
    await update.message.reply_text(
        f"👋 Привет! Я бот-ассистент проекта *Bookechi*.{summary_info}\n\n"
        "Просто напиши мне вопрос, и я отвечу.\n\n"
        "📚 *Bookechi — помощь по проекту:*\n"
        "/help — помощь по проекту (RAG + Git)\n"
        "/help <вопрос> — спросить о проекте\n"
        "/review <github\\_url> — ревью коммита/PR\n\n"
        "🎧 *Поддержка пользователей:*\n"
        "/support — статус режима поддержки\n"
        "/support\\_on — включить режим поддержки\n"
        "/support\\_off — выключить\n"
        "/ticket\\_new <тема> — создать тикет\n"
        "/ticket\\_list — мои тикеты\n"
        "/ticket\\_view <id> — просмотр тикета\n"
        "/ticket\\_select <id> — выбрать тикет\n"
        "/ticket\\_close <id> — закрыть тикет\n"
        "/tickets\\_all — все открытые тикеты\n\n"
        "🔧 *Git интеграция (MCP):*\n"
        "/git\\_status — статус репозитория\n"
        "/git\\_branch — текущая ветка\n"
        "/git\\_log [N] — последние коммиты\n"
        "/git\\_files [путь] — файлы в директории\n"
        "/git\\_show <файл> — содержимое файла\n\n"
        "📖 *KB (RAG) — база знаний Bookechi:*\n"
        "/kb\\_status — статус базы\n"
        "/kb\\_ask <вопрос> — спросить по базе\n"
        "/kb\\_auto\\_on — умный режим RAG\n"
        "/kb\\_off — выключить RAG\n\n"
        "⚙️ *Настройки бота:*\n"
        "/model — выбрать модель\n"
        "/clear — очистить историю\n"
        "/temperature — температура генерации\n\n"
        "_Для полного списка команд используй /commands_",
        parse_mode="Markdown"
    )


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /clear [all]"""
    user_id = update.effective_user.id
    
    # Проверяем, нужно ли очистить также суммаризации
    clear_all = context.args and context.args[0].lower() == "all"
    
    clear_history(user_id, clear_summaries_too=clear_all)
    
    summary_count = get_summary_count(user_id)
    
    if clear_all:
        await update.message.reply_text("🗑 История и все суммаризации полностью очищены!")
    elif summary_count > 0:
        await update.message.reply_text(
            f"🗑 История очищена!\n"
            f"📦 Сохранено суммаризаций: {summary_count}\n\n"
            "Используй /clear all для полной очистки включая суммаризации."
        )
    else:
        await update.message.reply_text("🗑 История очищена!")


async def cmd_set_system_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /set_system_prompt <новый промпт>"""
    user_id = update.effective_user.id
    
    # Получаем текст после команды
    # context.args содержит список слов после команды
    if not context.args:
        await update.message.reply_text(
            "⚠️ Укажи новый промпт после команды.\n\n"
        )
        return
    
    # Собираем все слова в один текст
    new_prompt = " ".join(context.args)
    
    # Меняем промпт
    change_system_prompt(user_id, new_prompt)
    
    await update.message.reply_text(
        f"✅ Системный промпт изменён!\n\n"
        f"Новый промпт:\n{new_prompt}"
    )


async def cmd_temperature(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /temperature - показывает текущую температуру"""
    user_id = update.effective_user.id
    current_temp = get_temperature(user_id)
    await update.message.reply_text(f"🌡 Текущая температура: {current_temp}")


async def cmd_set_temperature(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /set_temperature <число от 0 до 1>"""
    user_id = update.effective_user.id
    
    if not context.args:
        await update.message.reply_text(
            "Использование: /set_temperature <число>\n"
            "• 0 - строгие, точные ответы\n"
            "• 1 - креативные, разнообразные ответы\n\n"
            "Пример: /set_temperature 0.7"
        )
        return
    
    try:
        new_temp = float(context.args[0])
        if not 0 <= new_temp <= 1:
            raise ValueError("Температура должна быть от 0 до 1")
        
        set_temperature(user_id, new_temp)
        await update.message.reply_text(f"🌡 Температура установлена: {new_temp}")
    except ValueError as e:
        await update.message.reply_text(f"❌ Ошибка: {e}\nУкажи число от 0 до 1")


async def cmd_max_tokens(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /max_tokens - показывает текущий лимит токенов"""
    user_id = update.effective_user.id
    current_limit = get_max_tokens(user_id)
    if current_limit == 0:
        await update.message.reply_text("📏 Лимит токенов: без ограничений")
    else:
        await update.message.reply_text(f"📏 Лимит токенов: {current_limit}")


async def cmd_set_max_tokens(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /set_max_tokens <число>"""
    user_id = update.effective_user.id
    
    if not context.args:
        current_limit = get_max_tokens(user_id)
        await update.message.reply_text(
            f"📏 Текущий лимит: {current_limit if current_limit > 0 else 'без ограничений'}\n\n"
            "Использование: /set_max_tokens <число>\n"
            "• 0 - без ограничений\n"
            "• 100-8000 - лимит токенов в ответе\n\n"
            "Пример: /set_max_tokens 500"
        )
        return
    
    try:
        new_limit = int(context.args[0])
        if new_limit < 0:
            raise ValueError("Лимит не может быть отрицательным")
        
        set_max_tokens(user_id, new_limit)
        if new_limit == 0:
            await update.message.reply_text("📏 Лимит токенов снят (без ограничений)")
        else:
            await update.message.reply_text(f"📏 Лимит токенов установлен: {new_limit}")
    except ValueError as e:
        await update.message.reply_text(f"❌ Ошибка: {e}\nУкажи целое число >= 0")


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /model - показывает кнопки выбора модели"""
    user_id = update.effective_user.id
    current_model = get_model(user_id)
    
    # Создаём кнопки
    keyboard = [
        [
            InlineKeyboardButton(
                f"{'✅ ' if current_model == 'yandex' else ''}YandexGPT",
                callback_data="model_yandex"
            ),
            InlineKeyboardButton(
                f"{'✅ ' if current_model == 'deepseek' else ''}DeepSeek-V3",
                callback_data="model_deepseek"
            ),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        f"🤖 Текущая модель: {MODELS[current_model]}\n\nВыбери модель:",
        reply_markup=reply_markup
    )


async def handle_model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик нажатия кнопок выбора модели"""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    
    if query.data == "model_yandex":
        set_model(user_id, "yandex")
        selected = "YandexGPT"
    elif query.data == "model_deepseek":
        set_model(user_id, "deepseek")
        selected = "DeepSeek-V3"
    else:
        return
    
    # Обновляем кнопки с новой галочкой
    current_model = get_model(user_id)
    keyboard = [
        [
            InlineKeyboardButton(
                f"{'✅ ' if current_model == 'yandex' else ''}YandexGPT",
                callback_data="model_yandex"
            ),
            InlineKeyboardButton(
                f"{'✅ ' if current_model == 'deepseek' else ''}DeepSeek-V3",
                callback_data="model_deepseek"
            ),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        f"🤖 Модель выбрана: {selected}\n\nВыбери модель:",
        reply_markup=reply_markup
    )


async def cmd_compress_trigger(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /compress_trigger - показывает текущие настройки сжатия истории"""
    user_id = update.effective_user.id
    current_trigger = get_compress_trigger(user_id)
    if current_trigger == 0:
        await update.message.reply_text(
            "📦 Сжатие истории: отключено\n\n"
            "Используй /set_compress_trigger <число> для включения.\n"
            "Например: /set_compress_trigger 10"
        )
    else:
        await update.message.reply_text(
            f"📦 Триггер сжатия истории: каждые {current_trigger} сообщений (user + assistant)\n\n"
            "Используй /set_compress_trigger <число> для изменения.\n"
            "Используй /set_compress_trigger 0 для отключения."
        )


async def cmd_set_compress_trigger(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /set_compress_trigger <число>"""
    user_id = update.effective_user.id
    
    if not context.args:
        current_trigger = get_compress_trigger(user_id)
        await update.message.reply_text(
            f"📦 Текущий триггер: {current_trigger if current_trigger > 0 else 'отключено'}\n\n"
            "Использование: /set_compress_trigger <число>\n"
            "• 0 - отключить сжатие истории\n"
            "• 5-50 - количество сообщений (user + assistant, каждое считается отдельно) до сжатия\n\n"
            "Пример: /set_compress_trigger 10\n"
            "(История будет сжиматься каждые 10 сообщений)"
        )
        return
    
    try:
        new_trigger = int(context.args[0])
        if new_trigger < 0:
            raise ValueError("Триггер не может быть отрицательным")
        
        set_compress_trigger(user_id, new_trigger)
        if new_trigger == 0:
            await update.message.reply_text("📦 Сжатие истории отключено")
        else:
            await update.message.reply_text(
                f"📦 Триггер сжатия установлен: каждые {new_trigger} сообщений"
            )
    except ValueError as e:
        await update.message.reply_text(f"❌ Ошибка: {e}\nУкажи целое число >= 0")


# === MCP КОМАНДЫ ===

async def cmd_mcp_tools(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /mcp_tools - показывает список инструментов MCP сервера"""
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    # Примеры использования для каждого инструмента
    tool_examples = {
        # Calendar tools
        "get_today_events": "",
        "get_upcoming_events": "7",
        "get_events_for_date": "2024-12-25",
        "create_event": "Spatb 2025-12-18 15:00 16:00",
        "get_daily_summary": "",
        "list_calendars": "",
    }
    
    try:
        # Получаем список инструментов
        tools = await mcp_client.list_tools()
        
        if not tools:
            await update.message.reply_text("🔧 MCP сервер не предоставляет инструментов.")
            return
        
        # Формируем красивый вывод
        message = f"🔧 *MCP Tools*\n\n"
        
        for i, tool in enumerate(tools, 1):
            name = tool.get("name", "unknown")
            description = tool.get("description", "Нет описания")
            input_schema = tool.get("inputSchema", {})
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])
            
            message += f"*{i}. {name}*\n"
            message += f"📝 {description}\n"
            
            if properties:
                message += "📥 Параметры:\n"
                for prop_name, prop_info in properties.items():
                    prop_type = prop_info.get("type", "any")
                    is_required = "✅" if prop_name in required else "⬜"
                    message += f"  {is_required} {prop_name} ({prop_type})\n"
            
            # Добавляем пример использования
            example_arg = tool_examples.get(name, "")
            if example_arg:
                message += f"💡 `/mcp_call {name} {example_arg}`\n"
            else:
                message += f"💡 `/mcp_call {name}`\n"
            
            message += "\n"
        
        await update.message.reply_text(message, parse_mode="Markdown")
        
    except httpx.ConnectError:
        await update.message.reply_text(
            f"❌ Не удалось подключиться к MCP серверу.\n\n"
            f"URL: {MCP_SERVER_URL}\n"
            f"Убедитесь, что сервер запущен."
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка при получении инструментов: {e}")


async def cmd_mcp_call(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /mcp_call <tool_name> [args] - вызывает инструмент MCP"""
    if not context.args:
        await update.message.reply_text(
            "🔧 *Вызов MCP инструмента*\n\n"
            "*Календарь:*\n"
            "`/mcp_call get_today_events`\n"
            "`/mcp_call get_upcoming_events 7`\n"
            "`/mcp_call get_events_for_date 2024-12-25`\n"
            "`/mcp_call create_event Встреча 2024-12-20 14:00 15:00`\n"
            '`/mcp_call create_event "Team Sync" 2024-12-20 14:00 15:00`\n'
            "`/mcp_call get_daily_summary`\n"
            "`/mcp_call list_calendars`\n\n"
            "Используй /mcp\\_tools для списка всех инструментов.",
            parse_mode="Markdown"
        )
        return
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    tool_name = context.args[0]
    arguments = None
    
    # Парсим аргументы
    if len(context.args) > 1:
        args_str = " ".join(context.args[1:])
        
        # Проверяем, это JSON или простое значение
        if args_str.startswith("{"):
            # JSON формат
            try:
                arguments = json.loads(args_str)
            except json.JSONDecodeError as e:
                await update.message.reply_text(
                    f"❌ Ошибка парсинга JSON:\n`{e}`\n\n"
                    f"Входная строка: `{args_str}`",
                    parse_mode="Markdown"
                )
                return
        else:
            # Простой формат: автоматически определяем параметры
            parts = args_str.split()
            
            if tool_name == "list_pokemon":
                # /mcp_call list_pokemon [limit] [offset]
                arguments = {}
                if len(parts) >= 1:
                    try:
                        arguments["limit"] = int(parts[0])
                    except ValueError:
                        pass
                if len(parts) >= 2:
                    try:
                        arguments["offset"] = int(parts[1])
                    except ValueError:
                        pass
            elif tool_name == "get_upcoming_events":
                # /mcp_call get_upcoming_events [days]
                arguments = {}
                if len(parts) >= 1:
                    try:
                        arguments["days"] = int(parts[0])
                    except ValueError:
                        pass
            elif tool_name == "get_events_for_date":
                # /mcp_call get_events_for_date YYYY-MM-DD
                arguments = {"date": parts[0]} if parts else {}
            elif tool_name == "create_event":
                # /mcp_call create_event title date start_time end_time [description]
                # Example: /mcp_call create_event Meeting 2024-12-20 14:00 15:00 Team sync
                # Or with quotes: /mcp_call create_event "Team Meeting" 2024-12-20 14:00 15:00
                
                # Check if title is quoted
                import shlex
                try:
                    parsed_parts = shlex.split(args_str)
                except ValueError:
                    parsed_parts = parts
                
                if len(parsed_parts) >= 4:
                    arguments = {
                        "title": parsed_parts[0],
                        "date": parsed_parts[1],
                        "start_time": parsed_parts[2],
                        "end_time": parsed_parts[3],
                        "description": " ".join(parsed_parts[4:]) if len(parsed_parts) > 4 else ""
                    }
                else:
                    await update.message.reply_text(
                        "❌ Недостаточно параметров для create\\_event\n\n"
                        "Формат: `/mcp_call create_event title date start end [desc]`\n\n"
                        "Примеры:\n"
                        "`/mcp_call create_event Meeting 2024-12-20 14:00 15:00`\n"
                        '`/mcp_call create_event "Team Sync" 2024-12-20 14:00 15:00 Weekly`',
                        parse_mode="Markdown"
                    )
                    return
            elif tool_name in ["get_today_events", "get_daily_summary"]:
                # Эти инструменты не требуют параметров
                arguments = {}
            else:
                # Для остальных инструментов - параметр "name"
                arguments = {"name": args_str}
    
    try:
        # Вызываем инструмент
        start_time = time.time()
        result = await mcp_client.call_tool(tool_name, arguments)
        elapsed = time.time() - start_time
        
        # Формируем ответ (с защитой от None)
        if result is None:
            result = {}
        
        content = result.get("content", []) or []
        is_error = result.get("isError", False)
        
        # Извлекаем текст из content
        output_text = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                output_text += item.get("text", "") + "\n"
        
        if not output_text:
            output_text = json.dumps(result, ensure_ascii=False, indent=2)
        
        status = "❌ Ошибка" if is_error else "✅ Успешно"
        
        # Экранируем специальные символы Markdown в output_text
        # (чтобы не ломать форматирование)
        safe_output = output_text.replace("_", "\\_").replace("*", "\\*").replace("`", "\\`")
        
        message = (
            f"🔧 *MCP Tool Call*\n\n"
            f"📛 Инструмент: `{tool_name}`\n"
            f"📥 Аргументы: `{json.dumps(arguments, ensure_ascii=False) if arguments else 'нет'}`\n"
            f"⏱ Время: {elapsed:.3f}s\n"
            f"📊 Статус: {status}\n\n"
            f"📤 *Результат:*\n{safe_output}"
        )
        
        await update.message.reply_text(message, parse_mode="Markdown")
        
    except httpx.ConnectError:
        await update.message.reply_text(
            f"❌ Не удалось подключиться к MCP серверу.\n\n"
            f"URL: {MCP_SERVER_URL}\n"
            f"Убедитесь, что сервер запущен."
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка при вызове инструмента: {e}")


async def cmd_mcp_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /mcp_status - проверяет статус MCP сервера"""
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        start_time = time.time()
        server_info = await mcp_client.initialize()
        elapsed = time.time() - start_time
        
        protocol_version = server_info.get("protocolVersion", "unknown")
        server_name = server_info.get("serverInfo", {}).get("name", "unknown")
        server_version = server_info.get("serverInfo", {}).get("version", "unknown")
        capabilities = server_info.get("capabilities", {})
        
        # Получаем количество инструментов
        tools = await mcp_client.list_tools()
        tools_count = len(tools)
        
        message = (
            f"🟢 **MCP Server Status**\n\n"
            f"🔗 URL: `{MCP_SERVER_URL}`\n"
            f"📛 Имя: {server_name}\n"
            f"📦 Версия: {server_version}\n"
            f"📋 Протокол: {protocol_version}\n"
            f"🔧 Инструментов: {tools_count}\n"
            f"⏱ Ping: {elapsed*1000:.1f}ms\n\n"
            f"Capabilities: `{json.dumps(capabilities, ensure_ascii=False)}`"
        )
        
        await update.message.reply_text(message, parse_mode="Markdown")
        
    except httpx.ConnectError:
        await update.message.reply_text(
            f"🔴 **MCP Server Offline**\n\n"
            f"URL: `{MCP_SERVER_URL}`\n"
            f"Сервер недоступен.",
            parse_mode="Markdown"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка проверки статуса: {e}")


# === MOBILE MCP (stdio) COMMANDS ===

async def cmd_mobile_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    try:
        info = await mobile_mcp_service.ensure_started()
        await update.message.reply_text(
            "🟢 *Mobile MCP started*\n\n"
            "_Примечание: это запускает MCP сервер. Эмулятор/симулятор запускается отдельными командами:_\n"
            "`/android_boot ...` или `/ios_boot ...`\n\n"
            f"📛 {info.name}\n"
            f"📦 {info.version}\n"
            f"📋 Protocol: {info.protocol_version}\n",
            parse_mode="Markdown",
        )
    except Exception as e:
        stderr = mobile_mcp_service.recent_stderr()
        extra = f"\n\nstderr:\n{stderr[-1500:]}" if stderr else ""
        await update.message.reply_text(f"❌ Mobile MCP start error: {e}{extra}")


async def cmd_mobile_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    try:
        await mobile_mcp_service.stop()
        await update.message.reply_text("🛑 Mobile MCP stopped")
    except Exception as e:
        await update.message.reply_text(f"❌ Mobile MCP stop error: {e}")


async def cmd_mobile_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    running = mobile_mcp_service.client.is_running
    inited = mobile_mcp_service.client.initialized
    stderr = mobile_mcp_service.recent_stderr().strip()
    msg = (
        "📱 *Mobile MCP Status*\n\n"
        f"Running: {'✅' if running else '❌'}\n"
        f"Initialized: {'✅' if inited else '❌'}\n"
        f"Command: `{MOBILE_MCP_COMMAND}`\n"
    )
    if stderr:
        msg += f"\nRecent stderr (tail):\n`{stderr[-800:]}`"
    await update.message.reply_text(msg, parse_mode="Markdown")


async def cmd_mobile_tools(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    try:
        tools = await mobile_mcp_service.list_tools()
        if not tools:
            await update.message.reply_text("🔧 Mobile MCP server does not expose tools.")
            return

        # IMPORTANT: send as plain text (no Markdown) because tool names/descriptions
        # may contain characters that break Telegram entity parsing.
        header = "🔧 Mobile MCP Tools\n\n"
        footer = (
            "\n\nПример:\n"
            "/mobile_call <tool> {\"x\":10,\"y\":20}\n"
            "/mobile_call <tool> x=10 y=20"
        )

        max_tools = 120
        lines: List[str] = []
        for i, t in enumerate(tools[:max_tools], 1):
            name = str(t.get("name", "unknown"))
            desc = str(t.get("description", "") or "")
            line = f"{i}) {name}"
            if desc:
                # keep lines bounded
                if len(desc) > 240:
                    desc = desc[:240] + "…"
                line += f" — {desc}"
            lines.append(line)

        if len(tools) > max_tools:
            lines.append(f"\n… and {len(tools) - max_tools} more")

        text = header + "\n".join(lines) + footer

        # Telegram message limit ~4096 chars: chunk safely
        chunk_size = 3500
        for start in range(0, len(text), chunk_size):
            await update.message.reply_text(text[start : start + chunk_size])
    except Exception as e:
        stderr = mobile_mcp_service.recent_stderr()
        extra = f"\n\nstderr:\n{stderr[-1500:]}" if stderr else ""
        await update.message.reply_text(f"❌ Mobile tools error: {e}{extra}")


async def cmd_mobile_call(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "Использование:\n"
            "`/mobile_call <tool> {\"k\":\"v\"}`\n"
            "`/mobile_call <tool> k=v k2=v2`\n"
            "\nСписок tools: /mobile_tools",
            parse_mode="Markdown",
        )
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    tool = context.args[0]
    arg_str = " ".join(context.args[1:]).strip()

    args_obj: Optional[Dict[str, Any]] = None
    if arg_str:
        if arg_str.startswith("{"):
            try:
                args_obj = json.loads(arg_str)
            except json.JSONDecodeError as e:
                await update.message.reply_text(f"❌ JSON parse error: {e}\n`{arg_str}`", parse_mode="Markdown")
                return
        elif "=" in arg_str:
            args_obj = parse_kv_args(arg_str)
        else:
            # Best-effort convenience for common patterns.
            # Prefer explicit JSON via /mobile_tool <name> to see exact schema.
            if "open_url" in tool.lower() or tool.lower().endswith("openurl"):
                url = arg_str.strip()
                if url and "://" not in url:
                    url = "https://" + url
                args_obj = {"url": url}
            else:
                # fallback: many tools accept `text`
                args_obj = {"text": arg_str}

    # IMPORTANT: Mobile MCP expects an object for arguments (even if empty).
    if args_obj is None:
        args_obj = {}

    # Auto-inject selected device if not provided.
    chat_id = update.effective_chat.id if update.effective_chat else None
    if chat_id is not None and "device" not in args_obj and chat_id in mobile_selected_device:
        args_obj["device"] = mobile_selected_device[chat_id]

    result = await safe_call(mobile_mcp_service, tool, args_obj)
    # Some Mobile MCP tools model "no args" as a required `noParams` object.
    # If we see that validation error, retry once with {"noParams": {}} merged.
    if result.get("isError"):
        err_text = extract_text_from_mcp_result(result)
        if "noParams" in err_text and "expected object" in err_text and "received undefined" in err_text:
            retry_args = dict(args_obj)
            retry_args.setdefault("noParams", {})
            result = await safe_call(mobile_mcp_service, tool, retry_args)
            args_obj = retry_args
    text = extract_text_from_mcp_result(result)
    is_error = bool(result.get("isError"))
    status = "❌ Ошибка" if is_error else "✅ Успешно"

    images = extract_images_from_mcp_result(result)
    if images:
        # send images first
        for idx, (raw, mime) in enumerate(images, 1):
            bio = io.BytesIO(raw)
            bio.seek(0)
            filename = f"mobile_screen_{idx}.png" if "png" in mime else f"mobile_screen_{idx}.jpg"
            input_file = InputFile(bio, filename=filename)
            if mime.startswith("image/"):
                await context.bot.send_photo(chat_id=update.effective_chat.id, photo=input_file)
            else:
                await context.bot.send_document(chat_id=update.effective_chat.id, document=input_file)

    if not text:
        text = json.dumps(result, ensure_ascii=False, indent=2)[:3500]
    await update.message.reply_text(
        f"🔧 *Mobile MCP Tool Call*\n\n"
        f"📛 Tool: `{tool}`\n"
        f"📥 Args: `{json.dumps(args_obj, ensure_ascii=False)}`\n"
        f"📊 Статус: {status}\n\n"
        f"📤 Результат:\n{text}",
        parse_mode="Markdown",
    )


async def cmd_mobile_devices(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Best-effort: find a tool that lists devices and call it.
    Different Mobile MCP versions may expose different names.
    """
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    tools = await mobile_mcp_service.list_tools()
    candidates = [
        "mobile_list_devices",
        "mobile_devices",
        "list_devices",
        "get_devices",
        "devices_list",
        "mobile_connected_devices",
        "mobile_list_connected_devices",
    ]
    tool_name = pick_tool_name(tools, candidates)
    if not tool_name:
        # Fallback: find any tool containing "device" + "list"
        for t in tools:
            name = str(t.get("name", ""))
            if "device" in name.lower() and "list" in name.lower():
                tool_name = name
                break

    if not tool_name:
        await update.message.reply_text(
            "❌ Не нашёл tool для списка устройств.\n"
            "Сделай /mobile_tools и найди tool, который возвращает devices, затем вызови его через /mobile_call."
        )
        return

    args_obj: Dict[str, Any] = {}
    result = await safe_call(mobile_mcp_service, tool_name, args_obj)
    if result.get("isError"):
        err_text = extract_text_from_mcp_result(result)
        if "noParams" in err_text and "expected object" in err_text and "received undefined" in err_text:
            args_obj = {"noParams": {}}
            result = await safe_call(mobile_mcp_service, tool_name, args_obj)

    text = extract_text_from_mcp_result(result) or json.dumps(result, ensure_ascii=False, indent=2)

    msg = (
        f"📱 Mobile devices (tool: {tool_name})\n"
        f"Args: {json.dumps(args_obj, ensure_ascii=False)}\n\n"
        f"{text}\n\n"
        "Выбери device id и сделай:\n"
        "/mobile_use <device>\n\n"
        "Потом можно вызывать:\n"
        "/mobile_call mobile_list_apps\n"
        "/mobile_call mobile_open_url google.com"
    )
    for start in range(0, len(msg), 3500):
        await update.message.reply_text(msg[start : start + 3500])


async def cmd_mobile_use(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Select current device for this chat.
    Usage: /mobile_use <device>
    """
    if not context.args:
        await update.message.reply_text("Использование: /mobile_use <device>")
        return
    device = " ".join(context.args).strip()
    chat_id = update.effective_chat.id if update.effective_chat else None
    if chat_id is None:
        await update.message.reply_text("❌ Не удалось определить chat_id")
        return
    mobile_selected_device[chat_id] = device
    await update.message.reply_text(f"✅ Selected device: {device}\n\nТеперь /mobile_call будет подставлять device автоматически.")


async def cmd_mobile_tool(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Show tool description + input schema (plain text).
    Usage: /mobile_tool <name>
    """
    if not context.args:
        await update.message.reply_text("Использование: /mobile_tool <tool_name>")
        return
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    name = context.args[0]
    tools = await mobile_mcp_service.list_tools()
    found = None
    for t in tools:
        if str(t.get("name", "")) == name:
            found = t
            break
    if not found:
        # try case-insensitive
        for t in tools:
            if str(t.get("name", "")).lower() == name.lower():
                found = t
                break
    if not found:
        await update.message.reply_text("❌ Tool not found. Use /mobile_tools to list tools.")
        return

    desc = str(found.get("description", "") or "")
    schema = found.get("inputSchema", {}) or {}
    payload = {
        "name": found.get("name"),
        "description": desc,
        "inputSchema": schema,
    }
    text = "🔎 Mobile MCP Tool\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    # Chunk to avoid Telegram limit
    for start in range(0, len(text), 3500):
        await update.message.reply_text(text[start : start + 3500])


async def cmd_tap(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 2:
        await update.message.reply_text("Использование: `/tap <x> <y>`", parse_mode="Markdown")
        return
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    try:
        x = int(context.args[0])
        y = int(context.args[1])
    except ValueError:
        await update.message.reply_text("❌ x и y должны быть числами")
        return

    tools = await mobile_mcp_service.list_tools()
    tool = pick_tool_name(tools, ["tap", "click", "touch", "input_tap"])
    if not tool:
        await update.message.reply_text("❌ Не нашёл tool для tap. Проверь /mobile_tools и используй /mobile_call.")
        return

    result = await safe_call(mobile_mcp_service, tool, {"x": x, "y": y})
    text = extract_text_from_mcp_result(result) or "ok"
    await update.message.reply_text(f"✅ tap via `{tool}`: {text}", parse_mode="Markdown")


async def cmd_screenshot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    tools = await mobile_mcp_service.list_tools()
    tool = pick_tool_name(tools, ["screenshot", "take_screenshot", "screen_capture", "capture_screenshot"])
    if not tool:
        await update.message.reply_text("❌ Не нашёл tool для screenshot. Проверь /mobile_tools и используй /mobile_call.")
        return

    result = await safe_call(mobile_mcp_service, tool, {})
    images = extract_images_from_mcp_result(result)
    if not images:
        # Sometimes servers return base64 in text; show raw text then.
        text = extract_text_from_mcp_result(result) or json.dumps(result, ensure_ascii=False, indent=2)[:3500]
        await update.message.reply_text(f"📸 `{tool}` result:\n{text}", parse_mode="Markdown")
        return

    for idx, (raw, mime) in enumerate(images, 1):
        bio = io.BytesIO(raw)
        bio.seek(0)
        filename = f"screenshot_{idx}.png" if "png" in mime else f"screenshot_{idx}.jpg"
        input_file = InputFile(bio, filename=filename)
        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=input_file)

    text = extract_text_from_mcp_result(result)
    if text:
        await update.message.reply_text(text)


# === Emulator commands ===

async def cmd_android_avds(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    avds = await mobile_mcp_service.android_list_avds()
    if not avds:
        await update.message.reply_text(
            "❌ Не удалось получить список AVD.\n"
            "Проверь, что Android Emulator доступен в PATH, или задай ANDROID_EMULATOR_BIN."
        )
        return
    await update.message.reply_text("📱 Android AVD:\n" + "\n".join([f"- {a}" for a in avds]))


async def cmd_android_boot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Использование: `/android_boot <avd> [headless]`", parse_mode="Markdown")
        return
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    avd = context.args[0]
    headless = len(context.args) > 1 and context.args[1].lower() in ("1", "true", "yes", "headless")
    msg = await mobile_mcp_service.android_boot(avd, headless=headless)
    await update.message.reply_text(msg)


async def cmd_android_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    msg = await mobile_mcp_service.android_stop()
    await update.message.reply_text(msg)

async def cmd_android_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    proc = mobile_mcp_service.emulator.android_proc
    avd = mobile_mcp_service.emulator.android_avd
    running = bool(proc and proc.returncode is None)
    last_err = (mobile_mcp_service.emulator.android_last_error or "").strip()

    msg = (
        "📱 Android Emulator Status\n\n"
        f"Running: {'YES' if running else 'NO'}\n"
        f"AVD: {avd or '-'}\n"
    )
    if proc and proc.returncode is not None:
        msg += f"Exit code: {proc.returncode}\n"
    if last_err:
        msg += "\nLast error (tail):\n" + last_err[-1200:]
    await update.message.reply_text(msg)

async def cmd_ios_devices(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    text = await mobile_mcp_service.ios_list_devices()
    await update.message.reply_text(text)


async def cmd_ios_boot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Использование: `/ios_boot <name|udid>`", parse_mode="Markdown")
        return
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    device = " ".join(context.args)
    text = await mobile_mcp_service.ios_boot(device)
    await update.message.reply_text(text)


async def cmd_ios_open(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    text = await mobile_mcp_service.ios_open_simulator_app()
    await update.message.reply_text(text)

async def cmd_diag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Quick local diagnostics: where bot runs + availability of required binaries.
    """
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    def which(bin_name: str) -> str:
        p = shutil.which(bin_name)
        return p or "NOT FOUND"

    emulator_cfg = os.getenv("ANDROID_EMULATOR_BIN", "emulator")
    emulator_detected = which(emulator_cfg)
    sdk_root = os.getenv("ANDROID_SDK_ROOT") or os.getenv("ANDROID_HOME") or ""
    hint = ""
    if emulator_detected != "NOT FOUND" and "/Android/sdk/tools/emulator" in emulator_detected:
        hint = (
            "\n\nHint: you are using deprecated SDK Tools emulator.\n"
            "Prefer the modern emulator binary:\n"
            "  ~/Library/Android/sdk/emulator/emulator\n"
            "Set ANDROID_EMULATOR_BIN to that full path."
        )

    # IMPORTANT: send as plain text (no Markdown), underscores/backticks can break Telegram entities.
    lines = [
        "🧪 Diagnostics",
        "",
        f"OS: {platform.platform()}",
        f"Python: {platform.python_version()}",
        "",
        f"ANDROID_EMULATOR_BIN: {emulator_cfg}",
        f"emulator (detected): {emulator_detected}",
        f"ANDROID_SDK_ROOT/ANDROID_HOME: {sdk_root or '-'}",
        "",
        f"node: {which('node')}",
        f"npx: {which('npx')}",
        f"xcrun: {which('xcrun')}",
        "",
        f"MOBILE_MCP_COMMAND: {MOBILE_MCP_COMMAND}",
    ]
    await update.message.reply_text("\n".join(lines) + hint)


# === PIPELINE COMMAND ===
# Состояние pipeline для каждого пользователя
user_pipeline_state: Dict[int, Dict[str, Any]] = {}

async def cmd_pipeline(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Команда /pipeline - поиск событий KudaGo и АВТОМАТИЧЕСКОЕ добавление в календарь.
    
    Использование:
    /pipeline <category> <city> [from_date] [to_date] [limit]
    /pipeline concert Moscow
    /pipeline theater spb 2025-12-25 2025-12-31 5
    """
    user_id = update.effective_user.id
    
    if not context.args:
        await update.message.reply_text(
            "🎫 *Pipeline: KudaGo → Яндекс Календарь*\n\n"
            "Автоматический поиск событий и добавление в календарь!\n\n"
            "*Использование:*\n"
            "`/pipeline <категория> [город] [от] [до] [лимит]`\n\n"
            "*Примеры:*\n"
            "`/pipeline concert` — концерты в Москве на 30 дней\n"
            "`/pipeline concert Moscow` — то же самое\n"
            "`/pipeline concert Moscow 7` — на 7 дней вперёд\n"
            "`/pipeline theater spb 2025-12-25` — с 25 декабря\n"
            "`/pipeline theater spb 2025-12-25 2025-12-31` — с 25 по 31 дек\n"
            "`/pipeline exhibition Kazan 2025-12-20 2025-12-30 3` — 3 события\n\n"
            "*Форматы дат:*\n"
            "• `7` или `30` — дней вперёд от сегодня\n"
            "• `2025-12-25` — конкретная дата (YYYY-MM-DD)\n\n"
            "*Категории:*\n"
            "• `concert` — концерты\n"
            "• `theater` — театр\n"
            "• `exhibition` — выставки\n"
            "• `festival` — фестивали\n"
            "• `party` — вечеринки\n\n"
            "*Города:*\n"
            "• Moscow, spb, Kazan, ekb, nnv\n\n"
            "`/pipeline_cities` — все города\n"
            "`/pipeline_categories` — все категории",
            parse_mode="Markdown"
        )
        return
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    # Парсим аргументы
    category = context.args[0] if len(context.args) >= 1 else "concert"
    city = context.args[1] if len(context.args) >= 2 else "Moscow"
    
    # Парсим даты - могут быть в формате дней (7, 30) или дат (2025-12-25)
    from datetime import datetime, timedelta
    
    from_date = None
    to_date = None
    limit = 5
    
    def parse_date_arg(arg: str) -> tuple:
        """Возвращает (date или None, is_days_number)"""
        if arg.isdigit():
            # Это число дней
            return int(arg), True
        elif "-" in arg and len(arg) == 10:
            # Это дата в формате YYYY-MM-DD
            try:
                return datetime.strptime(arg, "%Y-%m-%d").date(), False
            except ValueError:
                return None, False
        return None, False
    
    # Аргумент 3: может быть from_date или days_ahead
    if len(context.args) >= 3:
        parsed, is_days = parse_date_arg(context.args[2])
        if is_days and parsed:
            # Это число дней
            from_date = datetime.now().date()
            to_date = from_date + timedelta(days=parsed)
        elif parsed:
            # Это дата начала
            from_date = parsed
    
    # Аргумент 4: может быть to_date или limit
    if len(context.args) >= 4:
        parsed, is_days = parse_date_arg(context.args[3])
        if is_days and parsed:
            # Если from_date уже установлена как дата, это limit
            if from_date and not to_date:
                limit = parsed
            else:
                # Иначе это to_date как дни
                to_date = datetime.now().date() + timedelta(days=parsed)
        elif parsed:
            # Это дата окончания
            to_date = parsed
    
    # Аргумент 5: limit
    if len(context.args) >= 5 and context.args[4].isdigit():
        limit = int(context.args[4])
    
    # Если даты не указаны, используем по умолчанию 30 дней
    if from_date is None:
        from_date = datetime.now().date()
    if to_date is None:
        to_date = from_date + timedelta(days=30)
    
    # Вычисляем days_ahead для MCP
    days_ahead = (to_date - datetime.now().date()).days
    if days_ahead < 1:
        days_ahead = 1
    
    # Форматируем даты для отображения
    from_str = from_date.strftime("%d.%m.%Y") if hasattr(from_date, 'strftime') else str(from_date)
    to_str = to_date.strftime("%d.%m.%Y") if hasattr(to_date, 'strftime') else str(to_date)
    
    status_msg = await update.message.reply_text(
        f"🔍 *Шаг 1/2:* Ищу {category} в {city}...\n"
        f"📅 Период: {from_str} — {to_str} (до {limit} событий)",
        parse_mode="Markdown"
    )
    
    # Форматируем даты для MCP (YYYY-MM-DD)
    start_date_str = from_date.strftime("%Y-%m-%d") if hasattr(from_date, 'strftime') else str(from_date)
    end_date_str = to_date.strftime("%Y-%m-%d") if hasattr(to_date, 'strftime') else str(to_date)
    
    try:
        # ШАГ 1: Поиск событий через KudaGo MCP
        search_result = await mcp_events.call_tool("search_events", {
            "city": city,
            "category": category,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "limit": limit
        })
        
        if search_result.get("isError"):
            content = search_result.get("content", [])
            error_text = content[0].get("text", "Unknown error") if content else "Unknown error"
            await status_msg.edit_text(f"❌ Ошибка поиска: {error_text}")
            return
        
        # Извлекаем текст результата
        content = search_result.get("content", [])
        result_text = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                result_text += item.get("text", "")
        
        if "No events found" in result_text or not result_text:
            await status_msg.edit_text(
                f"😔 Не найдено событий '{category}' в {city}.\n\n"
                "Попробуйте другую категорию или город."
            )
            return
        
        # Парсим события
        events = parse_events_from_result(result_text)
        
        if not events:
            await status_msg.edit_text(
                f"😔 Не удалось распарсить события.\n\n{result_text[:500]}"
            )
            return
        
        # Обновляем статус
        await status_msg.edit_text(
            f"✅ *Шаг 1/2:* Найдено {len(events)} событий!\n\n"
            f"📅 *Шаг 2/2:* Добавляю в Яндекс Календарь...",
            parse_mode="Markdown"
        )
        
        # ШАГ 2: Добавляем все события в календарь
        results = []
        success_count = 0
        
        for event in events:
            result = await add_event_to_calendar(event)
            if "✅" in result:
                success_count += 1
            results.append(result)
            # Небольшая задержка между запросами
            await asyncio.sleep(0.3)
        
        # Формируем итоговое сообщение
        summary = f"🎫 *Pipeline завершён!*\n\n"
        summary += f"🔍 Категория: {category}\n"
        summary += f"📍 Город: {city}\n"
        summary += f"📊 Добавлено в календарь: {success_count}/{len(events)}\n\n"
        summary += "━━━━━━━━━━━━━━━━━━━━━\n"
        summary += "*Результаты:*\n\n"
        
        for r in results:
            summary += f"{r}\n"
        
        # Сохраняем состояние для возможного повторного добавления
        user_pipeline_state[user_id] = {
            "events": events,
            "category": category,
            "city": city,
            "raw_result": result_text
        }
        
        try:
            await status_msg.edit_text(summary, parse_mode="Markdown")
        except Exception:
            # Если сообщение слишком длинное, отправляем новое
            await update.message.reply_text(summary.replace("*", ""))
        
    except httpx.ConnectError as e:
        error_msg = str(e)
        if "8081" in error_msg or "events" in error_msg.lower():
            await status_msg.edit_text(
                f"❌ KudaGo MCP сервер недоступен.\n\n"
                f"Запустите: `java -jar mcp-ticketmaster-kotlin-1.0.0.jar --http 8081`",
                parse_mode="Markdown"
            )
        else:
            await status_msg.edit_text(
                f"❌ Calendar MCP сервер недоступен.\n\n"
                f"Запустите: `java -jar mcp-server-kotlin-1.0.0.jar --http 8080`",
                parse_mode="Markdown"
            )
    except Exception as e:
        await status_msg.edit_text(f"❌ Ошибка pipeline: {e}")


def parse_events_from_result(result_text: str) -> List[Dict[str, Any]]:
    """Парсит события из текстового результата KudaGo."""
    events = []
    lines = result_text.split("\n")
    
    current_event = {}
    event_num = 0
    
    for line in lines:
        line = line.strip()
        
        # Новое событие начинается с номера и эмодзи 🎫
        if line and line[0].isdigit() and "🎫" in line:
            if current_event and current_event.get("name"):
                events.append(current_event)
            event_num += 1
            # Извлекаем название (после эмодзи)
            name_part = line.split("🎫")[-1].strip() if "🎫" in line else line
            current_event = {
                "num": event_num,
                "name": name_part,
                "date": None,
                "time": None,
                "venue": None,
                "address": None,
                "id": None
            }
        
        # Дата и время: "📅 2024-12-25 at 19:30"
        elif "📅" in line and current_event:
            date_part = line.replace("📅", "").strip()
            if " at " in date_part:
                parts = date_part.split(" at ")
                current_event["date"] = parts[0].strip()
                current_event["time"] = parts[1].strip()
            else:
                current_event["date"] = date_part
        
        # Venue name: "📍 Venue Name"
        elif "📍" in line and current_event:
            current_event["venue"] = line.replace("📍", "").strip()
        
        # Address: "🏠 Address"
        elif "🏠" in line and current_event:
            current_event["address"] = line.replace("🏠", "").strip()
        
        # Event ID: "🆔 ID: 12345"
        elif "🆔" in line and "ID:" in line and current_event:
            id_part = line.split("ID:")[-1].strip()
            current_event["id"] = id_part
    
    # Добавляем последнее событие
    if current_event and current_event.get("name"):
        events.append(current_event)
    
    return events


async def cmd_pipeline_add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Добавляет выбранное событие в календарь."""
    user_id = update.effective_user.id
    
    if user_id not in user_pipeline_state or not user_pipeline_state[user_id].get("events"):
        await update.message.reply_text(
            "❌ Нет сохраненных результатов поиска.\n\n"
            "Сначала выполните поиск: `/pipeline rock Moscow`",
            parse_mode="Markdown"
        )
        return
    
    if not context.args:
        await update.message.reply_text(
            "❌ Укажите номер события.\n\n"
            "Пример: `/pipeline_add 1`",
            parse_mode="Markdown"
        )
        return
    
    try:
        event_num = int(context.args[0])
    except ValueError:
        await update.message.reply_text("❌ Номер должен быть числом.")
        return
    
    events = user_pipeline_state[user_id]["events"]
    
    if event_num < 1 or event_num > len(events):
        await update.message.reply_text(f"❌ Номер должен быть от 1 до {len(events)}.")
        return
    
    event = events[event_num - 1]
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    # Добавляем в календарь
    result = await add_event_to_calendar(event)
    await update.message.reply_text(result)


async def cmd_pipeline_add_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Добавляет все найденные события в календарь."""
    user_id = update.effective_user.id
    
    if user_id not in user_pipeline_state or not user_pipeline_state[user_id].get("events"):
        await update.message.reply_text(
            "❌ Нет сохраненных результатов поиска.\n\n"
            "Сначала выполните поиск: `/pipeline rock Moscow`",
            parse_mode="Markdown"
        )
        return
    
    events = user_pipeline_state[user_id]["events"]
    
    await update.message.reply_text(f"📅 Добавляю {len(events)} событий в календарь...")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    results = []
    success_count = 0
    
    for event in events:
        result = await add_event_to_calendar(event)
        results.append(f"• {event['name'][:30]}... — {'✅' if '✅' in result else '❌'}")
        if "✅" in result:
            success_count += 1
        # Небольшая задержка между запросами
        await asyncio.sleep(0.5)
    
    summary = f"📊 *Результат:* {success_count}/{len(events)} добавлено\n\n" + "\n".join(results)
    
    try:
        await update.message.reply_text(summary, parse_mode="Markdown")
    except Exception:
        await update.message.reply_text(summary.replace("*", ""))


async def add_event_to_calendar(event: Dict[str, Any]) -> str:
    """Добавляет одно событие в календарь через Calendar MCP."""
    name = event.get("name", "Event")
    date = event.get("date")
    time_str = event.get("time", "19:00")
    venue = event.get("venue", "")
    address = event.get("address", "")
    
    if not date or date == "TBD" or "Date TBD" in str(date):
        return f"⏭️ {name[:30]}... — дата не определена, пропущено"
    
    # Парсим время
    if time_str and time_str != "TBD" and "00:00" not in time_str:
        start_time = time_str[:5] if len(time_str) >= 5 else time_str
        # Вычисляем время окончания (+3 часа)
        try:
            hour = int(start_time.split(":")[0])
            minute = start_time.split(":")[1] if ":" in start_time else "00"
            end_hour = (hour + 3) % 24
            end_time = f"{end_hour:02d}:{minute}"
        except Exception:
            end_time = "23:00"
    else:
        start_time = "19:00"
        end_time = "22:00"
    
    # Формируем описание
    description_parts = []
    if venue:
        description_parts.append(f"Место: {venue}")
    if address:
        description_parts.append(f"Адрес: {address}")
    description = "\n".join(description_parts)
    
    try:
        result = await mcp_client.call_tool("create_event", {
            "title": name[:100],  # Ограничиваем длину
            "date": date,
            "start_time": start_time,
            "end_time": end_time,
            "description": description
        })
        
        if result.get("isError"):
            content = result.get("content", [])
            error_text = content[0].get("text", "Error") if content else "Error"
            return f"❌ {name[:30]}... — {error_text}"
        
        return f"✅ {name[:30]}... — добавлено на {date} {start_time}"
        
    except Exception as e:
        return f"❌ {name[:30]}... — ошибка: {str(e)[:50]}"


async def cmd_pipeline_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Очищает сохраненные результаты pipeline."""
    user_id = update.effective_user.id
    
    if user_id in user_pipeline_state:
        del user_pipeline_state[user_id]
    
    await update.message.reply_text("🗑️ Результаты поиска очищены.")


async def cmd_pipeline_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает статус MCP серверов для pipeline."""
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    calendar_status = "❌ Offline"
    events_status = "❌ Offline"
    
    # Проверяем Calendar MCP
    try:
        await mcp_client.initialize()
        calendar_status = "✅ Online"
    except Exception:
        pass
    
    # Проверяем KudaGo Events MCP
    try:
        await mcp_events.initialize()
        events_status = "✅ Online"
    except Exception:
        pass
    
    await update.message.reply_text(
        f"🔗 *Pipeline Status*\n\n"
        f"📅 Calendar MCP: {calendar_status}\n"
        f"   `{MCP_SERVER_URL}`\n\n"
        f"🎫 KudaGo Events MCP: {events_status}\n"
        f"   `{MCP_EVENTS_URL}`\n\n"
        f"{'✅ Pipeline готов к работе!' if calendar_status == '✅ Online' and events_status == '✅ Online' else '⚠️ Запустите оба MCP сервера для работы pipeline.'}",
        parse_mode="Markdown"
    )


async def cmd_pipeline_cities(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает доступные города KudaGo."""
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        result = await mcp_events.call_tool("list_cities", {})
        content = result.get("content", [])
        text = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text += item.get("text", "")
        
        await update.message.reply_text(text or "Не удалось получить список городов")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")


async def cmd_pipeline_categories(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает доступные категории событий KudaGo."""
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        result = await mcp_events.call_tool("list_categories", {})
        content = result.get("content", [])
        text = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text += item.get("text", "")
        
        await update.message.reply_text(text or "Не удалось получить список категорий")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений"""
    user_id = update.effective_user.id
    user_message = update.message.text
    original_message = user_message  # Сохраняем оригинал для роутера
    
    if not user_message:
        return
    
    # === SUPPORT MODE CHECK ===
    # If support mode is enabled, handle through support agent
    if user_support_mode.get(user_id, False):
        response = await handle_support_message(update, user_id, user_message)
        if response:
            await update.message.reply_text(response, parse_mode="Markdown")
            return
    
    # Показываем, что бот "печатает"
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        rag_used = False
        router_reason = ""
        ctx_text = ""
        
        # Определяем режим работы RAG
        kb_auto = user_kb_auto_enabled.get(user_id, False)  # Умный режим
        kb_always = user_kb_enabled.get(user_id, False)      # Принудительный режим
        
        if kb_auto:
            # УМНЫЙ РЕЖИМ: агент сам решает, нужен ли RAG
            history = get_history(user_id)
            should_rag, router_reason = should_use_rag(user_id, original_message, history)
            
            if should_rag:
                try:
                    min_score = user_kb_min_score.get(user_id, KB_MIN_SCORE_DEFAULT)
                    ctx_text, _dbg = kb_retrieve(original_message, min_score=min_score)
                except Exception:
                    ctx_text = ""
                if ctx_text:
                    rag_used = True
                    user_message = (
                        "КОНТЕКСТ (из базы знаний):\n"
                        f"{ctx_text}\n\n"
                        "ВОПРОС:\n"
                        f"{original_message}"
                    )
        elif kb_always:
            # ПРИНУДИТЕЛЬНЫЙ РЕЖИМ: всегда используем RAG
            router_reason = "Принудительный режим (kb_on)"
            try:
                min_score = user_kb_min_score.get(user_id, KB_MIN_SCORE_DEFAULT)
                ctx_text, _dbg = kb_retrieve(user_message, min_score=min_score)
            except Exception:
                ctx_text = ""
            if ctx_text:
                rag_used = True
                user_message = (
                    "КОНТЕКСТ (из базы знаний):\n"
                    f"{ctx_text}\n\n"
                    "ВОПРОС:\n"
                    f"{user_message}"
                )
        
        # Получаем ответ от агента
        response = ask_agent(user_id, user_message, rag_used=rag_used)
        
        # Формируем информацию об источниках
        sources = []
        if response.rag_used:
            sources.append("📚 RAG (база знаний)")
        if response.history_used:
            sources.append("💬 История диалога")
        if not sources:
            sources.append("🆕 Без контекста")
        
        sources_text = " | ".join(sources)
        
        # Добавляем причину роутера, если был умный режим
        router_info = ""
        if kb_auto and router_reason:
            router_info = f"🧭 {router_reason}\n"
        
        # Формируем сообщение с метриками
        stats = (
            f"\n\n---\n"
            f"📌 Источники: {sources_text}\n"
            f"{router_info}"
            f"🤖 {response.model} | ⏱ {response.time_seconds:.2f}s | 💰 {response.cost_rub:.4f}₽\n"
            f"💬 Your message: {response.message_tokens} tokens\n"
            f"📥 Context (history): {response.input_tokens} tokens\n"
            f"📤 Response: {response.output_tokens} tokens\n"
            f"📊 This request total: {response.total_tokens} tokens"
        )
        
        await update.message.reply_text(response.text + stats)
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")


# === DAILY REMINDER ===

async def send_daily_reminder(context: ContextTypes.DEFAULT_TYPE):
    """Отправляет ежедневную сводку из календаря"""
    if not DAILY_REMINDER_CHAT_ID:
        print("⚠️ DAILY_REMINDER_CHAT_ID не установлен, пропускаем напоминание")
        return
    
    try:
        # Получаем daily summary из MCP сервера
        result = await mcp_client.call_tool("get_daily_summary", {})
        
        if result is None:
            result = {}
        
        content = result.get("content", []) or []
        
        # Извлекаем текст
        message_text = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                message_text += item.get("text", "")
        
        if not message_text:
            message_text = "📅 Не удалось получить сводку на сегодня"
        
        # Отправляем сообщение
        await context.bot.send_message(
            chat_id=int(DAILY_REMINDER_CHAT_ID),
            text=message_text
        )
        print(f"✅ Daily reminder sent to chat {DAILY_REMINDER_CHAT_ID}")
        
    except Exception as e:
        print(f"❌ Error sending daily reminder: {e}")


async def cmd_set_reminder(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /set_reminder - управление ежедневными напоминаниями"""
    global DAILY_REMINDER_CHAT_ID, DAILY_REMINDER_HOUR, DAILY_REMINDER_MINUTE
    
    chat_id = update.effective_chat.id
    
    # Показываем текущие настройки
    if not context.args:
        current_status = "✅ включены" if DAILY_REMINDER_CHAT_ID else "❌ отключены"
        
        # Check if there's a scheduled job
        jobs = context.job_queue.get_jobs_by_name("daily_reminder") if context.job_queue else []
        job_status = f"✅ запланировано ({len(jobs)} job)" if jobs else "❌ не запланировано"
        
        await update.message.reply_text(
            f"⏰ *Ежедневные напоминания*\n\n"
            f"Статус: {current_status}\n"
            f"Job: {job_status}\n"
            f"Время: {DAILY_REMINDER_HOUR}:{DAILY_REMINDER_MINUTE:02d} (UTC+{DAILY_REMINDER_TIMEZONE_OFFSET})\n"
            f"Chat ID: `{chat_id}`\n\n"
            f"*Команды:*\n"
            f"`/set_reminder HH:MM` - установить время\n"
            f"`/set_reminder on` - включить для этого чата\n"
            f"`/set_reminder off` - отключить\n"
            f"`/set_reminder test` - тестовая отправка",
            parse_mode="Markdown"
        )
        return
    
    arg = context.args[0].lower()
    
    # Включить напоминания для текущего чата
    if arg == "on":
        DAILY_REMINDER_CHAT_ID = str(chat_id)
        
        # Schedule the job if not already scheduled
        if context.job_queue:
            # Remove existing jobs
            for job in context.job_queue.get_jobs_by_name("daily_reminder"):
                job.schedule_removal()
            
            # Add new job
            tz = timezone(timedelta(hours=DAILY_REMINDER_TIMEZONE_OFFSET))
            reminder_time = dt_time(hour=DAILY_REMINDER_HOUR, minute=DAILY_REMINDER_MINUTE, second=0, tzinfo=tz)
            context.job_queue.run_daily(send_daily_reminder, time=reminder_time, name="daily_reminder")
        
        await update.message.reply_text(
            f"✅ Напоминания включены!\n\n"
            f"Время: {DAILY_REMINDER_HOUR}:{DAILY_REMINDER_MINUTE:02d}\n"
            f"Chat ID: {chat_id}"
        )
        return
    
    # Отключить напоминания
    if arg == "off":
        DAILY_REMINDER_CHAT_ID = None
        
        # Remove scheduled jobs
        if context.job_queue:
            for job in context.job_queue.get_jobs_by_name("daily_reminder"):
                job.schedule_removal()
        
        await update.message.reply_text("❌ Напоминания отключены")
        return
    
    # Тестовая отправка
    if arg == "test":
        await update.message.reply_text("📤 Отправляю тестовое напоминание...")
        
        try:
            result = await mcp_client.call_tool("get_daily_summary", {})
            
            if result is None:
                result = {}
            
            content = result.get("content", []) or []
            message_text = ""
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    message_text += item.get("text", "")
            
            if not message_text:
                message_text = "📅 Не удалось получить сводку"
            
            await update.message.reply_text(message_text)
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {e}")
        return
    
    # Установить время (формат HH:MM)
    if ":" in arg:
        try:
            parts = arg.split(":")
            hour = int(parts[0])
            minute = int(parts[1])
            
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError("Invalid time")
            
            DAILY_REMINDER_HOUR = hour
            DAILY_REMINDER_MINUTE = minute
            
            # Reschedule the job if active
            if DAILY_REMINDER_CHAT_ID and context.job_queue:
                # Remove existing jobs
                for job in context.job_queue.get_jobs_by_name("daily_reminder"):
                    job.schedule_removal()
                
                # Add new job with updated time
                tz = timezone(timedelta(hours=DAILY_REMINDER_TIMEZONE_OFFSET))
                reminder_time = dt_time(hour=DAILY_REMINDER_HOUR, minute=DAILY_REMINDER_MINUTE, second=0, tzinfo=tz)
                context.job_queue.run_daily(send_daily_reminder, time=reminder_time, name="daily_reminder")
            
            await update.message.reply_text(
                f"✅ Время напоминания установлено: {hour:02d}:{minute:02d}\n\n"
                f"{'Напоминание перезапланировано.' if DAILY_REMINDER_CHAT_ID else 'Используй /set_reminder on для включения.'}"
            )
        except ValueError:
            await update.message.reply_text(
                "❌ Неверный формат времени.\n\n"
                "Используй: `/set_reminder HH:MM`\n"
                "Пример: `/set_reminder 09:30`",
                parse_mode="Markdown"
            )
        return
    
    await update.message.reply_text(
        "❓ Неизвестная команда.\n\n"
        "Используй:\n"
        "`/set_reminder` - показать статус\n"
        "`/set_reminder HH:MM` - установить время\n"
        "`/set_reminder on` - включить\n"
        "`/set_reminder off` - отключить\n"
        "`/set_reminder test` - тест",
        parse_mode="Markdown"
    )


def main():
    """Запуск бота"""
    # Создаём приложение
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Регистрируем обработчики
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("set_system_prompt", cmd_set_system_prompt))
    app.add_handler(CommandHandler("temperature", cmd_temperature))
    app.add_handler(CommandHandler("set_temperature", cmd_set_temperature))
    app.add_handler(CommandHandler("max_tokens", cmd_max_tokens))
    app.add_handler(CommandHandler("set_max_tokens", cmd_set_max_tokens))
    app.add_handler(CommandHandler("compress_trigger", cmd_compress_trigger))
    app.add_handler(CommandHandler("set_compress_trigger", cmd_set_compress_trigger))
    # MCP команды
    app.add_handler(CommandHandler("mcp_tools", cmd_mcp_tools))
    app.add_handler(CommandHandler("mcp_call", cmd_mcp_call))
    app.add_handler(CommandHandler("mcp_status", cmd_mcp_status))
    app.add_handler(CommandHandler("set_reminder", cmd_set_reminder))
    # Mobile MCP команды
    app.add_handler(CommandHandler("mobile_start", cmd_mobile_start))
    app.add_handler(CommandHandler("mobile_stop", cmd_mobile_stop))
    app.add_handler(CommandHandler("mobile_status", cmd_mobile_status))
    app.add_handler(CommandHandler("mobile_tools", cmd_mobile_tools))
    app.add_handler(CommandHandler("mobile_tool", cmd_mobile_tool))
    app.add_handler(CommandHandler("mobile_devices", cmd_mobile_devices))
    app.add_handler(CommandHandler("mobile_use", cmd_mobile_use))
    app.add_handler(CommandHandler("mobile_call", cmd_mobile_call))
    app.add_handler(CommandHandler("tap", cmd_tap))
    app.add_handler(CommandHandler("screenshot", cmd_screenshot))
    # Emulator команды
    app.add_handler(CommandHandler("android_avds", cmd_android_avds))
    app.add_handler(CommandHandler("android_boot", cmd_android_boot))
    app.add_handler(CommandHandler("android_status", cmd_android_status))
    app.add_handler(CommandHandler("android_stop", cmd_android_stop))
    app.add_handler(CommandHandler("ios_devices", cmd_ios_devices))
    app.add_handler(CommandHandler("ios_boot", cmd_ios_boot))
    app.add_handler(CommandHandler("ios_open", cmd_ios_open))
    app.add_handler(CommandHandler("diag", cmd_diag))
    # KB / RAG commands
    app.add_handler(CommandHandler("kb_status", cmd_kb_status))
    app.add_handler(CommandHandler("kb_reindex", cmd_kb_reindex))
    app.add_handler(CommandHandler("kb_ask", cmd_kb_ask))
    app.add_handler(CommandHandler("kb_compare", cmd_kb_compare))
    app.add_handler(CommandHandler("kb_compare_filter", cmd_kb_compare_filter))
    app.add_handler(CommandHandler("kb_threshold", cmd_kb_threshold))
    app.add_handler(CommandHandler("kb_debug", cmd_kb_debug))
    app.add_handler(CommandHandler("kb_on", cmd_kb_on))
    app.add_handler(CommandHandler("kb_off", cmd_kb_off))
    app.add_handler(CommandHandler("kb_auto_on", cmd_kb_auto_on))
    app.add_handler(CommandHandler("kb_auto_off", cmd_kb_auto_off))
    # Help команда с RAG и Git интеграцией
    app.add_handler(CommandHandler("help", cmd_help))
    # Code Review команда
    app.add_handler(CommandHandler("review", cmd_review))
    # Support / Ticket команды
    app.add_handler(CommandHandler("support", cmd_support))
    app.add_handler(CommandHandler("support_on", cmd_support_on))
    app.add_handler(CommandHandler("support_off", cmd_support_off))
    app.add_handler(CommandHandler("ticket_new", cmd_ticket_new))
    app.add_handler(CommandHandler("ticket_list", cmd_ticket_list))
    app.add_handler(CommandHandler("ticket_view", cmd_ticket_view))
    app.add_handler(CommandHandler("ticket_select", cmd_ticket_select))
    app.add_handler(CommandHandler("ticket_close", cmd_ticket_close))
    app.add_handler(CommandHandler("tickets_all", cmd_tickets_all))
    # Git команды
    app.add_handler(CommandHandler("git_status", cmd_git_status))
    app.add_handler(CommandHandler("git_branch", cmd_git_branch))
    app.add_handler(CommandHandler("git_log", cmd_git_log))
    app.add_handler(CommandHandler("git_files", cmd_git_files))
    app.add_handler(CommandHandler("git_show", cmd_git_show))
    # Pipeline команды (MCP chaining: KudaGo → Calendar)
    app.add_handler(CommandHandler("pipeline", cmd_pipeline))
    app.add_handler(CommandHandler("pipeline_add", cmd_pipeline_add))
    app.add_handler(CommandHandler("pipeline_add_all", cmd_pipeline_add_all))
    app.add_handler(CommandHandler("pipeline_clear", cmd_pipeline_clear))
    app.add_handler(CommandHandler("pipeline_status", cmd_pipeline_status))
    app.add_handler(CommandHandler("pipeline_cities", cmd_pipeline_cities))
    app.add_handler(CommandHandler("pipeline_categories", cmd_pipeline_categories))
    app.add_handler(CallbackQueryHandler(handle_model_callback, pattern="^model_"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Настраиваем ежедневное напоминание
    if DAILY_REMINDER_CHAT_ID:
        job_queue = app.job_queue
        # Create timezone with offset
        tz = timezone(timedelta(hours=DAILY_REMINDER_TIMEZONE_OFFSET))
        reminder_time = dt_time(hour=DAILY_REMINDER_HOUR, minute=DAILY_REMINDER_MINUTE, second=0, tzinfo=tz)
        job_queue.run_daily(send_daily_reminder, time=reminder_time, name="daily_reminder")
        print(f"⏰ Daily reminder scheduled at {DAILY_REMINDER_HOUR}:{DAILY_REMINDER_MINUTE:02d} (UTC+{DAILY_REMINDER_TIMEZONE_OFFSET}) for chat {DAILY_REMINDER_CHAT_ID}")
    else:
        print("⚠️ Daily reminder disabled (DAILY_REMINDER_CHAT_ID not set)")
    
    # Запускаем
    print("🤖 Бот запущен!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

