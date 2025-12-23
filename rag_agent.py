#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

def _import_rag_helpers():
    """
    Lazy import so the script can print a helpful message when optional deps
    (numpy / sentence-transformers) are missing.
    """
    try:
        from tools.rag_ask import build_context, embed_query, load_index, topk_cosine
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "RAG dependencies are not installed. Activate venv and install requirements:\n"
            "  source new-env/bin/activate\n"
            "  python3 -m pip install -r requirements_rag.txt\n"
            f"Missing module: {e.name}"
        ) from e
    return load_index, embed_query, topk_cosine, build_context


def _env_is_configured_for_yandex() -> bool:
    return bool(os.getenv("YANDEX_FOLDER_ID") and os.getenv("YANDEX_AUTH"))


def _call_yandex(messages: list[dict[str, str]], *, temperature: float, max_tokens: int, model: str) -> Optional[str]:
    folder_id = os.getenv("YANDEX_FOLDER_ID")
    auth = os.getenv("YANDEX_AUTH")
    if not folder_id or not auth:
        return None
    try:
        from yandex_cloud_ml_sdk import YCloudML
    except Exception:
        return None

    yandex = YCloudML(folder_id=folder_id, auth=auth)
    result = yandex.models.completions(model).configure(temperature=temperature, max_tokens=max_tokens).run(messages)
    for alt in result:
        if hasattr(alt, "text"):
            return (alt.text or "").strip()
        if isinstance(alt, str):
            return alt.strip()
    return None


def answer_no_rag(question: str, *, temperature: float, max_tokens: int, model: str) -> Optional[str]:
    system = (
        "Ты полезный ассистент. Отвечай по делу, без выдуманных фактов. "
        "Если информации недостаточно — честно скажи об этом и попроси уточнение."
    )
    messages = [
        {"role": "system", "text": system},
        {"role": "user", "text": question},
    ]
    return _call_yandex(messages, temperature=temperature, max_tokens=max_tokens, model=model)


def answer_with_rag(question: str, context: str, *, temperature: float, max_tokens: int, model: str) -> Optional[str]:
    system = (
        "Ты помощник, использующий RAG. Отвечай, опираясь на КОНТЕКСТ. "
        "Если в контексте нет ответа — честно скажи, что в базе знаний этого нет, "
        "и не придумывай конкретику."
    )
    prompt = f"КОНТЕКСТ:\n{context}\n\nВОПРОС:\n{question}"
    messages = [
        {"role": "system", "text": system},
        {"role": "user", "text": prompt},
    ]
    return _call_yandex(messages, temperature=temperature, max_tokens=max_tokens, model=model)


@dataclass(frozen=True)
class CompareResult:
    answer_no_rag: str
    answer_rag: str
    conclusion: str
    judge_json: Optional[dict[str, Any]] = None


def _try_judge(
    *,
    question: str,
    context: str,
    no_rag_answer: str,
    rag_answer: str,
    temperature: float,
    max_tokens: int,
    model: str,
) -> Optional[dict[str, Any]]:
    system = (
        "Ты строгий reviewer качества ответов LLM. Сравни два ответа на один и тот же вопрос. "
        "Если дан КОНТЕКСТ — он является источником фактов для оценки. "
        "Верни СТРОГО валидный JSON без пояснений вокруг."
    )
    user = f"""ВОПРОС:
{question}

КОНТЕКСТ (RAG):
{context}

ОТВЕТ A (без RAG):
{no_rag_answer}

ОТВЕТ B (с RAG):
{rag_answer}

Сформируй JSON со схемой:
{{
  "winner": "A" | "B" | "tie",
  "where_rag_helped": [string, ...],
  "where_rag_hurt": [string, ...],
  "factuality_notes": [string, ...],
  "confidence": number
}}
"""
    messages = [{"role": "system", "text": system}, {"role": "user", "text": user}]
    text = _call_yandex(messages, temperature=temperature, max_tokens=max_tokens, model=model)
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _heuristic_conclusion(*, question: str, context: str, no_rag_answer: str, rag_answer: str) -> str:
    # Safe fallback when judge is unavailable.
    q = question.lower()
    ctx = context.lower()
    a = no_rag_answer.lower()
    b = rag_answer.lower()

    signals = [
        "подписк",
        "1 марта 2024",
        "ежемесячн",
        "ежегодн",
        "предоплат",
    ]
    signal_present_in_context = any(s in ctx for s in signals) or ("2024" in ctx and "подпис" in ctx)

    if signal_present_in_context:
        a_has = sum(1 for s in signals if s in a)
        b_has = sum(1 for s in signals if s in b)
        if b_has > a_has:
            return "Похоже, RAG помог: ответ с RAG использует факты из базы знаний более явно, чем ответ без RAG."
        if a_has > b_has:
            return "Похоже, RAG не помог: ответ без RAG выглядит конкретнее по ключевым фактам, чем ответ с RAG."
        return "Судя по ключевым признакам, ответы близки; явной пользы RAG не видно (возможно, вопрос не требует базы знаний)."

    # If context doesn't contain obvious anchors, RAG advantage is unclear.
    if len(context.strip()) == 0:
        return "RAG-контекст пустой: сравнение некорректно — нужно построить индекс/увеличить top-k."
    return "По этому вопросу контекст не содержит явных фактов, поэтому RAG может не давать преимущества; ориентируйтесь на полноту/точность ответов."


def compare(
    *,
    question: str,
    context: str,
    temperature: float,
    max_tokens: int,
    model: str,
    judge: bool,
) -> Optional[CompareResult]:
    no_rag = answer_no_rag(question, temperature=temperature, max_tokens=max_tokens, model=model)
    rag = answer_with_rag(question, context, temperature=temperature, max_tokens=max_tokens, model=model)
    if no_rag is None or rag is None:
        return None

    judge_json = None
    conclusion = ""
    if judge:
        judge_json = _try_judge(
            question=question,
            context=context,
            no_rag_answer=no_rag,
            rag_answer=rag,
            temperature=min(0.2, temperature),
            max_tokens=min(900, max_tokens),
            model=model,
        )
        if judge_json:
            winner = judge_json.get("winner")
            helped = judge_json.get("where_rag_helped") or []
            hurt = judge_json.get("where_rag_hurt") or []
            conclusion = f"Winner: {winner}. RAG helped: {len(helped)} points. RAG hurt: {len(hurt)} points."
        else:
            conclusion = _heuristic_conclusion(question=question, context=context, no_rag_answer=no_rag, rag_answer=rag)
    else:
        conclusion = _heuristic_conclusion(question=question, context=context, no_rag_answer=no_rag, rag_answer=rag)

    return CompareResult(answer_no_rag=no_rag, answer_rag=rag, conclusion=conclusion, judge_json=judge_json)


def _default_index_path() -> str:
    # Prefer the dedicated KB sqlite if present; otherwise fall back to the directory index.
    kb_sqlite = Path("doc_index/knowledge_base.sqlite")
    if kb_sqlite.exists():
        return str(kb_sqlite)
    return "doc_index"


def main() -> None:
    ap = argparse.ArgumentParser(description="Agent with 2 modes: with RAG / without RAG + comparison")
    ap.add_argument("--question", required=True, help="User question")
    ap.add_argument("--mode", choices=["rag", "no-rag", "compare"], default="compare")
    ap.add_argument("--index", default=_default_index_path(), help="Index path: directory, .json or .sqlite/.db")
    ap.add_argument("--top-k", type=int, default=5, help="How many chunks to retrieve")
    ap.add_argument("--max-context-chars", type=int, default=6000, help="Max context size (chars)")
    ap.add_argument("--model", default="yandexgpt", help="YandexGPT model name for yandex-cloud-ml-sdk")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--no-judge", action="store_true", help="Disable LLM-judge in compare mode")
    ap.add_argument("--hide-context", action="store_true", help="Do not print retrieved context")
    args = ap.parse_args()

    load_index, embed_query, topk_cosine, build_context = _import_rag_helpers()

    index_path = Path(args.index).expanduser().resolve()
    meta, chunks, emb = load_index(index_path)

    q_vec = embed_query(args.question, model_name=meta["model"])
    idxs = topk_cosine(q_vec, emb, k=args.top_k)
    context = build_context(chunks, emb, q_vec, idxs, max_chars=args.max_context_chars)

    if not args.hide_context:
        print("=== Retrieved context ===")
        print(context if context else "(empty)")
        print()

    if not _env_is_configured_for_yandex():
        print("LLM not configured: set YANDEX_FOLDER_ID + YANDEX_AUTH (and install yandex-cloud-ml-sdk).")
        if args.mode == "no-rag":
            print("=== Answer (no RAG) ===")
            print("(skipped)")
        elif args.mode == "rag":
            print("=== Answer (RAG) ===")
            print("(skipped)")
        else:
            print("=== Answer A (no RAG) ===")
            print("(skipped)")
            print()
            print("=== Answer B (RAG) ===")
            print("(skipped)")
            print()
            print("=== Conclusion ===")
            if args.hide_context:
                print("Сравнение ответов модели невозможно без настройки LLM; retrieval работает (контекст скрыт флагом --hide-context).")
            else:
                print("Сравнение ответов модели невозможно без настройки LLM; retrieval работает, контекст показан выше.")
        return

    if args.mode == "no-rag":
        ans = answer_no_rag(args.question, temperature=args.temperature, max_tokens=args.max_tokens, model=args.model)
        print("=== Answer (no RAG) ===")
        print(ans if ans is not None else "(no answer)")
        return

    if args.mode == "rag":
        ans = answer_with_rag(args.question, context, temperature=args.temperature, max_tokens=args.max_tokens, model=args.model)
        print("=== Answer (RAG) ===")
        print(ans if ans is not None else "(no answer)")
        return

    res = compare(
        question=args.question,
        context=context,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        model=args.model,
        judge=not args.no_judge,
    )
    if res is None:
        print("=== Compare ===")
        print("Не удалось получить один из ответов (проверьте Yandex env / сеть / лимиты).")
        return

    print("=== Answer A (no RAG) ===")
    print(res.answer_no_rag)
    print()
    print("=== Answer B (RAG) ===")
    print(res.answer_rag)
    print()
    print("=== Conclusion ===")
    print(res.conclusion)
    if res.judge_json:
        print()
        print("=== Judge JSON ===")
        print(json.dumps(res.judge_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


