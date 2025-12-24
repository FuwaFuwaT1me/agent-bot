#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _extract_json_from_text(text: str) -> Any:
    """
    Accepts:
    - pure JSON file
    - text file with a preface (lines) and then a JSON value starting with '[' or '{'
    """
    s = text.lstrip()
    if s.startswith("{") or s.startswith("["):
        return json.loads(s)

    # try to find first JSON array/object
    first_arr = text.find("[")
    first_obj = text.find("{")
    candidates = [p for p in [first_arr, first_obj] if p != -1]
    if not candidates:
        raise ValueError("No JSON object/array found in input.")
    start = min(candidates)
    return json.loads(text[start:])


def _status_ru(status: str) -> str:
    m = {
        "completed": "completed (закончил)",
        "planned": "planned (планирую)",
        "dropped": "dropped (бросил)",
        "watching": "watching (смотрю)",
        "on_hold": "on_hold (пауза)",
    }
    return m.get(status, status)


@dataclass(frozen=True)
class AnimeItem:
    title: str
    title_ru: Optional[str]
    target_id: Optional[int]
    target_type: Optional[str]
    score: Optional[float]
    status: Optional[str]
    rewatches: Optional[int]
    episodes: Optional[int]
    note: Optional[str]


def _coerce_item(obj: Dict[str, Any]) -> AnimeItem:
    def get_str(k: str) -> Optional[str]:
        v = obj.get(k)
        return v if isinstance(v, str) else None

    def get_int(k: str) -> Optional[int]:
        v = obj.get(k)
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, float) and v.is_integer():
            return int(v)
        return None

    def get_float(k: str) -> Optional[float]:
        v = obj.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        return None

    return AnimeItem(
        title=get_str("target_title") or get_str("title") or "Unknown",
        title_ru=get_str("target_title_ru") or get_str("title_ru"),
        target_id=get_int("target_id") or get_int("id"),
        target_type=get_str("target_type") or get_str("type"),
        score=get_float("score"),
        status=get_str("status"),
        rewatches=get_int("rewatches"),
        episodes=get_int("episodes"),
        note=get_str("text") or get_str("note") or get_str("comment"),
    )


def _summarize(items: List[AnimeItem]) -> Dict[str, Any]:
    by_status: Dict[str, int] = {}
    scored = 0
    for it in items:
        st = (it.status or "unknown").strip()
        by_status[st] = by_status.get(st, 0) + 1
        if it.score and it.score > 0:
            scored += 1

    top = sorted([it for it in items if (it.score or 0) > 0], key=lambda x: x.score or 0, reverse=True)[:15]
    planned = [it for it in items if (it.status or "").lower() == "planned"][:25]
    return {"by_status": by_status, "scored_count": scored, "top": top, "planned": planned}


def render_markdown(items: List[AnimeItem], *, title: str) -> str:
    s = _summarize(items)
    by_status: Dict[str, int] = s["by_status"]
    top: List[AnimeItem] = s["top"]
    planned: List[AnimeItem] = s["planned"]

    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("Этот документ сгенерирован из JSON и предназначен для RAG-поиска.")
    lines.append("")
    lines.append("## Сводка")
    lines.append(f"- Всего тайтлов: {len(items)}")
    lines.append(f"- Оценено (score > 0): {s['scored_count']}")
    if by_status:
        lines.append("- По статусам:")
        for k, v in sorted(by_status.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"  - {k}: {v}")
    lines.append("")

    if top:
        lines.append("## Топ по оценке (score)")
        for it in top:
            ru = f" / {it.title_ru}" if it.title_ru else ""
            lines.append(f"- {it.title}{ru} — score={it.score:g} status={_status_ru(it.status or 'unknown')}")
        lines.append("")

    if planned:
        lines.append("## В планах (planned)")
        for it in planned:
            ru = f" / {it.title_ru}" if it.title_ru else ""
            lines.append(f"- {it.title}{ru} — planned, episodes={it.episodes if it.episodes is not None else 'unknown'}")
        lines.append("")

    lines.append("## Полный список (для поиска)")
    for it in items:
        ru = f" / {it.title_ru}" if it.title_ru else ""
        meta = []
        if it.target_type:
            meta.append(f"type={it.target_type}")
        if it.target_id is not None:
            meta.append(f"id={it.target_id}")
        if it.status:
            meta.append(f"status={_status_ru(it.status)}")
        if it.score is not None:
            meta.append(f"score={it.score:g}")
        if it.episodes is not None:
            meta.append(f"episodes={it.episodes}")
        if it.rewatches is not None:
            meta.append(f"rewatches={it.rewatches}")
        meta_s = " | ".join(meta) if meta else "no-meta"

        lines.append(f"### {it.title}{ru}")
        lines.append(f"- {meta_s}")
        if it.note:
            lines.append(f"- note: {it.note}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert JSON/JSON-in-text into KB-friendly Markdown text for RAG")
    ap.add_argument("--input", required=True, help="Path to .json or text containing JSON")
    ap.add_argument("--output", required=True, help="Output .md/.txt file path")
    ap.add_argument("--title", default="Knowledge Base (normalized)", help="Document title")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    raw = in_path.read_text(encoding="utf-8", errors="replace")
    data = _extract_json_from_text(raw)

    if isinstance(data, list):
        objs = [x for x in data if isinstance(x, dict)]
        items = [_coerce_item(o) for o in objs]
    elif isinstance(data, dict):
        # Single dict => render as a list of one "item" with flattened keys in note.
        items = [_coerce_item(data)]
        if items[0].title == "Unknown":
            # try to create a stable pseudo-title
            items = [AnimeItem(title="JSON object", title_ru=None, target_id=None, target_type="Object", score=None, status=None, rewatches=None, episodes=None, note=json.dumps(data, ensure_ascii=False)[:4000])]
    else:
        raise SystemExit("Unsupported JSON root. Expected array or object.")

    md = render_markdown(items, title=args.title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"✓ Wrote: {out_path}  (items={len(items)})")


if __name__ == "__main__":
    main()


