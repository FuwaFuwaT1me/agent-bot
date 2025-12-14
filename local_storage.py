#!/usr/bin/env python3
"""
Модуль для сохранения и загрузки суммаризаций диалогов в локальное хранилище.
"""

import os
import json
from typing import Optional, List
from datetime import datetime

# Директория для хранения суммаризаций
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "summaries")


def ensure_storage_dir():
    """Создает директорию для хранения, если её нет."""
    if not os.path.exists(STORAGE_DIR):
        os.makedirs(STORAGE_DIR)


def get_user_file_path(user_id: int) -> str:
    """Возвращает путь к файлу суммаризаций пользователя."""
    ensure_storage_dir()
    return os.path.join(STORAGE_DIR, f"user_{user_id}.json")


def load_summaries(user_id: int) -> List[dict]:
    """
    Загружает все суммаризации пользователя из локального хранилища.
    
    Args:
        user_id: ID пользователя
    
    Returns:
        Список суммаризаций (каждая - dict с полями timestamp, text)
    """
    file_path = get_user_file_path(user_id)
    
    if not os.path.exists(file_path):
        return []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("summaries", [])
    except (json.JSONDecodeError, IOError) as e:
        print(f"Ошибка загрузки суммаризаций для пользователя {user_id}: {e}")
        return []


def save_summary(user_id: int, summary_text: str) -> bool:
    """
    Сохраняет новую суммаризацию в локальное хранилище.
    
    Args:
        user_id: ID пользователя
        summary_text: Текст суммаризации
    
    Returns:
        True если успешно, False если ошибка
    """
    file_path = get_user_file_path(user_id)
    
    # Загружаем существующие суммаризации
    summaries = load_summaries(user_id)
    
    # Добавляем новую суммаризацию
    summaries.append({
        "timestamp": datetime.now().isoformat(),
        "text": summary_text
    })
    
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"summaries": summaries}, f, ensure_ascii=False, indent=2)
        return True
    except IOError as e:
        print(f"Ошибка сохранения суммаризации для пользователя {user_id}: {e}")
        return False


def get_combined_summary(user_id: int) -> Optional[str]:
    """
    Возвращает объединенную суммаризацию всех предыдущих диалогов.
    
    Args:
        user_id: ID пользователя
    
    Returns:
        Объединенный текст суммаризаций или None если суммаризаций нет
    """
    summaries = load_summaries(user_id)
    
    if not summaries:
        return None
    
    # Объединяем все суммаризации в один текст
    combined_parts = []
    for i, summary in enumerate(summaries, 1):
        combined_parts.append(f"[Часть {i}]: {summary['text']}")
    
    return "\n\n".join(combined_parts)


def clear_summaries(user_id: int) -> bool:
    """
    Очищает все суммаризации пользователя.
    
    Args:
        user_id: ID пользователя
    
    Returns:
        True если успешно, False если ошибка
    """
    file_path = get_user_file_path(user_id)
    
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return True
        except IOError as e:
            print(f"Ошибка удаления суммаризаций для пользователя {user_id}: {e}")
            return False
    
    return True


def get_summary_count(user_id: int) -> int:
    """
    Возвращает количество сохраненных суммаризаций пользователя.
    
    Args:
        user_id: ID пользователя
    
    Returns:
        Количество суммаризаций
    """
    return len(load_summaries(user_id))

