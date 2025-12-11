#!/usr/bin/env python3
"""
Модуль для сжатия истории диалога.
Анализирует историю и сжимает старые сообщения в краткий конспект.
"""

from typing import List, Dict, Tuple
from yandex_cloud_ml_sdk import YCloudML
from openai import OpenAI


# Промпт для агента-суммаризатора
SUMMARIZATION_PROMPT = """Ты — полезный ассистент для сжатия истории диалога. Твоя задача — создать КРАТКИЙ конспект диалога между пользователем и ассистентом. Извлеки ключевые факты, решения, предпочтения пользователя и контекст. Этот конспект будет использоваться для продолжения беседы вместо оригинальных сообщений, поэтому сохрани все важное для будущего контекста. Избегай мелких деталей, диалоговых оборотов. Ответь ТОЛЬКО текстом конспекта.

Диалог для сжатия:
{history_block}"""


def count_dialogue_messages(history: List[dict], start_idx: int = 0) -> int:
    """
    Подсчитывает количество сообщений (user и assistant) в истории.
    Игнорирует системные сообщения и summary-сообщения.
    
    Args:
        history: Список сообщений истории
        start_idx: Индекс, с которого начинать подсчёт
    
    Returns:
        Количество сообщений (user + assistant, каждое считается отдельно)
    """
    count = 0
    
    print(f"  count_dialogue_messages: start_idx={start_idx}, len(history)={len(history)}")
    
    for i in range(start_idx, len(history)):
        msg = history[i]
        role = msg.get("role", "")
        name = msg.get("name", "")
        
        # Пропускаем системные сообщения и summary
        if role == "system" or name == "summary":
            print(f"    Пропуск [{i}]: role={role}, name={name}")
            continue
        
        # Считаем каждое сообщение user или assistant отдельно
        if role == "user" or role == "assistant":
            count += 1
            print(f"    Считаем [{i}]: role={role}, count={count}")
    
    print(f"  Итого сообщений: {count}")
    return count


def find_compressible_block(history: List[dict], last_compressed_idx: int, trigger_turns: int) -> Tuple[int, int]:
    """
    Находит блок истории для сжатия.
    
    Args:
        history: Полная история сообщений
        last_compressed_idx: Индекс последнего сообщения после последнего сжатия
        trigger_turns: Количество сообщений (user + assistant), после которых нужно сжимать
    
    Returns:
        Tuple (start_idx, end_idx) - индексы начала и конца блока для сжатия.
        Если сжатие не требуется, возвращает (None, None)
    """
    # Начинаем поиск с индекса после последнего сжатия
    start_idx = last_compressed_idx + 1
    print(f"  find_compressible_block: last_compressed_idx={last_compressed_idx}, начальный start_idx={start_idx}")
    
    # Пропускаем системные сообщения и summary в начале блока
    original_start_idx = start_idx
    while start_idx < len(history):
        msg = history[start_idx]
        role = msg.get("role", "")
        name = msg.get("name", "")
        if role != "system" and name != "summary":
            break
        print(f"  Пропуск системного сообщения [{start_idx}]: role={role}, name={name}")
        start_idx += 1
    
    print(f"  После пропуска системных: start_idx={start_idx} (было {original_start_idx})")
    
    # Если после пропуска системных сообщений ничего не осталось
    if start_idx >= len(history):
        print(f"  Нет сообщений после пропуска системных")
        return None, None
    
    # Подсчитываем количество сообщений от start_idx
    message_count = count_dialogue_messages(history, start_idx)
    
    print(f"  Найдено сообщений: {message_count}, требуется: {trigger_turns}")
    
    if message_count < trigger_turns:
        return None, None
    
    # Находим конец блока для сжатия
    # Нужно найти последнее сообщение (user или assistant) в пределах trigger_turns
    target_messages = trigger_turns
    current_count = 0
    end_idx = start_idx
    
    for i in range(start_idx, len(history)):
        msg = history[i]
        role = msg.get("role", "")
        
        # Пропускаем системные сообщения и summary
        if role == "system" or msg.get("name") == "summary":
            continue
        
        # Считаем каждое сообщение user или assistant
        if role == "user" or role == "assistant":
            current_count += 1
            end_idx = i
            
            if current_count >= target_messages:
                break
    
    return start_idx, end_idx


def format_history_for_summarization(history: List[dict], start_idx: int, end_idx: int) -> str:
    """
    Форматирует блок истории для суммаризации.
    
    Args:
        history: Полная история сообщений
        start_idx: Начальный индекс блока
        end_idx: Конечный индекс блока (включительно)
    
    Returns:
        Отформатированная строка для промпта
    """
    lines = []
    for i in range(start_idx, end_idx + 1):
        msg = history[i]
        role = msg.get("role", "")
        content = msg.get("text", msg.get("content", ""))
        
        # Пропускаем системные сообщения и summary при форматировании
        if role == "system" or msg.get("name") == "summary":
            continue
        
        if role == "user":
            lines.append(f"Пользователь: {content}")
        elif role == "assistant":
            lines.append(f"Ассистент: {content}")
    
    return "\n".join(lines)


def summarize_with_yandex(
    yandex_sdk: YCloudML,
    history_block: str
) -> str:
    """
    Суммаризирует блок истории с помощью YandexGPT.
    
    Args:
        yandex_sdk: Клиент YandexGPT SDK
        history_block: Отформатированный блок истории
    
    Returns:
        Текст суммаризации
    """
    prompt = SUMMARIZATION_PROMPT.format(history_block=history_block)
    
    # Формируем историю для запроса
    messages = [
        {"role": "user", "text": prompt}
    ]
    
    model = yandex_sdk.models.completions("yandexgpt")
    result = model.configure(temperature=0.1, max_tokens=500).run(messages)
    
    response_text = ""
    for alt in result:
        if hasattr(alt, 'text'):
            response_text = alt.text
            break
    
    return response_text.strip()


def summarize_with_deepseek(
    hf_client: OpenAI,
    history_block: str
) -> str:
    """
    Суммаризирует блок истории с помощью DeepSeek.
    
    Args:
        hf_client: Клиент HuggingFace (OpenAI-совместимый)
        history_block: Отформатированный блок истории
    
    Returns:
        Текст суммаризации
    """
    prompt = SUMMARIZATION_PROMPT.format(history_block=history_block)
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    completion = hf_client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=messages,
        temperature=0.1,
        max_tokens=500
    )
    
    return completion.choices[0].message.content.strip()


def compress_history_block(
    history: List[dict],
    start_idx: int,
    end_idx: int,
    yandex_sdk: YCloudML = None,
    hf_client: OpenAI = None,
    model: str = "yandex"
) -> str:
    """
    Сжимает блок истории в краткий конспект.
    
    Args:
        history: Полная история сообщений
        start_idx: Начальный индекс блока для сжатия
        end_idx: Конечный индекс блока для сжатия
        yandex_sdk: Клиент YandexGPT SDK
        hf_client: Клиент HuggingFace (OpenAI-совместимый)
        model: Модель для суммаризации ("yandex" или "deepseek")
    
    Returns:
        Текст суммаризации
    """
    # Форматируем блок истории
    history_block = format_history_for_summarization(history, start_idx, end_idx)
    
    if not history_block.strip():
        return "Предыдущая часть диалога была пустой."
    
    # Вызываем соответствующую модель для суммаризации
    if model == "deepseek":
        if not hf_client:
            raise ValueError("hf_client не предоставлен для модели DeepSeek")
        return summarize_with_deepseek(hf_client, history_block)
    else:
        if not yandex_sdk:
            raise ValueError("yandex_sdk не предоставлен для модели YandexGPT")
        return summarize_with_yandex(yandex_sdk, history_block)


def check_and_compress_history(
    user_id: int,
    history: List[dict],
    last_compressed_idx: Dict[int, int],
    trigger_turns: Dict[int, int],
    yandex_sdk: YCloudML = None,
    hf_client: OpenAI = None,
    model: str = "yandex"
) -> bool:
    """
    Проверяет историю на необходимость сжатия и выполняет сжатие, если нужно.
    
    Args:
        user_id: ID пользователя
        history: История сообщений пользователя
        last_compressed_idx: Словарь с индексами последнего сжатия для каждого пользователя
        trigger_turns: Словарь с количеством сообщений (user + assistant) для триггера сжатия
        yandex_sdk: Клиент YandexGPT SDK
        hf_client: Клиент HuggingFace (OpenAI-совместимый)
        model: Модель для суммаризации ("yandex" или "deepseek")
    
    Returns:
        True, если было выполнено сжатие, False иначе
    """
    # Получаем параметры для пользователя
    # ВАЖНО: используем .get() с дефолтным значением -1 и сразу сохраняем в словарь
    # чтобы гарантировать, что ключ всегда существует
    last_idx = last_compressed_idx.get(user_id, -1)
    # Гарантируем, что ключ существует в словаре
    if user_id not in last_compressed_idx:
        last_compressed_idx[user_id] = last_idx
    
    trigger = trigger_turns.get(user_id, 10)  # По умолчанию 10 сообщений
    
    print(f"Проверка сжатия для user_id={user_id}: last_idx={last_idx}, trigger={trigger}")
    print(f"Размер истории: {len(history)}")
    print(f"Ключ user_id присутствует в словаре: {user_id in last_compressed_idx}")
    print(f"Значение в словаре: {last_compressed_idx.get(user_id, 'KEY_NOT_FOUND')}")
    if last_idx >= 0 and last_idx < len(history):
        print(f"Элемент на позиции last_idx: {history[last_idx]}")
    print("=======")
    # Если триггер = 0, сжатие отключено
    if trigger == 0:
        return False
    
    # Находим блок для сжатия
    start_idx, end_idx = find_compressible_block(history, last_idx, trigger)
    
    print(f"find_compressible_block вернул: start_idx={start_idx}, end_idx={end_idx}")
    
    if start_idx is None or end_idx is None:
        print(f"Сжатие не требуется: недостаточно сообщений для триггера {trigger}")
        return False
    
    # Выполняем сжатие
    try:
        summary_text = compress_history_block(
            history,
            start_idx,
            end_idx,
            yandex_sdk=yandex_sdk,
            hf_client=hf_client,
            model=model
        )
        
        # Создаём служебное сообщение с суммаризацией
        summary_msg = {
            "role": "system",
            "name": "summary",
            "text": f"Краткий конспект предыдущей части диалога: {summary_text}"
        }

        print(f"summary_msg: {summary_msg}")
        
        # Заменяем блок истории на суммаризацию
        # Удаляем старые сообщения и вставляем summary
        new_history = history[:start_idx] + [summary_msg] + history[end_idx + 1:]
        
        # Обновляем историю
        history.clear()
        history.extend(new_history)

        # Обновляем индекс последнего сжатия
        # После сжатия summary находится на позиции start_idx в новой истории
        # Это правильный индекс для отслеживания последнего сжатия
        # ВАЖНО: обновляем словарь напрямую, чтобы изменения сохранились
        old_value = last_compressed_idx.get(user_id, -999)
        last_compressed_idx[user_id] = start_idx
        new_value = last_compressed_idx[user_id]
        
        print(f"Обновление индекса: user_id={user_id}, старое значение={old_value}, новое значение={new_value}")
        print(f"Размер новой истории: {len(history)}")
        if start_idx < len(history):
            summary_role = history[start_idx].get("role", "?")
            summary_name = history[start_idx].get("name", "?")
            print(f"Элемент на позиции {start_idx}: role={summary_role}, name={summary_name}")
        print("=======")
        
        return True
    except Exception as e:
        # В случае ошибки просто пропускаем сжатие
        print(f"Ошибка при сжатии истории для пользователя {user_id}: {e}")
        return False

