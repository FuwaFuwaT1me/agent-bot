#!/usr/bin/env python3
"""
Простой Telegram-бот на базе YandexGPT и DeepSeek.
Отвечает на вопросы пользователей.
"""

import os
import time
import json
import httpx
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI
from yandex_cloud_ml_sdk import YCloudML
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from history_compressor import check_and_compress_history
from local_storage import get_combined_summary, clear_summaries, get_summary_count

# Загружаем переменные окружения
load_dotenv()

YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
YANDEX_AUTH = os.getenv("YANDEX_AUTH")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

if not YANDEX_FOLDER_ID or not YANDEX_AUTH or not TELEGRAM_BOT_TOKEN:
    raise ValueError("Установите YANDEX_FOLDER_ID, YANDEX_AUTH и TELEGRAM_BOT_TOKEN в .env файле!")

# MCP Server URL (Kotlin MCP Server)
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8080/mcp")

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


# Глобальный MCP клиент
mcp_client = McpClient(MCP_SERVER_URL)


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


def ask_agent(user_id: int, question: str) -> AgentResponse:
    """Отправляет вопрос агенту и получает ответ с метриками."""
    history = get_history(user_id)
    model = get_model(user_id)
    temperature = get_temperature(user_id)
    max_tokens = get_max_tokens(user_id)
    
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
        model=MODELS[model]
    )


# === ОБРАБОТЧИКИ КОМАНД ===

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    user_id = update.effective_user.id
    clear_history(user_id)
    
    summary_count = get_summary_count(user_id)
    summary_info = f"\n📦 Загружено суммаризаций из памяти: {summary_count}" if summary_count > 0 else ""
    
    await update.message.reply_text(
        f"👋 Привет! Я простой бот-ассистент.{summary_info}\n\n"
        "Просто напиши мне вопрос, и я отвечу.\n\n"
        "Команды:\n"
        "/model - выбрать модель (YandexGPT / DeepSeek)\n"
        "/clear - очистить историю (суммаризации сохраняются)\n"
        "/clear all - полная очистка включая суммаризации\n"
        "/set_system_prompt <текст> - изменить системный промпт\n"
        "/temperature - показать текущую температуру\n"
        "/set_temperature <0-1> - изменить температуру\n"
        "/max_tokens - показать лимит токенов\n"
        "/set_max_tokens <число> - установить лимит токенов\n"
        "/compress_trigger - показать настройки сжатия истории\n"
        "/set_compress_trigger <число> - установить триггер сжатия (0 = отключить)\n\n"
        "🔧 MCP инструменты:\n"
        "/mcp_status - статус MCP сервера\n"
        "/mcp_tools - список доступных инструментов\n"
        "/mcp_call <tool> [args] - вызвать инструмент"
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
        "get_pokemon": "pikachu",
        "get_type": "fire",
        "get_move": "thunderbolt",
        "get_ability": "static",
        "list_pokemon": "10 0",
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
            "🔧 **Вызов MCP инструмента**\n\n"
            "Использование:\n"
            "`/mcp_call <tool_name> [value]`\n"
            "`/mcp_call <tool_name> {json}`\n\n"
            "Примеры:\n"
            "`/mcp_call get_pokemon pikachu`\n"
            "`/mcp_call get_type fire`\n"
            "`/mcp_call get_move thunderbolt`\n"
            "`/mcp_call list_pokemon 10 0`\n"
            '`/mcp_call get_pokemon {"name": "charizard"}`\n\n'
            "Используй /mcp_tools для списка доступных инструментов.",
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
            # Простой формат: автоматически определяем параметр
            # Для большинства инструментов это "name", для list_pokemon - limit/offset
            if tool_name == "list_pokemon":
                # /mcp_call list_pokemon [limit] [offset]
                parts = args_str.split()
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


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений"""
    user_id = update.effective_user.id
    user_message = update.message.text
    
    if not user_message:
        return
    
    # Показываем, что бот "печатает"
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        # Получаем ответ от агента
        response = ask_agent(user_id, user_message)
        
        # Формируем сообщение с метриками
        stats = (
            f"\n\n---\n"
            f"🤖 {response.model} | ⏱ {response.time_seconds:.2f}s | 💰 {response.cost_rub:.4f}₽\n"
            f"💬 Your message: {response.message_tokens} tokens\n"
            f"📥 Context (history): {response.input_tokens} tokens\n"
            f"📤 Response: {response.output_tokens} tokens\n"
            f"📊 This request total: {response.total_tokens} tokens"
        )
        
        await update.message.reply_text(response.text + stats)
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")


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
    app.add_handler(CallbackQueryHandler(handle_model_callback, pattern="^model_"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Запускаем
    print("🤖 Бот запущен!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

