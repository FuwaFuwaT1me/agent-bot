#!/usr/bin/env python3
"""
Простой Telegram-бот на базе YandexGPT и DeepSeek.
Отвечает на вопросы пользователей.
"""

import os
import time
from typing import Dict, List
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI
from yandex_cloud_ml_sdk import YCloudML
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes

# Загружаем переменные окружения
load_dotenv()

YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
YANDEX_AUTH = os.getenv("YANDEX_AUTH")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

if not YANDEX_FOLDER_ID or not YANDEX_AUTH or not TELEGRAM_BOT_TOKEN:
    raise ValueError("Установите YANDEX_FOLDER_ID, YANDEX_AUTH и TELEGRAM_BOT_TOKEN в .env файле!")

# === 1. СОЗДАНИЕ SDK КЛИЕНТОВ ===
# YandexGPT
yandex_sdk = YCloudML(folder_id=YANDEX_FOLDER_ID, auth=YANDEX_AUTH)

# HuggingFace (DeepSeek)
hf_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN or ""
)

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

def get_history(user_id: int) -> List[dict]:
    """Получает историю для пользователя. Создаёт новую, если её нет."""
    if user_id not in user_histories:
        user_histories[user_id] = [{"role": "system", "text": SYSTEM_PROMPT}]
    return user_histories[user_id]


def clear_history(user_id: int):
    """Очищает историю пользователя."""
    user_histories[user_id] = [{"role": "system", "text": SYSTEM_PROMPT}]
    user_prev_input_tokens[user_id] = 0  # Сбрасываем счётчик токенов

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
    await update.message.reply_text(
        "👋 Привет! Я простой бот-ассистент.\n\n"
        "Просто напиши мне вопрос, и я отвечу.\n\n"
        "Команды:\n"
        "/model - выбрать модель (YandexGPT / DeepSeek)\n"
        "/clear - очистить историю диалога\n"
        "/set_system_prompt <текст> - изменить системный промпт\n"
        "/temperature - показать текущую температуру\n"
        "/set_temperature <0-1> - изменить температуру\n"
        "/max_tokens - показать лимит токенов\n"
        "/set_max_tokens <число> - установить лимит токенов"
    )


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /clear"""
    user_id = update.effective_user.id
    clear_history(user_id)
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
    app.add_handler(CallbackQueryHandler(handle_model_callback, pattern="^model_"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Запускаем
    print("🤖 Бот запущен!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

