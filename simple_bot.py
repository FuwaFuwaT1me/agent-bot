#!/usr/bin/env python3
"""
Простой Telegram-бот на базе YandexGPT.
Отвечает на вопросы пользователей.
"""

import os
from typing import Dict, List
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Загружаем переменные окружения
load_dotenv()

YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
YANDEX_AUTH = os.getenv("YANDEX_AUTH")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not YANDEX_FOLDER_ID or not YANDEX_AUTH or not TELEGRAM_BOT_TOKEN:
    raise ValueError("Установите YANDEX_FOLDER_ID, YANDEX_AUTH и TELEGRAM_BOT_TOKEN в .env файле!")

# === 1. СОЗДАНИЕ SDK КЛИЕНТА ===
sdk = YCloudML(folder_id=YANDEX_FOLDER_ID, auth=YANDEX_AUTH)

# === 2. СИСТЕМНЫЙ ПРОМПТ ===
SYSTEM_PROMPT = """
"""

# === 3. ИСТОРИЯ СООБЩЕНИЙ ДЛЯ КАЖДОГО ПОЛЬЗОВАТЕЛЯ ===
# Ключ - ID пользователя в Telegram, значение - список сообщений
user_histories: Dict[int, List[dict]] = {}

# === 4. ТЕМПЕРАТУРА ДЛЯ КАЖДОГО ПОЛЬЗОВАТЕЛЯ ===
# 0 = строгие ответы, 1 = креативные ответы
user_temperatures: Dict[int, float] = {}

def get_history(user_id: int) -> List[dict]:
    """Получает историю для пользователя. Создаёт новую, если её нет."""
    if user_id not in user_histories:
        user_histories[user_id] = [{"role": "system", "text": SYSTEM_PROMPT}]
    return user_histories[user_id]


def clear_history(user_id: int):
    """Очищает историю пользователя."""
    user_histories[user_id] = [{"role": "system", "text": SYSTEM_PROMPT}]

def change_system_prompt(user_id: int, prompt: str):
    """Изменяет системный промпт для пользователя."""
    user_histories[user_id].append({"role": "system", "text": prompt})


def get_temperature(user_id: int) -> float:
    """Получает температуру для пользователя. По умолчанию 0.5."""
    return user_temperatures.get(user_id, 0.5)


def set_temperature(user_id: int, temp: float):
    """Устанавливает температуру для пользователя."""
    user_temperatures[user_id] = temp


def ask_agent(user_id: int, question: str) -> str:
    """Отправляет вопрос агенту и получает ответ."""
    history = get_history(user_id)
    
    # Добавляем вопрос в историю
    history.append({"role": "user", "text": question})

    print(history)
    print("--------------------------------")
    
    # Запрос к модели
    result = sdk.models.completions("yandexgpt").configure(
        temperature=get_temperature(user_id)
    ).run(history)
    
    # Извлекаем текст ответа
    response_text = ""
    for alt in result:
        if hasattr(alt, 'text'):
            response_text = alt.text
            break
        elif isinstance(alt, str):
            response_text = alt
            break
    
    # Добавляем ответ в историю
    history.append({"role": "assistant", "text": response_text})
    
    return response_text


# === ОБРАБОТЧИКИ КОМАНД ===

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    user_id = update.effective_user.id
    clear_history(user_id)
    await update.message.reply_text(
        "👋 Привет! Я простой бот-ассистент.\n\n"
        "Просто напиши мне вопрос, и я отвечу.\n\n"
        "Команды:\n"
        "/clear - очистить историю диалога\n"
        "/set_system_prompt <текст> - изменить системный промпт\n"
        "/temperature - показать текущую температуру\n"
        "/set_temperature <0-1> - изменить температуру"
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
        await update.message.reply_text(response)
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")


def main():
    """Запуск бота"""
    # Создаём приложение
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Регистрируем обработчики
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("set_system_prompt", cmd_set_system_prompt))
    app.add_handler(CommandHandler("temperature", cmd_temperature))
    app.add_handler(CommandHandler("set_temperature", cmd_set_temperature))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Запускаем
    print("🤖 Бот запущен!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

