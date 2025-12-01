#!/usr/bin/env python3

from __future__ import annotations
import os
from typing import Dict, List
from yandex_cloud_ml_sdk import YCloudML
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

user_conversations: Dict[int, List[Dict[str, str]]] = {}

YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
YANDEX_AUTH = os.getenv("YANDEX_AUTH")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

sdk = YCloudML(
    folder_id=YANDEX_FOLDER_ID,
    auth=YANDEX_AUTH,
)


def get_user_messages(user_id: int) -> List[Dict[str, str]]:
    """Получить историю сообщений пользователя"""
    if user_id not in user_conversations:
        # Инициализируем с системным сообщением
        user_conversations[user_id] = [
            {
                "role": "system",
                "text": "Ты полезный ассистент. Отвечай на вопросы пользователя вежливо и информативно.",
            }
        ]
    return user_conversations[user_id]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start"""
    user_id = update.effective_user.id
    user_conversations[user_id] = [
        {
            "role": "system",
            "text": "Ты полезный ассистент. Отвечай на вопросы пользователя вежливо и информативно.",
        }
    ]
    await update.message.reply_text(
        "Привет! Я бот с интеграцией Yandex GPT. "
        "Можешь задавать мне вопросы, и я буду отвечать.\n\n"
        "Команды:\n"
        "/start - начать новый диалог\n"
        "/clear - очистить историю диалога"
    )


async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /clear"""
    user_id = update.effective_user.id
    user_conversations[user_id] = [
        {
            "role": "system",
            "text": "Ты полезный ассистент. Отвечай на вопросы пользователя вежливо и информативно.",
        }
    ]
    await update.message.reply_text("История диалога очищена. Начинаем новый разговор!")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик текстовых сообщений"""
    user_id = update.effective_user.id
    user_message = update.message.text
    
    if not user_message:
        return
    
    # Показываем, что бот печатает
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    # Получаем историю диалога пользователя
    messages = get_user_messages(user_id)
    
    # Добавляем сообщение пользователя
    messages.append({
        "role": "user",
        "text": user_message,
    })
    
    try:
        # Отправляем запрос в Yandex GPT
        # Пробуем использовать chat, если не работает - используем completions
        try:
            result = (
                sdk.models.chat("yandexgpt")
                .configure(temperature=0.5)
                .run(messages)
            )
        except AttributeError:
            # Если метод chat не доступен, используем completions
            result = (
                sdk.models.completions("yandexgpt")
                .configure(temperature=0.5)
                .run(messages)
            )
        
        # Получаем ответ от модели
        assistant_response = ""
        for alternative in result:
            if hasattr(alternative, 'text'):
                assistant_response = alternative.text
                break
            elif hasattr(alternative, 'message') and hasattr(alternative.message, 'text'):
                assistant_response = alternative.message.text
                break
            elif isinstance(alternative, str):
                assistant_response = alternative
                break
            elif isinstance(alternative, dict):
                if 'text' in alternative:
                    assistant_response = alternative['text']
                    break
                elif 'message' in alternative and isinstance(alternative['message'], dict):
                    if 'text' in alternative['message']:
                        assistant_response = alternative['message']['text']
                        break
        
        if not assistant_response:
            # Если не удалось получить ответ в ожидаемом формате, попробуем другой способ
            assistant_response = str(result)
            if len(assistant_response) > 4000:
                assistant_response = assistant_response[:4000] + "..."
        
        # Добавляем ответ ассистента в историю
        messages.append({
            "role": "assistant",
            "text": assistant_response,
        })
        
        # Отправляем ответ пользователю
        await update.message.reply_text(assistant_response)
        
    except Exception as e:
        await update.message.reply_text(
            f"Произошла ошибка при обработке вашего сообщения: {str(e)}"
        )
        # Удаляем последнее сообщение пользователя из истории при ошибке
        if messages and messages[-1]["role"] == "user":
            messages.pop()


def main() -> None:
    """Запуск бота"""
    # Создаем приложение
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("clear", clear_history))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Запускаем бота
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
