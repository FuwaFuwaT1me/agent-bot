#!/usr/bin/env python3

from __future__ import annotations
import os
import json
import re
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

load_dotenv()

YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
YANDEX_AUTH = os.getenv("YANDEX_AUTH")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not YANDEX_FOLDER_ID:
    raise ValueError("YANDEX_FOLDER_ID не установлен! Добавьте его в файл .env")
if not YANDEX_AUTH:
    raise ValueError("YANDEX_AUTH не установлен! Добавьте его в файл .env")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN не установлен! Добавьте его в файл .env")

yandex_sdk = YCloudML(
    folder_id=YANDEX_FOLDER_ID,
    auth=YANDEX_AUTH,
)

user_conversations: Dict[int, List[Dict[str, Any]]] = {}

FORMAT_RESPONSE_TOOL = {
    "type": "function",
    "function": {
        "name": "format_response",
        "description": "Форматирует ответ пользователю в структурированном JSON формате",
        "parameters": {
            "type": "object",
            "properties": {
                "tldr": {
                    "type": "string",
                    "description": "Краткое резюме ответа в 1-2 предложениях"
                },
                "response": {
                    "type": "string",
                    "description": "Полный развернутый ответ на вопрос пользователя"
                }
            },
            "required": ["tldr", "response"]
        }
    }
}

def create_formatted_response(tldr: str, response: str) -> Dict[str, str]:
    return {
        "tldr": tldr,
        "response": response
    }

def get_user_conversation_history(user_id: int) -> List[Dict[str, Any]]:
    if user_id not in user_conversations:
        system_prompt = """Ты полезный ассистент. 
Когда пользователь задает вопрос, ты должен использовать функцию format_response 
для форматирования ответа. Всегда вызывай эту функцию с параметрами tldr и response."""
        
        user_conversations[user_id] = [
            {
                "role": "system",
                "text": system_prompt
            }
        ]
    
    return user_conversations[user_id]

def extract_function_parameters_from_text(text: str) -> Optional[Dict[str, str]]:
    text = re.sub(r'```[a-z]*\n?', '', text)
    text = re.sub(r'```', '', text)
    text = text.strip()
    
    pattern = r'format_response\s*\((.*?)\)'
    match = re.search(pattern, text, re.DOTALL)
    
    if not match:
        return None
    
    params_string = match.group(1).strip()
    
    def get_param_value(param_name: str, text: str) -> Optional[str]:
        patterns = [
            rf'{param_name}\s*=\s*"((?:[^"\\]|\\.)*)"',
            rf"{param_name}\s*=\s*'((?:[^'\\]|\\.)*)'",
            rf'{param_name}\s*=\s*"""((?:[^"]|"(?!""))*?)"""',
            rf"{param_name}\s*=\s*'''((?:[^']|'(?!''))*?)'''",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                value = match.group(1)
                value = value.replace('\\"', '"').replace("\\'", "'").replace('\\n', '\n')
                return value.strip()
        
        return None
    
    tldr_value = get_param_value('tldr', params_string)
    response_value = get_param_value('response', params_string)
    
    if tldr_value is not None and response_value is not None:
        return {
            'tldr': tldr_value,
            'response': response_value
        }
    
    return None

def execute_tool_function(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, str]:
    if tool_name == "format_response":
        return create_formatted_response(
            tldr=parameters.get("tldr", ""),
            response=parameters.get("response", "")
        )
    else:
        raise ValueError(f"Неизвестный инструмент: {tool_name}")

async def handle_start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    
    system_prompt = """Ты полезный ассистент. 
Когда пользователь задает вопрос, ты должен использовать функцию format_response 
для форматирования ответа. Всегда вызывай эту функцию с параметрами tldr и response."""
    
    user_conversations[user_id] = [
        {
            "role": "system",
            "text": system_prompt
        }
    ]
    
    welcome_message = (
        "Привет! Я бот с интеграцией Yandex GPT через Tools.\n\n"
        "Команды:\n"
        "/start - начать новый диалог\n"
        "/clear - очистить историю диалога"
    )
    await update.message.reply_text(welcome_message)

async def handle_clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    
    system_prompt = """Ты полезный ассистент. 
Когда пользователь задает вопрос, ты должен использовать функцию format_response 
для форматирования ответа. Всегда вызывай эту функцию с параметрами tldr и response."""
    
    user_conversations[user_id] = [
        {
            "role": "system",
            "text": system_prompt
        }
    ]
    
    await update.message.reply_text("История диалога очищена. Начинаем новый разговор!")

async def handle_user_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_message = update.message.text
    
    if not user_message:
        return
    
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )
    
    conversation_history = get_user_conversation_history(user_id)
    
    conversation_history.append({
        "role": "user",
        "text": user_message,
    })
    
    try:
        result = (
            yandex_sdk.models.completions("yandexgpt")
            .configure(temperature=0.5)
            .run(conversation_history)
        )
        
        assistant_response_text = ""
        for alternative in result:
            if hasattr(alternative, 'text'):
                assistant_response_text = alternative.text
                break
            elif isinstance(alternative, str):
                assistant_response_text = alternative
                break
        
        function_parameters = None
        
        try:
            parsed_json = json.loads(assistant_response_text)
            if "tldr" in parsed_json and "response" in parsed_json:
                function_parameters = parsed_json
        except (json.JSONDecodeError, KeyError):
            pass
        
        if not function_parameters:
            function_parameters = extract_function_parameters_from_text(assistant_response_text)
        
        if function_parameters and "tldr" in function_parameters and "response" in function_parameters:
            formatted_result = execute_tool_function(
                "format_response",
                function_parameters
            )
            
            conversation_history.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "format_response",
                        "arguments": json.dumps(function_parameters)
                    }
                }]
            })
            
            conversation_history.append({
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "format_response",
                "content": json.dumps(formatted_result)
            })
            
            json_string = json.dumps(formatted_result, ensure_ascii=False, indent=2)
            formatted_message = f"<pre>{json_string}</pre>\n\n"
            formatted_message += f"📝 <b>TLDR:</b> {formatted_result['tldr']}\n\n"
            formatted_message += f"💬 <b>Ответ:</b>\n{formatted_result['response']}"
            
            await update.message.reply_text(formatted_message, parse_mode="HTML")
            return
        
        warning_message = (
            f"⚠️ <b>Внимание:</b> Модель не вызвала функцию. "
            f"Ответ: {assistant_response_text[:4000]}"
        )
        await update.message.reply_text(warning_message, parse_mode="HTML")
        
        conversation_history.append({
            "role": "assistant",
            "text": assistant_response_text,
        })
        
    except Exception as error:
        await update.message.reply_text(
            f"Произошла ошибка при обработке вашего сообщения: {str(error)}"
        )
        if conversation_history and conversation_history[-1]["role"] == "user":
            conversation_history.pop()

def main() -> None:
    bot_application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    bot_application.add_handler(CommandHandler("start", handle_start_command))
    bot_application.add_handler(CommandHandler("clear", handle_clear_command))
    bot_application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_user_message
        )
    )
    
    print("Бот запущен и готов к работе!")
    bot_application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
