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

if not YANDEX_FOLDER_ID or not YANDEX_AUTH or not TELEGRAM_BOT_TOKEN:
    raise ValueError("Не все переменные окружения установлены!")

yandex_sdk = YCloudML(folder_id=YANDEX_FOLDER_ID, auth=YANDEX_AUTH)

user_conversations: Dict[int, List[Dict[str, Any]]] = {}
user_states: Dict[int, Dict[str, Any]] = {}

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

SYSTEM_PROMPT = """Ты полезный ассистент, который работает по протоколу:

1. Когда пользователь задает новый вопрос, создай план из 4 шагов с вопросами для уточнения и выведи их все сразу пользователю.
   Формат:
   1. [вопрос]
   2. [вопрос]
   3. [вопрос]
   4. [вопрос]

2. Когда пользователь скажет "хватит", "стоп", "готово" или что-то подобное, заверши диалог.
   - Скажи "ДИАЛОГ ЗАВЕРШЕН. Готовлю финальный ответ."
   - Суммаризируй запрос и дай полный ответ используя format_response(tldr="...", response="...")

3. Когда получишь ответы на все 4 вопроса ИЛИ поймешь, что информации достаточно:
   - Скажи "ДИАЛОГ ЗАВЕРШЕН. Готовлю финальный ответ."
   - Суммаризируй запрос и дай полный ответ используя format_response(tldr="...", response="...")

4. Если пользователь не полностью ответил на все вопросы, задай те, которые остались неотвеченными.

ВАЖНО:
- Перечисляй все вопросы СРАЗУ
- НЕ ДЕЛИ ВОПРСОЫ НА РАЗНЫЕ СООБЩЕНИЯ
- Пользователь дожлен ответить на ВСЕ вопросы

ПРИМЕР:
пользователь: "Я хочу написать песню"

ассистент (ВСЕ В ОДНОМ СООБЩЕНИИ): "План:
1. В каком стиле вы хотите написать песню?
2. Есть ли у вас готовые рифмы или слова?
3. Используете ли вы синтетический звук или живые инструменты?
4. Какую идею вы закладываете в песню?"
"""

def get_user_state(user_id: int) -> Dict:
    if user_id not in user_states:
        reset_user_state(user_id)
    return user_states[user_id]

def reset_user_state(user_id: int):
    user_states[user_id] = {
        "plan": [],
        "current_question": 0,
        "answers": {},
        "original_request": "",
        "is_completed": False
    }

def get_user_conversation_history(user_id: int) -> List[Dict]:
    if user_id not in user_conversations:
        user_conversations[user_id] = [{"role": "system", "text": SYSTEM_PROMPT}]
    return user_conversations[user_id]

def extract_plan(text: str) -> Optional[List[str]]:
    patterns = [
        r'План:\s*\n((?:\d+[\.\)]\s*[^\n]+\n?)+)',
        r'1[\.\)]\s*([^\n]+)\n2[\.\)]\s*([^\n]+)\n3[\.\)]\s*([^\n]+)\n4[\.\)]\s*([^\n]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            if len(match.groups()) == 4:
                return [match.group(i).strip() for i in range(1, 5)]
            elif len(match.groups()) == 1:
                plan_text = match.group(1)
                steps = re.findall(r'\d+[\.\)]\s*([^\n]+)', plan_text)
                if len(steps) >= 4:
                    return [s.strip() for s in steps[:4]]
    
    lines = text.split('\n')
    plan = []
    for line in lines:
        match = re.match(r'^\d+[\.\)]\s*(.+)', line.strip())
        if match:
            plan.append(match.group(1).strip())
            if len(plan) == 4:
                return plan
    return None

def extract_json(text: str) -> Optional[Dict]:
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    try:
        return json.loads(text.strip())
    except:
        return None

async def handle_start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_conversations[user_id] = [{"role": "system", "text": SYSTEM_PROMPT}]
    reset_user_state(user_id)
    await update.message.reply_text("Привет! Задай вопрос, и я задам несколько уточняющих вопросов.")

async def handle_clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_conversations[user_id] = [{"role": "system", "text": SYSTEM_PROMPT}]
    reset_user_state(user_id)
    await update.message.reply_text("История очищена.")

async def handle_user_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_message = update.message.text
    
    if not user_message:
        return
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    state = get_user_state(user_id)
    history = get_user_conversation_history(user_id)
    
    if state["is_completed"]:
        await update.message.reply_text("Диалог завершен. Задай новый вопрос.")
        return
    
    history.append({"role": "user", "text": user_message})
    
    try:
        if not state["plan"]:
            state["original_request"] = user_message
            
            result = yandex_sdk.models.completions("yandexgpt").configure(temperature=0.5).run(history)
            
            response_text = ""
            for alt in result:
                if hasattr(alt, 'text'):
                    response_text = alt.text
                    break
                elif isinstance(alt, str):
                    response_text = alt
                    break
            
            plan = extract_plan(response_text)
            if plan and len(plan) >= 4:
                state["plan"] = plan[:4]
                state["current_question"] = 0
                
                # Выводим все вопросы сразу одним сообщением
                questions_text = "План:\n"
                for i, question in enumerate(state['plan'], 1):
                    # Убираем нумерацию и форматирование, если есть
                    clean_question = re.sub(r'^\d+[\.\)]\s*', '', question.strip())
                    questions_text += f"{i}. {clean_question}\n"
                
                await update.message.reply_text(questions_text.strip())
                history.append({"role": "assistant", "text": questions_text.strip()})
                return
            else:
                # Если план не найден, показываем ответ модели
                await update.message.reply_text(response_text)
                history.append({"role": "assistant", "text": response_text})
                return
        
        # Используем модель для извлечения ответов на вопросы из сообщения пользователя
        extraction_prompt = f"""Вопросы плана:
{chr(10).join([f"{i+1}. {q}" for i, q in enumerate(state['plan'])])}

Текущие ответы:
{json.dumps(state['answers'], ensure_ascii=False, indent=2) if state['answers'] else 'Нет ответов'}

Сообщение пользователя: {user_message}

Проанализируй сообщение пользователя и определи, на какие вопросы из плана он ответил.
Верни JSON в формате: {{"q0": "ответ на вопрос 1", "q1": "ответ на вопрос 2", ...}}
Если пользователь не ответил на какой-то вопрос, не включай его в JSON.
Если пользователь сказал "хватит", "стоп", "готово" или подобное, верни {{"done": true}}"""
        
        extraction_history = history + [{"role": "system", "text": extraction_prompt}]
        extraction_result = yandex_sdk.models.completions("yandexgpt").configure(temperature=0.3).run(extraction_history)
        
        extraction_text = ""
        for alt in extraction_result:
            if hasattr(alt, 'text'):
                extraction_text = alt.text
                break
            elif isinstance(alt, str):
                extraction_text = alt
                break
        
        # Проверяем, не сказал ли пользователь "хватит" или подобное
        if "хватит" in user_message.lower() or "стоп" in user_message.lower() or "готово" in user_message.lower() or '"done"' in extraction_text.lower():
            state["is_completed"] = True
        else:
            # Извлекаем ответы из JSON
            extracted_answers = extract_json(extraction_text)
            if extracted_answers and "done" not in extracted_answers:
                # Обновляем ответы
                for key, value in extracted_answers.items():
                    if key.startswith("q") and key[1:].isdigit():
                        state["answers"][key] = value
        
        # Проверяем, все ли вопросы отвечены
        all_answered = all(f"q{i}" in state["answers"] for i in range(len(state["plan"])))
        
        if not all_answered and not state["is_completed"]:
            # Находим неотвеченные вопросы
            unanswered = []
            for i in range(len(state["plan"])):
                if f"q{i}" not in state["answers"]:
                    unanswered.append((i, state["plan"][i]))
            
            if unanswered:
                # Запрашиваем недостающие вопросы
                missing_questions = "Пожалуйста, ответьте на следующие вопросы:\n"
                for idx, question in unanswered:
                    clean_question = re.sub(r'^\d+[\.\)]\s*', '', question.strip())
                    missing_questions += f"{idx + 1}. {clean_question}\n"
                
                await update.message.reply_text(missing_questions.strip())
                history.append({"role": "assistant", "text": missing_questions.strip()})
                return
        
        # Если все вопросы отвечены или пользователь сказал "хватит"
        if all_answered or state["is_completed"]:
            state["is_completed"] = True
            
            summary_prompt = f"""Запрос: {state['original_request']}
Ответы: {json.dumps(state['answers'], ensure_ascii=False)}

Суммаризируй запрос и дай полный ответ. Используй format_response(tldr="...", response="...")"""
            
            summary_history = history + [{"role": "system", "text": summary_prompt}]
            result = yandex_sdk.models.completions("yandexgpt").configure(temperature=0.5).run(summary_history)
            
            summary_text = ""
            for alt in result:
                if hasattr(alt, 'text'):
                    summary_text = alt.text
                    break
                elif isinstance(alt, str):
                    summary_text = alt
                    break
            
            params = extract_json(summary_text)
            if not params:
                pattern = r'format_response\s*\([^)]*tldr\s*=\s*["\']([^"\']+)["\'][^)]*response\s*=\s*["\']([^"\']+)["\']'
                match = re.search(pattern, summary_text, re.DOTALL)
                if match:
                    params = {"tldr": match.group(1), "response": match.group(2)}
            
            if params and "tldr" in params and "response" in params:
                result_json = {"tldr": params["tldr"], "response": params["response"]}
                json_str = json.dumps(result_json, ensure_ascii=False, indent=2)
                msg = f"<pre>{json_str}</pre>\n\n📝 <b>TLDR:</b> {result_json['tldr']}\n\n💬 <b>Ответ:</b>\n{result_json['response']}"
                await update.message.reply_text(msg, parse_mode="HTML")
            else:
                await update.message.reply_text(summary_text)
    
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {str(e)}")
        if history and history[-1]["role"] == "user":
            history.pop()

def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", handle_start_command))
    app.add_handler(CommandHandler("clear", handle_clear_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_message))
    print("Бот запущен!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
