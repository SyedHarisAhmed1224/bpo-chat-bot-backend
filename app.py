from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import requests
import json
import os
import traceback
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("OPENROUTER_API_KEY")
model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
API_BASE = os.getenv("ALT_API_BASE", "http://localhost:8585")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

conversation_store: dict[str, list[dict]] = {}

# Keep this small
MAX_HISTORY_MESSAGES = 6   # last 3 user/assistant pairs
MAX_MESSAGE_CHARS = 1200   # trim long messages before sending
MAX_COMPLETION_TOKENS = 800  # explicitly cap output tokens


class ChatRequest(BaseModel):
    session_id: str
    message: str


SYSTEM_PROMPT = """
You are Alt Academy's student support assistant.

Critical response rules:
- Never introduce yourself unless the user's message is only a greeting.
- If the user asks a question, requests information, or describes a problem, answer directly.
- Do not give a generic introduction before answering.
- Do not repeat that you are here to help unless it is actually useful.
- Never introduce yourself in the middle of a conversation.
- Never restart the conversation unless the user clearly starts over.

Behavior rules:
- If the user says only "hi", "hello", "hey", or similar, respond briefly and naturally.
- If the user says "thank you", "thanks", or "no thanks", respond briefly and close politely.
- For subject questions, plan questions, billing, payment, access, or technical problems, use tools when needed.
- If required information is missing, ask only for the missing detail.
- If the user uses foul language like swear words point it out and politely ask them to stay professional.
- Answer only question related to Alt Academy related, tasks like performing calculations can be answered but also question the user about the relevence to Alt Academy.
- Do not invent subjects, plans, pricing, payment details, or account status.

Tone:
- Warm
- Clear
- Professional
- Short and human
""".strip()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_subjects",
            "description": "Get Alt Academy subject catalog or search available subjects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                    "level": {"type": "string"},
                    "board": {"type": "string"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_plans",
            "description": "Get pricing or plan options for a subject or level.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "level": {"type": "string"},
                    "exam_session": {"type": "string"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_payments",
            "description": "Get a student's payment history or payment-related records.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string"},
                    "student_id": {"type": "string"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_payment",
            "description": "Get a specific payment using payment reference or payment ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "payment_reference": {"type": "string"},
                    "payment_id": {"type": "string"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_ticket",
            "description": "Create a support ticket for billing, access, or technical follow-up.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "category": {"type": "string"},
                    "subject": {"type": "string"},
                    "message": {"type": "string"}
                },
                "required": ["email", "category", "subject", "message"]
            }
        }
    }
]


def clamp_text(text: str, max_chars: int = MAX_MESSAGE_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def call_backend(method: str, path: str, params: dict | None = None, payload: dict | None = None):
    url = f"{API_BASE}{path}"
    try:
        if method.upper() == "GET":
            response = requests.get(url, params=params, timeout=15)
        elif method.upper() == "POST":
            response = requests.post(url, json=payload, timeout=15)
        else:
            return json.dumps({"error": f"Unsupported method: {method}"})

        if 200 <= response.status_code < 300:
            try:
                return json.dumps(response.json(), ensure_ascii=False)
            except Exception:
                return json.dumps({"raw": response.text}, ensure_ascii=False)

        return json.dumps({
            "error": f"Backend error: {response.status_code}",
            "body": response.text
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"Request failed: {str(e)}"}, ensure_ascii=False)


def get_subjects(keyword: str | None = None, level: str | None = None, board: str | None = None):
    params = {}
    if keyword:
        params["keyword"] = keyword
    if level:
        params["level"] = level
    if board:
        params["board"] = board
    return call_backend("GET", "/get-subjects", params=params)


def get_plans(subject: str | None = None, level: str | None = None, exam_session: str | None = None):
    params = {}
    if subject:
        params["subject"] = subject
    if level:
        params["level"] = level
    if exam_session:
        params["examSession"] = exam_session
    return call_backend("GET", "/get-plans", params=params)


def get_payments(email: str | None = None, student_id: str | None = None):
    params = {}
    if email:
        params["email"] = email
    if student_id:
        params["studentId"] = student_id
    return call_backend("GET", "/get-payments", params=params)


def get_payment(payment_reference: str | None = None, payment_id: str | None = None):
    params = {}
    if payment_reference:
        params["paymentReference"] = payment_reference
    if payment_id:
        params["paymentId"] = payment_id
    return call_backend("GET", "/get-payment", params=params)


def generate_ticket(
    name: str | None = None,
    email: str | None = None,
    category: str | None = None,
    subject: str | None = None,
    message: str | None = None
):
    payload = {
        "name": name,
        "email": email,
        "category": category,
        "subject": subject,
        "message": message
    }
    return call_backend("POST", "/generate-ticket", payload=payload)


tool_map = {
    "get_subjects": get_subjects,
    "get_plans": get_plans,
    "get_payments": get_payments,
    "get_payment": get_payment,
    "generate_ticket": generate_ticket,
}


def get_session_messages(session_id: str) -> list[dict]:
    if session_id not in conversation_store:
        conversation_store[session_id] = []
    return conversation_store[session_id]


def trim_history(history: list[dict]) -> list[dict]:
    filtered = []
    for msg in history:
        if msg.get("role") in ("user", "assistant"):
            filtered.append({
                "role": msg["role"],
                "content": clamp_text(msg.get("content", ""))
            })
    return filtered[-MAX_HISTORY_MESSAGES:]


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def is_greeting_only(text: str) -> bool:
    t = normalize_text(text)
    greeting_phrases = {
        "hi",
        "hello",
        "hey",
        "heya",
        "yo",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "thanks",
        "thank you",
        "no thanks",
    }
    return t in greeting_phrases


def build_turn_rule(history: list[dict], user_message: str) -> str:
    first_turn = len(history) == 0
    greeting_only = is_greeting_only(user_message)

    if first_turn and greeting_only:
        return (
            "This is the first user message and it is only a greeting or brief courtesy. "
            "Respond briefly and naturally. "
            "You may introduce yourself in one short sentence only if natural."
        )

    if first_turn and not greeting_only:
        return (
            "This is the first user message and it is a direct question, request, or issue. "
            "Do not introduce yourself. "
            "Do not greet. "
            "Do not give a generic welcome message. "
            "Answer immediately."
        )

    return (
        "This is not the first turn. "
        "Do not introduce yourself. "
        "Do not restart the conversation. "
        "Continue naturally and respond to the latest user message."
    )


def safe_json_loads(raw: str) -> dict:
    try:
        data = json.loads(raw or "{}")
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not set")

        history = get_session_messages(req.session_id)
        history[:] = trim_history(history)

        print('req.session_id -> ', req.session_id)

        user_message = clamp_text(req.message)
        turn_rule = build_turn_rule(history, user_message)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": turn_rule},
            *history,
            {"role": "user", "content": user_message},
        ]

        first = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=MAX_COMPLETION_TOKENS,
        )

        assistant_message = first.choices[0].message

        history.append({"role": "user", "content": user_message})
        history[:] = trim_history(history)

        if assistant_message.tool_calls:
            current_turn_messages = messages + [
                {
                    "role": "assistant",
                    "content": assistant_message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                }
            ]

            for tc in assistant_message.tool_calls:
                fn_name = tc.function.name
                args = safe_json_loads(tc.function.arguments)

                if fn_name not in tool_map:
                    result = json.dumps({"error": f"Unknown tool: {fn_name}"})
                else:
                    result = tool_map[fn_name](**args)

                # Also clamp tool results so huge payloads do not explode token usage
                current_turn_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": clamp_text(result, 2000)
                })

            final = client.chat.completions.create(
                model=model_name,
                messages=current_turn_messages,
                max_tokens=MAX_COMPLETION_TOKENS,
            )

            reply = final.choices[0].message.content or "No response generated."
            history.append({"role": "assistant", "content": clamp_text(reply)})
            history[:] = trim_history(history)
            return {"reply": reply}

        reply = assistant_message.content or "No response generated."
        history.append({"role": "assistant", "content": clamp_text(reply)})
        history[:] = trim_history(history)
        return {"reply": reply}

    except HTTPException:
        raise
    except Exception as e:
        print("BACKEND ERROR:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))