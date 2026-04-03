from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
import requests
import json
import os
import traceback

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

class ChatRequest(BaseModel):
    session_id: str
    message: str

SYSTEM_PROMPT = """
You are Alt Academy's student support assistant.

Behavior rules:
- Only introduce yourself if the user starts with a greeting (e.g., "hi", "hello", "hey").
- If the first message is a direct question or request, do NOT introduce yourself — answer immediately.
- Never introduce yourself in the middle of a conversation.
- Do not repeat your introduction in later turns.
- If the user says things like "hi", "hello", "how are you", "thanks", or "no thanks",
  respond naturally and briefly.
- If the user says "thank you" or "no thanks", close politely instead of restarting the conversation.
- Do not ask 'How can I help you today?' again and again unless the conversation is actually restarting.
- For subject questions, plan questions, billing, payment, access, or technical problems, use tools when needed.
- If required information is missing, ask only for the missing detail.
- Do not invent subjects, plans, pricing, payment details, or account status.

Tone:
- Warm
- Clear
- Professional
- Short and human
"""

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

def generate_ticket(name: str | None = None, email: str | None = None, category: str | None = None, subject: str | None = None, message: str | None = None):
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

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not set")

        print(req)

        history = get_session_messages(req.session_id)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": req.message},
        ]

        first = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        assistant_message = first.choices[0].message

        # save current user turn
        history.append({"role": "user", "content": req.message})

        if assistant_message.tool_calls:
            assistant_tool_message = {
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
            history.append(assistant_tool_message)

            tool_results_for_model = []
            for tc in assistant_message.tool_calls:
                fn_name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")

                if fn_name not in tool_map:
                    result = json.dumps({"error": f"Unknown tool: {fn_name}"})
                else:
                    result = tool_map[fn_name](**args)

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                }
                history.append(tool_msg)
                tool_results_for_model.append(tool_msg)

            final_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                *history
            ]

            final = client.chat.completions.create(
                model=model_name,
                messages=final_messages,
            )

            reply = final.choices[0].message.content or "No response generated."
            history.append({"role": "assistant", "content": reply})

            return {"reply": reply}

        reply = assistant_message.content or "No response generated."
        history.append({"role": "assistant", "content": reply})
        return {"reply": reply}

    except HTTPException:
        raise
    except Exception as e:
        print("BACKEND ERROR:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))