from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import requests
import json
import os

app = FastAPI()

# CORS must be added right after app creation
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []

SYSTEM_PROMPT = """
You are a helpful call center agent.

When a user asks about their order status:
1. Ask them for their order number if they haven't provided it.
2. Once they provide the order number, call the get_order_status function.
3. Explain the returned result clearly.
"""

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Get order status",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"}
                },
                "required": ["order_id"]
            }
        }
    }
]

def get_order_status(order_id: str):
    url = f"http://103.86.52.158:8989/get-order?orderNumber={order_id}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return json.dumps(response.json())
        return json.dumps({"error": f"Backend error: {response.status_code}"})
    except Exception as e:
        return json.dumps({"error": f"Request failed: {str(e)}"})

@app.post("/chat")
def chat(req: ChatRequest):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *req.history,
        {"role": "user", "content": req.message},
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message

        if message.tool_calls:
            tool_call = message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)

            result = get_order_status(args["order_id"])

            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                ]
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

            final = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )

            return {"reply": final.choices[0].message.content or "No response generated."}

        return {"reply": message.content or "No response generated."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))