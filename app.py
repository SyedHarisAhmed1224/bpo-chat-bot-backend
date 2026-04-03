from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import requests
import json

app = FastAPI()

client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)

class ChatRequest(BaseModel):
    message: str

messages = [
    {
        "role": "system",
        "content": """
You are a helpful call center agent.

When a user asks about their order status:
1. Ask them for their order number if they haven't provided it.
2. Once they provide the order number, call the get_order_status function.
3. Explain the returned result clearly.
"""
    }
]

def get_order_status(order_id):
    url = f"http://103.86.52.158:8989/get-order?orderNumber={order_id}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return json.dumps(response.json())
        return f"Backend error: {response.status_code}"
    except Exception as e:
        return f"Request failed: {str(e)}"


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

@app.post("/chat")
def chat(req: ChatRequest):
    global messages

    messages.append({"role": "user", "content": req.message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools
    )

    message = response.choices[0].message

    if message.tool_calls:
        tool_call = message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        result = get_order_status(args["order_id"])

        messages.append(message)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })

        final = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        reply = final.choices[0].message.content
    else:
        reply = message.content

    messages.append({"role": "assistant", "content": reply})

    return {"reply": reply}