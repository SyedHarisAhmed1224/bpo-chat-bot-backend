from openai import OpenAI
import requests
import json
import os

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

def get_order_status(order_id):

    url = f"http://103.86.52.158:8989/get-order?orderNumber={order_id}"

    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            return json.dumps(data)
        else:
            return f"Backend error: {response.status_code}"

    except Exception as e:
        return f"Request failed: {str(e)}"
    
print(get_order_status("10100"))

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Get the status of a customer's order using their order number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order number provided by the customer"
                    }
                },
                "required": ["order_id"]
            }
        }
    }
]

messages = [
    {
        "role": "system",
        "content": """
You are a helpful call center agent.

When a user asks about their order status:
1. Ask them for their order number if they haven't provided it.
2. Once they provide the order number, call the get_order_status function.
3. Explain the returned result to the user clearly.
"""
    }
]

while True:

    user_input = input("User: ")

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools
    )

    print(response)

    message = response.choices[0].message

    if message.tool_calls:

        tool_call = message.tool_calls[0]

        args = json.loads(tool_call.function.arguments)

        order_result = get_order_status(args["order_id"])

        messages.append(message)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": order_result
        })

        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        reply = final_response.choices[0].message.content

    else:
        reply = message.content

    print("Bot:", reply)

    messages.append({"role": "assistant", "content": reply})