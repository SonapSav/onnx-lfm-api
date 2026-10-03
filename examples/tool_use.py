"""End-to-end tool-calling example against a running onnx-lfm-api server.

The server only *decides* which tool to call; this client *executes* it and
feeds the result back. Any function you describe (schema) and implement can be
a tool.

Run (server must be up, e.g. `docker compose up -d`):
    python examples/tool_use.py
    python examples/tool_use.py "what time is it?"

Config via env (./.env is read if present):
    LFM_URL       base URL   (default http://127.0.0.1:8383/v1)
    LFM_API_KEY   API key    (only if the server has auth enabled)
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path

from openai import OpenAI


def _load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


# 1) DESCRIBE the tools (schemas sent to the model).
TOOLS = [
    {"type": "function", "function": {
        "name": "get_current_time",
        "description": "Return the current local date and time.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    }},
]


# 2) IMPLEMENT the tools (the server never runs these — your code does).
def get_current_time() -> dict:
    return {"now": datetime.datetime.now().isoformat(timespec="seconds")}


def get_weather(city: str) -> dict:
    # Stub — call a real weather API here.
    return {"city": city, "temp_c": 18, "conditions": "sunny"}


IMPL = {"get_current_time": get_current_time, "get_weather": get_weather}


def chat(client: OpenAI, user_text: str, max_rounds: int = 5) -> str:
    """model decides -> we execute -> feed result back -> repeat."""
    messages: list = [{"role": "user", "content": user_text}]
    for _ in range(max_rounds):
        resp = client.chat.completions.create(
            model="lfm2.5", messages=messages, tools=TOOLS, temperature=0
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or ""
        messages.append(msg)  # the assistant's tool-call turn
        for tc in msg.tool_calls:
            fn = IMPL.get(tc.function.name)
            args = json.loads(tc.function.arguments or "{}")
            print(f"  [tool] {tc.function.name}({args})")
            result = fn(**args) if fn else {"error": f"unknown tool {tc.function.name}"}
            print(f"  [ran ] -> {result}")
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(result)})
    return "(max rounds reached)"


def main() -> None:
    _load_dotenv()
    client = OpenAI(
        base_url=os.environ.get("LFM_URL", "http://127.0.0.1:8383/v1"),
        api_key=os.environ.get("LFM_API_KEY", "") or "no-auth",
    )
    questions = sys.argv[1:] or ["What time is it right now?", "What's the weather in Tokyo?"]
    for q in questions:
        print(f"\nUSER: {q}")
        print("ANSWER:", chat(client, q))


if __name__ == "__main__":
    main()
