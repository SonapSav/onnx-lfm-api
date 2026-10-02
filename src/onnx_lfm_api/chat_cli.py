"""Interactive CLI chat client for a running onnx-lfm-api server.

Run:
    python -m onnx_lfm_api.chat_cli
    lfm-chat                                   # after `pip install -e .`
    lfm-chat --url http://192.168.0.125:8383/v1 --system "Be concise."

Config (flags override env; ./.env is loaded if present):
    LFM_URL       base URL   (default http://127.0.0.1:8383/v1)
    LFM_API_KEY   API key    (only needed if the server has auth enabled)

In-chat commands:  /reset   /system <text>   /exit
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from openai import OpenAI


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no hard dependency); does not overwrite real env."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


def main() -> None:
    _load_dotenv()
    ap = argparse.ArgumentParser(description="CLI chat for onnx-lfm-api")
    ap.add_argument("--url", default=os.environ.get("LFM_URL", "http://127.0.0.1:8383/v1"))
    ap.add_argument("--api-key", default=os.environ.get("LFM_API_KEY", ""))
    ap.add_argument("--model", default="lfm2.5")
    ap.add_argument("--system", default=None, help="optional system prompt")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--no-stream", action="store_true", help="wait for full reply")
    args = ap.parse_args()

    # OpenAI SDK requires a non-empty key even when the server ignores it.
    client = OpenAI(base_url=args.url, api_key=args.api_key or "no-auth")

    messages: list[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    print(f"onnx-lfm-api chat  |  {args.url}  |  model={args.model}")
    print("Type a message. Commands: /reset  /system <text>  /exit\n")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if not user:
            continue
        if user in ("/exit", "/quit"):
            print("bye")
            break
        if user == "/reset":
            messages = [m for m in messages if m["role"] == "system"]
            print("(history cleared)\n")
            continue
        if user.startswith("/system"):
            text = user[len("/system"):].strip()
            messages = [m for m in messages if m["role"] != "system"]
            if text:
                messages.insert(0, {"role": "system", "content": text})
            print("(system prompt updated)\n")
            continue

        messages.append({"role": "user", "content": user})
        kwargs: dict = {"model": args.model, "messages": messages, "max_tokens": args.max_tokens}
        if args.temperature is not None:
            kwargs["temperature"] = args.temperature

        print("bot> ", end="", flush=True)
        reply = ""
        try:
            if args.no_stream:
                resp = client.chat.completions.create(**kwargs)
                reply = resp.choices[0].message.content or ""
                print(reply)
            else:
                stream = client.chat.completions.create(stream=True, **kwargs)
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        print(delta, end="", flush=True)
                        reply += delta
                print()
        except Exception as e:  # noqa: BLE001 - surface any client/server error
            print(f"\n[error] {e}")
            messages.pop()  # drop the user turn so history stays consistent
            continue

        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
