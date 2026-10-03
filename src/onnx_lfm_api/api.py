from __future__ import annotations

import asyncio
import hmac
import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from .config import settings
from .generate import (
    GenParams,
    generate_collect,
    generate_stream,
    generate_text,
    generate_turn,
)
from .model import ModelBundle, load_model


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    max_tokens: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    stream: bool = False


class ChatResponse(BaseModel):
    reply: str


def require_api_key(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    """Gate protected endpoints on a matching key.

    Accepts either ``X-API-Key: <key>`` or ``Authorization: Bearer <key>`` (the
    latter is what OpenAI clients send). No-op when ``LFM_API_KEY`` is unset
    (auth disabled). Uses a constant-time compare to avoid timing leaks.
    """
    expected = settings.api_key
    if not expected:
        return
    provided = x_api_key
    if not provided and authorization and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _params(req: ChatRequest) -> GenParams:
    return GenParams(
        max_tokens=req.max_tokens or settings.max_tokens,
        temperature=settings.temperature if req.temperature is None else req.temperature,
        top_k=settings.top_k if req.top_k is None else req.top_k,
        repetition_penalty=(
            settings.repetition_penalty
            if req.repetition_penalty is None
            else req.repetition_penalty
        ),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model ONCE at startup; serialize generation with a lock since a
    # single ORT session + a blocking decode loop can't service requests
    # concurrently on CPU without thrashing.
    app.state.model = load_model()
    app.state.lock = asyncio.Lock()
    yield


app = FastAPI(title="onnx-lfm-api", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    bundle = getattr(app.state, "model", None)
    return {
        "status": "ok" if bundle is not None else "loading",
        "model": settings.model_repo,
        "quant": settings.quant,
        "providers": bundle.providers if bundle is not None else [],
        "io_binding": bundle.io_binding if bundle is not None else None,
        "prefix_cache": bundle.prefix_cache.stats()
        if bundle is not None and bundle.prefix_cache is not None else None,
    }


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
async def chat(req: ChatRequest):
    bundle: ModelBundle = app.state.model
    p = _params(req)
    msgs = [m.model_dump() for m in req.messages]
    async with app.state.lock:
        reply = await asyncio.to_thread(generate_text, bundle, msgs, p)
    return ChatResponse(reply=reply)


@app.post("/chat/stream", dependencies=[Depends(require_api_key)])
async def chat_stream(req: ChatRequest):
    bundle: ModelBundle = app.state.model
    p = _params(req)
    msgs = [m.model_dump() for m in req.messages]

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        sentinel = object()

        def produce():
            try:
                for delta in generate_stream(bundle, msgs, p):
                    loop.call_soon_threadsafe(queue.put_nowait, delta)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        async with app.state.lock:
            worker = asyncio.create_task(asyncio.to_thread(produce))
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                yield f"data: {json.dumps({'delta': item})}\n\n"
            await worker
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# --------------------------------------------------------------------------- #
# OpenAI-compatible surface: point any OpenAI client at base_url=".../v1".
# --------------------------------------------------------------------------- #


class OpenAIMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: str | None = None
    tool_calls: list[dict] | None = None  # assistant turns from a prior response
    tool_call_id: str | None = None  # on role="tool" results
    name: str | None = None


class OpenAIChatRequest(BaseModel):
    # Ignore the many OpenAI fields we don't implement rather than 422-ing.
    model_config = ConfigDict(extra="ignore")

    messages: list[OpenAIMessage]
    model: str | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False
    tools: list[dict] | None = None
    tool_choice: object = None  # accepted but not enforced


def _openai_params(req: OpenAIChatRequest) -> GenParams:
    max_t = req.max_tokens or req.max_completion_tokens or settings.max_tokens
    return GenParams(
        max_tokens=max_t,
        temperature=settings.temperature if req.temperature is None else req.temperature,
        top_k=settings.top_k,
        repetition_penalty=settings.repetition_penalty,
    )


def _py_literal(v: object) -> str:
    """Render a JSON value as the Python-ish literal LFM2 uses in tool calls."""
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, bool):
        return "True" if v else "False"
    if v is None:
        return "None"
    return json.dumps(v)


def _to_template_messages(messages: list[OpenAIMessage]) -> list[dict]:
    """Convert OpenAI-shaped messages (including assistant `tool_calls` and
    `tool` results) into the string-content form LFM2's chat template expects."""
    out: list[dict] = []
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            calls = []
            for tc in m.tool_calls:
                fn = (tc or {}).get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}
                kw = ", ".join(f"{k}={_py_literal(v)}" for k, v in args.items())
                calls.append(f"{fn.get('name', '')}({kw})")
            content = f"<|tool_call_start|>[{', '.join(calls)}]<|tool_call_end|>"
            if m.content:
                content += m.content
            out.append({"role": "assistant", "content": content})
        else:
            out.append({"role": m.role, "content": m.content or ""})
    return out


def _openai_tool_calls(calls: list[dict]) -> list[dict]:
    return [
        {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])},
        }
        for c in calls
    ]


@app.get("/v1/models", dependencies=[Depends(require_api_key)])
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": settings.model_repo,
                "object": "model",
                "created": 0,
                "owned_by": "liquidai",
            }
        ],
    }


@app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
async def chat_completions(req: OpenAIChatRequest):
    bundle: ModelBundle = app.state.model
    p = _openai_params(req)
    msgs = _to_template_messages(req.messages)
    model_name = req.model or settings.model_repo
    cid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def chunk(delta: dict, finish: str | None = None) -> str:
        payload = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    # --- Tool-aware path: generate fully, then shape (no incremental deltas) ---
    if req.tools:
        async with app.state.lock:
            result = await asyncio.to_thread(generate_turn, bundle, msgs, p, req.tools)
        message: dict = {"role": "assistant", "content": result["content"]}
        if result["tool_calls"]:
            message["tool_calls"] = _openai_tool_calls(result["tool_calls"])
        finish = result["finish_reason"]
        usage = {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        }

        if req.stream:
            async def tool_stream():
                yield chunk({"role": "assistant"})
                if message.get("tool_calls"):
                    yield chunk({"tool_calls": [
                        {"index": i, "id": tc["id"], "type": "function",
                         "function": tc["function"]}
                        for i, tc in enumerate(message["tool_calls"])
                    ]})
                elif result["content"]:
                    yield chunk({"content": result["content"]})
                yield chunk({}, finish=finish)
                yield "data: [DONE]\n\n"

            return StreamingResponse(tool_stream(), media_type="text/event-stream")

        return {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": usage,
        }

    # --- No tools: real token-by-token streaming / plain completion ---
    if req.stream:
        async def event_stream():
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()
            sentinel = object()

            def produce():
                try:
                    for delta in generate_stream(bundle, msgs, p):
                        loop.call_soon_threadsafe(queue.put_nowait, delta)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, sentinel)

            yield chunk({"role": "assistant"})
            async with app.state.lock:
                worker = asyncio.create_task(asyncio.to_thread(produce))
                while True:
                    item = await queue.get()
                    if item is sentinel:
                        break
                    yield chunk({"content": item})
                await worker
            yield chunk({}, finish="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    async with app.state.lock:
        text, prompt_tokens, completion_tokens = await asyncio.to_thread(
            generate_collect, bundle, msgs, p
        )
    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "length" if completion_tokens >= p.max_tokens else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
