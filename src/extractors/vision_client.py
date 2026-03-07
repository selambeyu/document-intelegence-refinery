"""
Vision API client for page-image extraction via OpenRouter (or OpenAI-compatible) API.
Set OPENROUTER_API_KEY (or OPENAI_API_KEY) in the environment to enable.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from typing import Any

VISION_PROMPT = """Extract all text from this document page image. Return a single JSON object with this exact structure:
{"blocks": [{"text": "extracted text segment", "x0": 0, "y0": 0, "x1": 100, "y1": 100}]}
Use x0,y0,x1,y1 in points (approximate). One block per paragraph or logical segment. No other keys or text."""


def _encode_image(image_bytes: bytes, format: str = "png") -> str:
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    return f"data:image/{format};base64,{b64}"


def _chat_completion(
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    base_url: str = "https://openrouter.ai/api/v1",
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    body = {"model": model, "messages": messages}
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://doc-intle-refinery.local",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def extract_page_with_vision(
    image_bytes: bytes,
    page: int,
    api_key: str,
    model: str = "openai/gpt-4o-mini",
    cost_per_page: float = 0.02,
    base_url: str = "https://openrouter.ai/api/v1",
) -> tuple[list[dict[str, Any]], float]:
    """
    Send one page image to the vision API; return list of block dicts (text, x0, y0, x1, y1) and cost.
    """
    content = [
        {"type": "text", "text": VISION_PROMPT},
        {"type": "image_url", "image_url": {"url": _encode_image(image_bytes)}},
    ]
    messages = [{"role": "user", "content": content}]
    out = _chat_completion(api_key, model, messages, base_url)
    usage = out.get("usage", {}) or {}
    in_tok = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    out_tok = usage.get("output_tokens") or usage.get("completion_tokens") or 0
    cost = cost_per_page
    try:
        choice = out["choices"][0]
        raw = choice.get("message", {}).get("content", "{}")
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:].strip()
        obj = json.loads(raw)
        blocks = obj.get("blocks") or []
        for b in blocks:
            b.setdefault("page", page)
        return blocks, cost
    except (KeyError, IndexError, json.JSONDecodeError):
        return [], cost


def extract_document_with_vision(
    page_images: list[bytes],
    api_key: str,
    model: str = "openai/gpt-4o-mini",
    cost_per_page: float = 0.02,
    budget: float | None = None,
    base_url: str = "https://openrouter.ai/api/v1",
) -> tuple[list[dict[str, Any]], float]:
    """
    Extract all pages; stop if cumulative cost would exceed budget. Returns (all_blocks, total_cost).
    """
    all_blocks: list[dict[str, Any]] = []
    total_cost = 0.0
    for i, img in enumerate(page_images):
        if budget is not None and total_cost >= budget:
            break
        page_num = i + 1
        blocks, cost = extract_page_with_vision(
            img, page_num, api_key, model, cost_per_page, base_url
        )
        all_blocks.extend(blocks)
        total_cost += cost
    return all_blocks, total_cost


def get_vision_api_key() -> str | None:
    return os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
