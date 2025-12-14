"""Token and cost estimation utilities for Claude using tiktoken.

Pricing refs (Claude Sonnet 3.5):
- Input <=200K: $3 / M tokens, >200K: $6 / M
- Output <=200K: $15 / M tokens, >200K: $22.5 / M
- Prompt caching (write/read): not applied automatically; expose helpers.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None

ENCODING = "cl100k_base"


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if tiktoken is None:
        # Fallback rough estimate: 4 chars per token
        return max(1, math.ceil(len(text) / 4))
    enc = tiktoken.get_encoding(ENCODING)
    return len(enc.encode(text))


def _tier_cost_per_million(tokens: int, low: float, high: float) -> float:
    return low if tokens <= 200_000 else high


def estimate_prompt_cost(prompt: str) -> Dict[str, float]:
    tokens = count_tokens(prompt)
    rate = _tier_cost_per_million(tokens, 3.0, 6.0)
    cost = (tokens / 1_000_000.0) * rate
    return {"tokens": float(tokens), "rate_per_million": rate, "cost_usd": cost}


def estimate_response_cost(response_text: str) -> Dict[str, float]:
    tokens = count_tokens(response_text)
    rate = _tier_cost_per_million(tokens, 15.0, 22.5)
    cost = (tokens / 1_000_000.0) * rate
    return {"tokens": float(tokens), "rate_per_million": rate, "cost_usd": cost}


def estimate_prompt_caching_cost(tokens: int) -> Dict[str, float]:
    write_rate = _tier_cost_per_million(tokens, 3.75, 7.5)
    read_rate = _tier_cost_per_million(tokens, 0.30, 0.60)
    return {
        "tokens": float(tokens),
        "write_rate_per_million": write_rate,
        "read_rate_per_million": read_rate,
        "write_cost_usd": (tokens / 1_000_000.0) * write_rate,
        "read_cost_usd": (tokens / 1_000_000.0) * read_rate,
    }
