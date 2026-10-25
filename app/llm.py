"""Thin wrapper so the rest of the code doesn't care which LLM provider is used.

Also makes provider limits visible instead of silent:
  - rate-limit errors (429) are retried with backoff
  - a reply cut off by the output-token limit prints a warning (a cut-off JSON reply would
    otherwise parse as "no results" and quietly lower the score)
"""
import os
import sys
import time

MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "16000"))
TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT", "180"))  # one hung request can't freeze a run
RETRIES = 4


def _warn(msg: str) -> None:
    print(f"  [warning] {msg}", file=sys.stderr, flush=True)


def _is_rate_limit(e: Exception) -> bool:
    return getattr(e, "status_code", None) == 429 or "rate limit" in str(e).lower() or "429" in str(e)


def _is_timeout(e: Exception) -> bool:
    return "timeout" in type(e).__name__.lower() or "timed out" in str(e).lower()


def _with_retries(call):
    for attempt in range(RETRIES + 1):
        try:
            return call()
        except Exception as e:
            if attempt < RETRIES and _is_rate_limit(e):
                wait = 2 ** attempt * 5  # 5, 10, 20, 40 s
                _warn(f"rate limit hit, retrying in {wait}s ({attempt + 1}/{RETRIES})")
                time.sleep(wait)
                continue
            if attempt < 1 and _is_timeout(e):  # one retry after a timeout
                _warn(f"request took over {TIMEOUT_SECONDS:.0f}s, retrying once")
                continue
            raise


def complete(system: str, user: str) -> str:
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    model = os.getenv("LLM_MODEL")

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(timeout=TIMEOUT_SECONDS, max_retries=0)
        resp = _with_retries(lambda: client.messages.create(
            model=model or "claude-sonnet-5-5",
            max_tokens=MAX_OUTPUT_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        ))
        if getattr(resp, "stop_reason", None) == "max_tokens":
            _warn("reply was cut off by the output-token limit; some results from this chunk are lost")
        return "".join(b.text for b in resp.content if b.type == "text")

    if provider in ("openai", "xai", "grok"):
        from openai import OpenAI

        if not model:
            raise RuntimeError(f"Set LLM_MODEL in .env for the {provider} provider")
        if provider in ("xai", "grok"):
            # xAI's API is OpenAI-compatible; only the key and base URL differ
            key = os.getenv("XAI_API_KEY")
            if not key:
                raise RuntimeError("Set XAI_API_KEY in .env")
            client = OpenAI(api_key=key, base_url=os.getenv("LLM_BASE_URL", "https://api.x.ai/v1"),
                            timeout=TIMEOUT_SECONDS, max_retries=0)
        else:
            client = OpenAI(timeout=TIMEOUT_SECONDS, max_retries=0)
        resp = _with_retries(lambda: client.chat.completions.create(
            model=model,
            max_tokens=MAX_OUTPUT_TOKENS,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        ))
        choice = resp.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            _warn("reply was cut off by the output-token limit; some results from this chunk are lost")
        return choice.message.content or ""

    raise RuntimeError(f"Unknown LLM_PROVIDER: {provider}")
