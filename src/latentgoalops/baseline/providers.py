"""OpenAI-compatible provider adapter with retry and checkpoint-friendly errors."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI


GRADIENT_BASE_URL = "https://inference.do-ai.run/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL_PRICING = {
    "openai-gpt-oss-20b": {"input": 0.05 / 1_000_000, "output": 0.45 / 1_000_000},
    "openai-gpt-oss-120b": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "openai/gpt-oss-20b": {"input": 0.03 / 1_000_000, "output": 0.11 / 1_000_000},
    "openai/gpt-oss-120b": {"input": 0.09 / 1_000_000, "output": 0.35 / 1_000_000},
    "alibaba-qwen3-32b": {"input": 0.25 / 1_000_000, "output": 0.55 / 1_000_000},
    "deepseek-r1-distill-llama-70b": {"input": 0.99 / 1_000_000, "output": 0.99 / 1_000_000},
}
GRADIENT_MODEL_PRICING = {
    "openai-gpt-oss-20b": {"input": 0.05 / 1_000_000, "output": 0.45 / 1_000_000},
    "openai-gpt-oss-120b": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
}
OPENROUTER_MODEL_PRICING = {
    "openai/gpt-oss-20b": {"input": 0.03 / 1_000_000, "output": 0.11 / 1_000_000},
    "openai/gpt-oss-120b": {"input": 0.09 / 1_000_000, "output": 0.35 / 1_000_000},
}
OPENROUTER_MODEL_ALIASES = {
    "openai-gpt-oss-20b": "openai/gpt-oss-20b",
    "openai-gpt-oss-120b": "openai/gpt-oss-120b",
}
GRADIENT_MODEL_ALIASES = {value: key for key, value in OPENROUTER_MODEL_ALIASES.items()}
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0


class ProviderExhaustedError(RuntimeError):
    """Raised when billing or account limits stop the run."""


class ProviderCredentialError(RuntimeError):
    """Raised when the API token is invalid or revoked."""


@dataclass(slots=True)
class ProviderResponse:
    """Structured provider response."""

    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    raw: dict | None = None


def _is_gradient_endpoint(base_url: str) -> bool:
    """Return whether the configured endpoint is the DO/Gradient-compatible host."""
    return "inference.do-ai.run" in base_url


def _is_openrouter_endpoint(base_url: str) -> bool:
    """Return whether the configured endpoint is the OpenRouter-compatible host."""
    return "openrouter.ai" in base_url


def _default_provider_base_url() -> str:
    """Resolve the implicit provider host when the user didn't set one explicitly."""
    return (
        os.getenv("API_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENROUTER_BASE_URL")
        or (OPENROUTER_BASE_URL if os.getenv("OPENROUTER_API_KEY") else GRADIENT_BASE_URL)
    )


def _resolve_provider_token(base_url: str) -> tuple[str | None, str | None]:
    """Resolve the most likely valid API token for one provider endpoint."""
    if _is_gradient_endpoint(base_url):
        candidates = [
            ("DIGITALOCEAN_API_TOKEN", os.getenv("DIGITALOCEAN_API_TOKEN")),
            ("MODEL_ACCESS_KEY", os.getenv("MODEL_ACCESS_KEY")),
            ("HF_TOKEN", os.getenv("HF_TOKEN")),
            ("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")),
            ("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY")),
        ]
    elif _is_openrouter_endpoint(base_url):
        candidates = [
            ("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY")),
            ("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")),
            ("MODEL_ACCESS_KEY", os.getenv("MODEL_ACCESS_KEY")),
            ("HF_TOKEN", os.getenv("HF_TOKEN")),
            ("DIGITALOCEAN_API_TOKEN", os.getenv("DIGITALOCEAN_API_TOKEN")),
        ]
    else:
        candidates = [
            ("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")),
            ("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY")),
            ("MODEL_ACCESS_KEY", os.getenv("MODEL_ACCESS_KEY")),
            ("HF_TOKEN", os.getenv("HF_TOKEN")),
            ("DIGITALOCEAN_API_TOKEN", os.getenv("DIGITALOCEAN_API_TOKEN")),
        ]
    for source, value in candidates:
        if value:
            return value, source
    return None, None


def _resolve_provider_model_name(model_name: str, base_url: str) -> str:
    """Translate friendly model aliases into endpoint-specific provider IDs."""
    if _is_openrouter_endpoint(base_url):
        return OPENROUTER_MODEL_ALIASES.get(model_name, model_name)
    if _is_gradient_endpoint(base_url):
        return GRADIENT_MODEL_ALIASES.get(model_name, model_name)
    return model_name


def _resolve_model_pricing(model_name: str, request_model_name: str, base_url: str) -> dict[str, float]:
    """Return endpoint-aware token pricing for budget accounting."""
    candidate_keys = [request_model_name, model_name]
    pricing_table = DEFAULT_MODEL_PRICING
    if _is_openrouter_endpoint(base_url):
        pricing_table = {**DEFAULT_MODEL_PRICING, **OPENROUTER_MODEL_PRICING}
    elif _is_gradient_endpoint(base_url):
        pricing_table = {**DEFAULT_MODEL_PRICING, **GRADIENT_MODEL_PRICING}
    for key in candidate_keys:
        if key in pricing_table:
            return pricing_table[key]
    return {"input": 0.0, "output": 0.0}


class GradientChatProvider:
    """Small chat-completions wrapper for OpenAI-compatible endpoints."""

    def __init__(self, model_name: str, max_retries: int = 6, temperature: float = 0.0) -> None:
        self.model_name = model_name
        self.max_retries = max_retries
        self.temperature = temperature
        self.request_timeout_seconds = float(
            os.getenv("MODEL_REQUEST_TIMEOUT_SECONDS", str(DEFAULT_REQUEST_TIMEOUT_SECONDS))
        )
        base_url = _default_provider_base_url()
        token, token_source = _resolve_provider_token(base_url)
        self.base_url = base_url
        self.token_source = token_source
        self.request_model_name = _resolve_provider_model_name(model_name, base_url)
        if not token:
            raise ValueError(
                "Set HF_TOKEN (or DIGITALOCEAN_API_TOKEN / OPENAI_API_KEY / MODEL_ACCESS_KEY / OPENROUTER_API_KEY) "
                "before running model baselines."
            )
        # Keep retries in one place so provider-specific hangs degrade into a bounded retry loop.
        self.client = OpenAI(base_url=base_url, api_key=token, timeout=self.request_timeout_seconds, max_retries=0)

    def complete(self, messages: list[dict], max_tokens: int = 700) -> ProviderResponse:
        """Run a chat completion with retry handling."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                request_kwargs = {
                    "model": self.request_model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "reasoning_effort": "low",
                }
                if _is_openrouter_endpoint(self.base_url):
                    request_kwargs["max_tokens"] = max_tokens
                else:
                    request_kwargs["max_completion_tokens"] = max_tokens
                request_kwargs["timeout"] = self.request_timeout_seconds
                response = self.client.chat.completions.create(**request_kwargs)
                usage = response.usage
                input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                pricing = _resolve_model_pricing(self.model_name, self.request_model_name, self.base_url)
                cost_usd = input_tokens * pricing["input"] + output_tokens * pricing["output"]
                message = response.choices[0].message
                content = message.content or "{}"
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                return ProviderResponse(
                    content=content,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    raw=response.model_dump(mode="json"),
                )
            except APIStatusError as exc:
                last_error = exc
                message = str(exc).lower()
                if exc.status_code in (401, 403):
                    hint = ""
                    if (
                        _is_gradient_endpoint(self.base_url)
                        and self.token_source == "HF_TOKEN"
                        and os.getenv("DIGITALOCEAN_API_TOKEN")
                    ):
                        hint = (
                            " HF_TOKEN was selected before DIGITALOCEAN_API_TOKEN; "
                            "set HF_TOKEN equal to the DigitalOcean token or unset HF_TOKEN."
                        )
                    raise ProviderCredentialError(
                        f"Provider rejected {self.token_source or 'configured'} credentials for {self.base_url}.{hint}"
                    ) from exc
                if exc.status_code == 429 or 500 <= exc.status_code < 600:
                    if "quota" in message or "billing" in message or "credit" in message:
                        raise ProviderExhaustedError("Provider reported quota or billing exhaustion.") from exc
                    retry_after = 0.0
                    if getattr(exc, "response", None) is not None:
                        retry_value = exc.response.headers.get("retry-after")
                        if retry_value:
                            try:
                                retry_after = float(retry_value)
                            except ValueError:
                                retry_after = 0.0
                    if exc.status_code == 429 and retry_after <= 0.0:
                        # Serverless endpoints often omit Retry-After on burst limits.
                        # Use a calmer backoff so long benchmark jobs can survive transient throttling.
                        retry_after = min(15.0 * (2**attempt), 120.0)
                    elif retry_after <= 0.0:
                        retry_after = min(2**attempt, 12)
                    time.sleep(retry_after)
                    continue
                raise
            except (APIConnectionError, APITimeoutError) as exc:
                last_error = exc
                time.sleep(min(2**attempt, 12))
                continue
        raise RuntimeError(f"Provider request failed after retries: {last_error}")
