"""Provider token-resolution tests."""

from __future__ import annotations

import latentgoalops.baseline.providers as providers
from latentgoalops.baseline.providers import (
    OPENROUTER_BASE_URL,
    _resolve_model_pricing,
    _default_provider_base_url,
    _resolve_provider_model_name,
    _resolve_provider_token,
)


def test_gradient_endpoint_prefers_digitalocean_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_token")
    monkeypatch.setenv("DIGITALOCEAN_API_TOKEN", "do_token")
    token, source = _resolve_provider_token("https://inference.do-ai.run/v1")
    assert token == "do_token"
    assert source == "DIGITALOCEAN_API_TOKEN"


def test_non_gradient_endpoint_prefers_openai_key(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_token")
    monkeypatch.setenv("DIGITALOCEAN_API_TOKEN", "do_token")
    monkeypatch.setenv("OPENAI_API_KEY", "openai_token")
    token, source = _resolve_provider_token("https://api.openai.com/v1")
    assert token == "openai_token"
    assert source == "OPENAI_API_KEY"


def test_openrouter_endpoint_prefers_openrouter_key(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_token")
    monkeypatch.setenv("DIGITALOCEAN_API_TOKEN", "do_token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or_token")
    token, source = _resolve_provider_token(OPENROUTER_BASE_URL)
    assert token == "or_token"
    assert source == "OPENROUTER_API_KEY"


def test_openrouter_key_sets_default_base_url(monkeypatch):
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or_token")
    assert _default_provider_base_url() == OPENROUTER_BASE_URL


def test_model_aliases_follow_provider_endpoint():
    assert _resolve_provider_model_name("openai-gpt-oss-20b", OPENROUTER_BASE_URL) == "openai/gpt-oss-20b"
    assert _resolve_provider_model_name("openai/gpt-oss-20b", "https://inference.do-ai.run/v1") == "openai-gpt-oss-20b"


def test_model_pricing_tracks_provider_endpoint():
    openrouter_pricing = _resolve_model_pricing("openai-gpt-oss-20b", "openai/gpt-oss-20b", OPENROUTER_BASE_URL)
    gradient_pricing = _resolve_model_pricing("openai-gpt-oss-20b", "openai-gpt-oss-20b", "https://inference.do-ai.run/v1")
    assert openrouter_pricing["input"] < gradient_pricing["input"]
    assert openrouter_pricing["output"] < gradient_pricing["output"]


def test_provider_client_uses_bounded_timeout(monkeypatch):
    captured: dict[str, float | int | str] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("DIGITALOCEAN_API_TOKEN", "do_token")
    monkeypatch.setenv("API_BASE_URL", "https://inference.do-ai.run/v1")
    monkeypatch.setenv("MODEL_REQUEST_TIMEOUT_SECONDS", "37")
    monkeypatch.setattr(providers, "OpenAI", FakeOpenAI)

    provider = providers.GradientChatProvider("openai-gpt-oss-20b")

    assert provider.request_timeout_seconds == 37.0
    assert captured["timeout"] == 37.0
    assert captured["max_retries"] == 0
