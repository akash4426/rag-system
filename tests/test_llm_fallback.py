"""LLM engine fallback tests: OpenRouter failures retry on local Ollama."""

from __future__ import annotations

from types import SimpleNamespace

from core.generation.llm_engine import LLMEngine


class FailingPrimary:
    def invoke(self, prompt):
        raise RuntimeError("Error code: 402 - Insufficient credits")


class StaticPrimary:
    def __init__(self, content):
        self.content = content

    def invoke(self, prompt):
        return SimpleNamespace(content=self.content)


class FakeFallback:
    model = "gemma3:4b"

    def __init__(self, answer="local answer", error=None):
        self.answer = answer
        self.error = error
        self.calls = []

    def generate(self, prompt, json_mode=False, temperature=0.0):
        self.calls.append((prompt, json_mode, temperature))
        if self.error:
            raise self.error
        return self.answer


def make_engine(monkeypatch, fallback_enabled: str = "true"):
    monkeypatch.delenv("OPEN_ROUTER_API", raising=False)
    monkeypatch.setenv("OLLAMA_FALLBACK_ENABLED", fallback_enabled)
    return LLMEngine()


def test_openrouter_success_does_not_call_ollama(monkeypatch):
    engine = make_engine(monkeypatch)
    engine.llm = StaticPrimary("remote answer")
    fallback = FakeFallback()
    engine.fallback = fallback

    answer, provider = engine.generate_with_provider("prompt")

    assert answer == "remote answer"
    assert provider == "openrouter"
    assert fallback.calls == []


def test_openrouter_402_falls_back_to_ollama(monkeypatch):
    engine = make_engine(monkeypatch)
    engine.llm = FailingPrimary()
    fallback = FakeFallback("local answer")
    engine.fallback = fallback

    answer, provider = engine.generate_with_provider("prompt")

    assert answer == "local answer"
    assert provider == "ollama:gemma3:4b"
    assert fallback.calls == [("prompt", False, 0.3)]


def test_missing_openrouter_key_uses_ollama(monkeypatch):
    engine = make_engine(monkeypatch)
    fallback = FakeFallback("local answer")
    engine.fallback = fallback

    answer, provider = engine.generate_with_provider("prompt")

    assert answer == "local answer"
    assert provider == "ollama:gemma3:4b"


def test_fallback_can_be_disabled(monkeypatch):
    engine = make_engine(monkeypatch, fallback_enabled="false")
    engine.llm = FailingPrimary()

    answer, provider = engine.generate_with_provider("prompt")

    assert engine.fallback is None
    assert provider == "error"
    assert answer.startswith("Error during generation:")
    assert "OpenRouter" in answer


def test_both_providers_failing_reports_both_errors(monkeypatch):
    engine = make_engine(monkeypatch)
    engine.llm = FailingPrimary()
    engine.fallback = FakeFallback(error=RuntimeError("connection refused"))

    answer, provider = engine.generate_with_provider("prompt")

    assert provider == "error"
    assert "OpenRouter" in answer
    assert "Ollama fallback" in answer
    assert "connection refused" in answer


def test_generate_keeps_string_api(monkeypatch):
    engine = make_engine(monkeypatch)
    engine.llm = FailingPrimary()
    engine.fallback = FakeFallback("local answer")

    assert engine.generate("prompt") == "local answer"
