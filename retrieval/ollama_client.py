"""Minimal Ollama HTTP client used by SAH-RAG's structural selector."""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


class OllamaUnavailable(RuntimeError):
    """Raised when the local Ollama server or model cannot answer."""


class OllamaClient:
    """Small synchronous client for a local Ollama model.

    The existing OpenRouter LLM path remains untouched; this client exists so
    structure selection can run on a local model as requested.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gemma3:4b",
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def generate(self, prompt: str, json_mode: bool = False, temperature: float = 0.0) -> str:
        """Return raw model text; callers validate and parse it."""
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 512},
        }
        if json_mode:
            payload["format"] = "json"
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return str(data.get("response", ""))
        except requests.RequestException as exc:
            raise OllamaUnavailable(f"Ollama request failed: {exc}") from exc
        except (ValueError, KeyError) as exc:
            raise OllamaUnavailable(f"Ollama returned an invalid response: {exc}") from exc


def strip_code_fence(text: str) -> str:
    """Remove Markdown fences and surrounding prose before JSON parsing."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def parse_json_payload(text: str) -> Any:
    """Parse a JSON object/array even if the model added surrounding text."""
    cleaned = strip_code_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("No valid JSON object or array found")
