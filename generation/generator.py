"""Grounded answer generation over an evidence bundle."""

from __future__ import annotations

from evidence.types import EvidenceSet
from generation.prompt import GroundedPromptBuilder
from retrieval.ollama_client import OllamaClient
from sah_config import SAHConfig


class GroundedGenerator:
    """Call the local generation model with a citation-first evidence prompt."""

    def __init__(
        self,
        config: SAHConfig | None = None,
        client: OllamaClient | None = None,
        prompt_builder: GroundedPromptBuilder | None = None,
    ):
        self.config = config or SAHConfig.from_env()
        self.client = client or OllamaClient(
            base_url=self.config.ollama_base_url,
            model=self.config.ollama_model,
            timeout=self.config.ollama_timeout,
        )
        self.prompt_builder = prompt_builder or GroundedPromptBuilder()

    def build_prompt(self, query: str, evidence: EvidenceSet) -> str:
        return self.prompt_builder.build(query, evidence)

    def generate(self, query: str, evidence: EvidenceSet) -> str:
        """Return raw model text; validation happens outside the generator."""
        prompt = self.build_prompt(query, evidence)
        return self.client.generate(prompt, json_mode=False, temperature=0.0)
