"""Typed, environment-driven feature flags for SAH-RAG.

Every experiment can run independently:
E1 baseline, E2 structure-aware retrieval, E3 adaptive fusion,
E4 reranking, E5 validation + grounded generation + citation verification.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class SAHConfig:
    enabled: bool = True
    structure_enabled: bool = True
    structural_retrieval: bool = False
    adaptive_controller: bool = False
    rerank: bool = False
    evidence_expansion: bool = False
    evidence_validation: bool = False
    grounded_generation: bool = False
    citation_verification: bool = False
    structure_guided_retry: bool = True
    max_retries: int = 2
    top_k: int = 5
    structural_top_n: int = 3
    context_token_budget: int = 1800
    relevance_threshold: float = -2.0
    sufficiency_threshold: float = 0.35
    citation_threshold: float = 0.0

    db_path: str = "data/sah_rag.db"
    structure_json_dir: str = "data/structure"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"
    ollama_timeout: int = 120
    llm_provider: str = "ollama"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_pinecone: bool = False

    @classmethod
    def from_env(cls) -> "SAHConfig":
        """Read SAH_* environment variables with safe defaults."""
        provider = os.getenv("SAH_LLM_PROVIDER", "ollama").strip().lower()
        retry_mode = os.getenv("SAH_RETRY_MODE", "structure").strip().lower()
        if retry_mode not in {"structure", "plain"}:
            retry_mode = "structure"
        return cls(
            enabled=_env_bool("SAH_ENABLED", True),
            structure_enabled=_env_bool("SAH_STRUCTURE_ENABLED", True),
            structural_retrieval=_env_bool("SAH_STRUCTURAL_RETRIEVAL", False),
            adaptive_controller=_env_bool("SAH_ADAPTIVE_CONTROLLER", False),
            rerank=_env_bool("SAH_RERANK", False),
            evidence_expansion=_env_bool("SAH_EVIDENCE_EXPANSION", False),
            evidence_validation=_env_bool("SAH_EVIDENCE_VALIDATION", False),
            grounded_generation=_env_bool("SAH_GROUNDED_GENERATION", False),
            citation_verification=_env_bool("SAH_CITATION_VERIFICATION", False),
            structure_guided_retry=retry_mode == "structure",
            max_retries=_env_int("SAH_MAX_RETRIES", 2),
            top_k=_env_int("SAH_TOP_K", 5),
            structural_top_n=_env_int("SAH_STRUCTURAL_TOP_N", 3),
            context_token_budget=_env_int("SAH_CONTEXT_TOKEN_BUDGET", 1800),
            relevance_threshold=_env_float("SAH_RELEVANCE_THRESHOLD", -2.0),
            sufficiency_threshold=_env_float("SAH_SUFFICIENCY_THRESHOLD", 0.35),
            citation_threshold=_env_float("SAH_CITATION_THRESHOLD", 0.0),
            db_path=os.getenv("SAH_DB_PATH", "data/sah_rag.db"),
            structure_json_dir=os.getenv("SAH_STRUCTURE_JSON_DIR", "data/structure"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:4b"),
            ollama_timeout=_env_int("OLLAMA_TIMEOUT", 120),
            llm_provider=provider,
            reranker_model=os.getenv("SAH_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            use_pinecone=_env_bool("SAH_USE_PINECONE", False),
        )


EXPERIMENTS = {"E1", "E2", "E3", "E4", "E5"}


def config_for_experiment(name: str, base: SAHConfig | None = None) -> SAHConfig:
    """Return a copy of the config with experiment-relevant flags enabled."""
    import copy

    config = copy.deepcopy(base or SAHConfig.from_env())
    name = name.upper()
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {name}")

    config.structural_retrieval = name in {"E2", "E3", "E4", "E5"}
    config.adaptive_controller = name in {"E3", "E4", "E5"}
    config.rerank = name in {"E4", "E5"}
    config.evidence_expansion = name in {"E4", "E5"}
    config.evidence_validation = name == "E5"
    config.grounded_generation = name == "E5"
    config.citation_verification = name == "E5"
    if name == "E1":
        config.structure_enabled = False
    return config
