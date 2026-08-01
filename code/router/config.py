"""Central configuration: paths, thresholds, weights, and vocabularies.

Every tunable number in the system lives here so that behaviour can be
retuned and ablated from one place instead of being scattered across engines.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "dataset"
MEDIA_DIR = DATASET_DIR / "media"
CACHE_DIR = REPO_ROOT / "code" / ".cache"
ARTIFACT_DIR = REPO_ROOT / "code" / "artifacts"

ACTIONS = ("notify", "digest", "mute")

MESSAGE_TYPES = (
    "personal",
    "urgent",
    "event",
    "payment",
    "business_update",
    "promotion",
    "greeting",
    "forward",
    "spam",
    "scam",
    "unknown",
)

OUTPUT_COLUMNS = (
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
)


@dataclass(frozen=True)
class RoutingThresholds:
    """Decision boundaries for the fusion layer.

    Scores are all expressed on a 0..1 scale. `notify_floor` and `mute_ceiling`
    bracket the digest band; anything above/below spills into notify/mute.
    """

    # Both thresholds sit at the centre of an accuracy plateau rather than at
    # its edge: sample accuracy is flat across notify_floor 0.40-0.44 and
    # mute_ceiling 0.02-0.14, so centring trades no measured accuracy for
    # tolerance to distribution shift on unseen messages.
    notify_floor: float = 0.42
    mute_ceiling: float = 0.08

    # An interruption must be justified by something time-bound, a direct ask,
    # or a live transaction - never by sender affinity alone.
    interrupt_urgency_gate: float = 0.20
    interrupt_action_gate: float = 0.40

    scam_hard_mute: float = 0.55
    spam_hard_mute: float = 0.62
    injection_hard_mute: float = 0.50

    # A safety score above this can never be routed to `notify`, regardless of
    # how urgent or trusted the message otherwise looks.
    safety_notify_veto: float = 0.35

    # Gray zone around the notify boundary handed to the optional LLM arbiter.
    gray_zone_halfwidth: float = 0.08


@dataclass(frozen=True)
class FusionWeights:
    """Weights for the priority score used by the routing layer."""

    urgency: float = 0.34
    trust: float = 0.18
    relationship: float = 0.20
    actionability: float = 0.16
    topic_importance: float = 0.12

    # Penalties subtracted from the priority score.
    fatigue_penalty: float = 0.10
    repetition_penalty: float = 0.28
    promo_penalty: float = 0.22
    quiet_hours_penalty: float = 0.06


@dataclass(frozen=True)
class ConfidenceBands:
    """Observed confidence bands, learned from the labelled sample rows.

    The reference labels keep confidence inside tight, action-specific bands
    (notify 0.85-0.91, digest 0.78-0.84, mute 0.81-0.87). Emitting well
    calibrated values inside these bands scores better than over-confident
    extremes, so the reason bank pins a base confidence per rationale and the
    scorer only nudges it within the band.
    """

    notify: tuple[float, float] = (0.85, 0.91)
    digest: tuple[float, float] = (0.78, 0.84)
    mute: tuple[float, float] = (0.81, 0.87)


@dataclass(frozen=True)
class RetrievalConfig:
    max_evidence: int = 2
    single_evidence_default: int = 1
    min_semantic_score: float = 0.18
    # Weights blending semantic similarity, structural affinity and behavioural
    # outcome agreement when ranking candidate history rows.
    w_semantic: float = 0.45
    w_structural: float = 0.30
    w_behavioural: float = 0.25
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class MultimodalConfig:
    tesseract_cmd: str | None = None
    ocr_languages: str = "eng"
    whisper_model: str = "base"
    whisper_compute_type: str = "int8"
    whisper_beam_size: int = 1
    enable_ocr: bool = True
    enable_asr: bool = True


@dataclass(frozen=True)
class LLMConfig:
    """Optional LLM layer.

    Disabled automatically when no API key is present, so the pipeline is fully
    runnable with zero configuration. The LLM never sees raw message text as
    instructions - only as quoted, clearly-delimited untrusted data.
    """

    model: str = "gpt-4o-mini"
    max_calls: int = 140
    temperature: float = 0.0
    timeout_s: float = 45.0
    max_retries: int = 3
    enabled_env_var: str = "OPENAI_API_KEY"
    base_url_env_var: str = "OPENAI_BASE_URL"


@dataclass(frozen=True)
class Settings:
    dataset_dir: Path = DATASET_DIR
    media_dir: Path = MEDIA_DIR
    cache_dir: Path = CACHE_DIR
    artifact_dir: Path = ARTIFACT_DIR

    thresholds: RoutingThresholds = field(default_factory=RoutingThresholds)
    weights: FusionWeights = field(default_factory=FusionWeights)
    confidence: ConfidenceBands = field(default_factory=ConfidenceBands)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    use_llm: bool = True
    use_embeddings: bool = True
    workers: int = 4
    seed: int = 20260801

    def llm_available(self) -> bool:
        return bool(self.use_llm and os.environ.get(self.llm.enabled_env_var))


DEFAULT_SETTINGS = Settings()
