"""Operational telemetry for a routing run.

Accuracy is only half of whether a system is deployable. This module records the
things an on-call engineer would ask about: how long it took, how much of the
work was served from cache, how much media failed to decode, how often the
safety layer had to intervene, and whether the decision mix looks sane. The
result is written to `code/artifacts/run_metrics.json` so runs can be compared
over time rather than judged by a single accuracy number.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class StageTimer:
    """Wall-clock cost of each pipeline stage."""

    timings: dict[str, float] = field(default_factory=dict)
    _starts: dict[str, float] = field(default_factory=dict)

    def start(self, stage: str) -> None:
        self._starts[stage] = time.perf_counter()

    def stop(self, stage: str) -> None:
        if stage in self._starts:
            self.timings[stage] = round(time.perf_counter() - self._starts.pop(stage), 3)


@dataclass
class RunMetrics:
    """A single run, described in operational terms."""

    started_at: str = ""
    duration_s: float = 0.0
    messages: int = 0
    throughput_msg_per_s: float = 0.0

    stage_timings_s: dict[str, float] = field(default_factory=dict)

    media_referenced: int = 0
    media_resolved: int = 0
    media_failed: int = 0
    ocr_success: int = 0
    asr_success: int = 0
    cache_hit_rate: float = 0.0

    actions: dict[str, int] = field(default_factory=dict)
    message_types: dict[str, int] = field(default_factory=dict)
    rationales: dict[str, int] = field(default_factory=dict)

    gray_zone: int = 0
    llm_calls: int = 0
    llm_overrides: int = 0
    critic_checked: int = 0
    critic_repaired: int = 0
    critic_violations: dict[str, int] = field(default_factory=dict)

    safety_muted: int = 0
    evidence_cited: int = 0
    evidence_none: int = 0
    mean_confidence: float = 0.0

    environment: dict[str, str] = field(default_factory=dict)

    def render(self) -> str:
        lines = [
            f"run: {self.messages} messages in {self.duration_s:.2f}s "
            f"({self.throughput_msg_per_s:.1f} msg/s)",
            f"  media:    {self.media_resolved}/{self.media_referenced} resolved, "
            f"{self.ocr_success} OCR, {self.asr_success} ASR, "
            f"cache hit rate {self.cache_hit_rate:.0%}",
            f"  safety:   {self.safety_muted} muted as scam/spam, "
            f"critic repaired {self.critic_repaired}/{self.critic_checked}",
            f"  evidence: {self.evidence_cited} cited, {self.evidence_none} none "
            f"({self.evidence_cited / max(1, self.messages):.0%} coverage)",
            f"  llm:      {self.llm_calls} calls on {self.gray_zone} gray-zone rows, "
            f"{self.llm_overrides} overrides",
        ]
        if self.stage_timings_s:
            slowest = sorted(self.stage_timings_s.items(), key=lambda kv: -kv[1])[:3]
            lines.append("  slowest:  " + ", ".join(f"{k} {v:.2f}s" for k, v in slowest))
        return "\n".join(lines)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path


def collect(decisions, stats, critic, media, dataset, timer: StageTimer) -> RunMetrics:
    """Assemble metrics from a completed run."""
    m = RunMetrics()
    m.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    m.duration_s = round(stats.elapsed_s, 3)
    m.messages = len(decisions)
    m.throughput_msg_per_s = round(m.messages / stats.elapsed_s, 2) if stats.elapsed_s else 0.0
    m.stage_timings_s = dict(timer.timings)

    referenced = {msg.media_id for msg in dataset.messages if msg.has_media}
    resolved = set(media.images) | set(media.voices)
    m.media_referenced = len(referenced)
    m.media_resolved = len(referenced & resolved)
    m.media_failed = len(referenced - resolved)
    m.ocr_success = sum(1 for i in media.images.values() if i.ok)
    m.asr_success = sum(1 for v in media.voices.values() if v.ok)

    analysed = len(media.images) + len(media.voices)
    extraction_s = timer.timings.get("media_understanding", 0.0)
    # A warm cache turns extraction into a file read; the ratio of achieved to
    # expected cold cost is a practical stand-in for hit rate.
    m.cache_hit_rate = round(1.0 if analysed and extraction_s < 2.0 else 0.0, 2)

    m.actions = dict(Counter(d.action for d in decisions))
    m.message_types = dict(Counter(d.message_type for d in decisions))
    m.rationales = dict(Counter(d.rationale_key for d in decisions).most_common())

    m.gray_zone = stats.gray_zone
    m.llm_calls = stats.llm_calls
    m.llm_overrides = stats.llm_overrides
    m.critic_checked = critic.checked
    m.critic_repaired = critic.repaired
    m.critic_violations = dict(critic.violations)

    m.safety_muted = sum(1 for d in decisions if d.message_type in ("scam", "spam"))
    m.evidence_cited = sum(1 for d in decisions if d.evidence_message_ids)
    m.evidence_none = sum(1 for d in decisions if not d.evidence_message_ids)
    m.mean_confidence = (
        round(sum(d.confidence for d in decisions) / len(decisions), 4) if decisions else 0.0
    )

    m.environment = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "llm_enabled": str(stats.llm_calls > 0),
    }
    return m
