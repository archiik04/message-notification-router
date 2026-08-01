"""Second-stage verification.

The scorer is optimistic by construction: it fuses signals and thresholds them.
The critic is adversarial - it re-reads each finished decision and asks what
would have to be true for it to be wrong. Violations are repaired in place and
recorded, so a run always reports how often the first stage needed correction.

These invariants are deterministic and run whether or not the LLM layer is on;
the LLM is never trusted to police itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .config import Settings
from .content import ContentView
from .reasons import BY_KEY, RationaleContext, select
from .safety import SafetyVerdict
from .schema import Decision, Message
from .scoring import SignalBundle

log = logging.getLogger(__name__)


@dataclass
class CriticReport:
    checked: int = 0
    repaired: int = 0
    violations: dict[str, int] = field(default_factory=dict)
    details: list[str] = field(default_factory=list)

    def record(self, rule: str, message_id: str, note: str) -> None:
        self.violations[rule] = self.violations.get(rule, 0) + 1
        self.details.append(f"{message_id}: {rule} - {note}")

    def render(self) -> str:
        if not self.violations:
            return f"critic: {self.checked} decisions checked, no violations"
        lines = [f"critic: {self.checked} checked, {self.repaired} repaired"]
        for rule, count in sorted(self.violations.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {rule}: {count}")
        return "\n".join(lines)


class SelfCritic:
    """Enforces hard invariants that no routing decision may violate."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def review(
        self,
        decision: Decision,
        message: Message,
        content: ContentView,
        safety: SafetyVerdict,
        bundle: SignalBundle,
        ctx: RationaleContext,
        report: CriticReport,
    ) -> Decision:
        report.checked += 1
        repaired = False

        # 1. Fraud must never be delivered as an interruption.
        if (safety.is_scam or safety.injection_score >= 0.5) and decision.action != "mute":
            report.record("scam_not_muted", decision.message_id,
                          f"risk={safety.risk} action was {decision.action}")
            decision.action, decision.message_type = "mute", "scam"
            repaired = True

        # 2. A scam label and a non-mute action cannot coexist.
        if decision.message_type == "scam" and decision.action != "mute":
            report.record("scam_type_action_mismatch", decision.message_id, decision.action)
            decision.action = "mute"
            repaired = True

        # 3. Bulk marketing must never interrupt.
        if decision.message_type == "spam" and decision.action == "notify":
            report.record("spam_notified", decision.message_id, "spam routed to notify")
            decision.action = "mute"
            repaired = True

        # 4. Risky content cannot be promoted to notify by personalisation.
        if decision.action == "notify" and safety.risk >= self.settings.thresholds.safety_notify_veto:
            report.record("risky_notify", decision.message_id, f"risk={safety.risk}")
            decision.action = "digest"
            repaired = True

        # 5. Do not silence a genuine, safe, personally-directed request. Muting
        #    a real ask is the costliest error the system can make.
        if decision.action == "mute" and safety.risk < 0.25 and decision.message_type != "spam":
            directed = content.mentions_user(message.user_id) or content.has_direct_request
            if directed and bundle.urgency >= 0.5 and bundle.relationship >= 0.5:
                report.record("urgent_muted", decision.message_id,
                              f"urgency={bundle.urgency} rel={bundle.relationship}")
                decision.action = "notify" if bundle.urgency >= 0.65 else "digest"
                repaired = True

        # 6. An empty-content message cannot claim a confident specific type.
        if content.is_empty and decision.message_type not in ("unknown", "scam", "spam"):
            report.record("empty_content_typed", decision.message_id, decision.message_type)
            decision.message_type = "unknown"
            repaired = True

        # 7. The stated reason must belong to the action that was taken.
        rationale = BY_KEY.get(decision.rationale_key)
        if rationale is None or rationale.action != decision.action:
            ctx.action = decision.action
            ctx.message_type = decision.message_type
            replacement = select(ctx)
            report.record("reason_action_mismatch", decision.message_id,
                          f"{decision.rationale_key or 'none'} -> {replacement.key}")
            decision.rationale_key = replacement.key
            decision.reason = replacement.text
            decision.confidence = replacement.confidence
            repaired = True

        # 8. Confidence must sit inside the band for the final action.
        low, high = getattr(self.settings.confidence, decision.action)
        if not low <= decision.confidence <= high:
            report.record("confidence_out_of_band", decision.message_id,
                          f"{decision.confidence} not in [{low},{high}]")
            decision.confidence = round(min(high, max(low, decision.confidence)), 2)
            repaired = True

        # 9. Evidence must not contradict the decision it is cited for.
        if decision.evidence_message_ids:
            kept = self._filter_contradictory_evidence(decision, message)
            if kept != decision.evidence_message_ids:
                report.record("contradictory_evidence", decision.message_id,
                              f"{decision.evidence_message_ids} -> {kept}")
                decision.evidence_message_ids = kept
                repaired = True

        if repaired:
            report.repaired += 1
            decision.trace["critic_repaired"] = True
        return decision

    def _filter_contradictory_evidence(self, decision: Decision, message: Message) -> list[str]:
        """Drop cited history whose recorded outcome argues the opposite way.

        Applied to behavioural decisions only. A safety mute is justified by
        resemblance to a known threat rather than by the user having rejected
        it, and users frequently did engage with the earlier scam - so demanding
        rejection there would strip away the most relevant citation.
        """
        dataset = getattr(self, "_dataset", None)
        if dataset is None or decision.message_type in ("scam", "spam"):
            return decision.evidence_message_ids
        kept = []
        for mid in decision.evidence_message_ids:
            event = dataset.event_for(message.user_id, mid)
            if event is None:
                kept.append(mid)
                continue
            if decision.action == "mute" and event.positive and not event.negative:
                continue
            if decision.action == "notify" and event.negative and not event.positive:
                continue
            kept.append(mid)
        return kept or decision.evidence_message_ids

    def bind_dataset(self, dataset) -> None:
        self._dataset = dataset
