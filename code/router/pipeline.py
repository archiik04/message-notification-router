"""End-to-end routing pipeline.

Stage order:
  1. media understanding   (OCR / ASR, cached)
  2. content normalisation (text + media into one view)
  3. safety assessment     (independent of the user)
  4. retrieval             (similar history for this user)
  5. scoring               (urgency, trust, relationship, repetition -> priority)
  6. routing               (thresholds + safety veto)
  7. rationale + evidence  (canonical explanation, outcome-consistent citations)
  8. optional LLM arbitration on genuinely ambiguous rows
  9. self-critic           (hard invariants, always runs last)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .config import Settings
from .content import ContentView, build_content
from .critic import CriticReport, SelfCritic
from .dataio import Dataset
from .memory import MemoryStore
from .multimodal import MediaIndex, build_media_index
from .reasons import BY_KEY, RationaleContext, select
from .retrieval import EvidenceRetriever, evidence_intent
from .safety import SafetyEngine, SafetyVerdict
from .semantic import SemanticIntentScorer, non_latin_ratio
from .schema import Decision, Message
from .scoring import RoutingEngine, SignalBundle

log = logging.getLogger(__name__)

SCHOOL_TYPES = {"school_group"}
WORK_TYPES = {"coworker", "college_faculty", "tech_community"}


@dataclass
class RoutedMessage:
    """A decision plus the intermediate state that produced it.

    Carried forward so the critic can re-examine a decision without paying to
    recompute OCR, retrieval and scoring.
    """

    decision: Decision
    message: Message
    content: ContentView
    safety: SafetyVerdict
    bundle: SignalBundle
    ctx: RationaleContext


@dataclass
class RunStats:
    total: int = 0
    by_action: dict[str, int] = field(default_factory=dict)
    by_type: dict[str, int] = field(default_factory=dict)
    llm_calls: int = 0
    llm_overrides: int = 0
    gray_zone: int = 0
    elapsed_s: float = 0.0

    def render(self) -> str:
        actions = ", ".join(f"{k}={v}" for k, v in sorted(self.by_action.items()))
        types = ", ".join(f"{k}={v}" for k, v in sorted(self.by_type.items(), key=lambda kv: -kv[1]))
        return (
            f"routed {self.total} messages in {self.elapsed_s:.1f}s\n"
            f"  actions: {actions}\n"
            f"  types:   {types}\n"
            f"  gray zone: {self.gray_zone}, llm calls: {self.llm_calls}, llm overrides: {self.llm_overrides}"
        )


class NotificationRouter:
    """Owns the full decision path for a dataset."""

    def __init__(self, settings: Settings, dataset: Dataset, media: MediaIndex | None = None) -> None:
        self.settings = settings
        self.ds = dataset
        self.media = media if media is not None else build_media_index(settings, dataset)
        self.memory = MemoryStore(dataset)
        self.safety = SafetyEngine(dataset)
        self.semantic = SemanticIntentScorer()
        self.retriever = EvidenceRetriever(settings, dataset, self.media)
        self.engine = RoutingEngine(settings, dataset, self.memory)
        self.critic = SelfCritic(settings)
        self.critic.bind_dataset(dataset)
        self._llm = None
        if settings.llm_available():
            from .llm import LLMArbiter

            self._llm = LLMArbiter(settings)
            log.info("LLM arbitration enabled (%s)", settings.llm.model)
        else:
            log.info("LLM disabled - running fully deterministic")

    # ------------------------------------------------------------------
    def _rationale_context(
        self,
        message: Message,
        content: ContentView,
        safety: SafetyVerdict,
        bundle: SignalBundle,
        action: str,
        mtype: str,
    ) -> RationaleContext:
        group = self.ds.groups.get(message.group_id) if message.group_id else None
        biz = self.ds.businesses.get(message.business_id) if message.business_id else None
        rel = (
            self.ds.relationship(message.user_id, message.business_id)
            if message.business_id else None
        )
        sender_stats = (
            self.memory.profile(message.user_id).sender(message.sender_user_id)
            if message.sender_user_id else None
        )

        why = (rel.why_user_knows_account.lower() if rel else "")
        ctx = RationaleContext(action=action, message_type=mtype, drivers=list(bundle.drivers))
        ctx.is_group = message.is_group
        ctx.is_business = message.is_business
        ctx.is_personal = message.is_personal
        ctx.group_type = group.group_type if group else ""
        ctx.sender_is_admin = self.ds.sender_is_admin(message.group_id, message.sender_user_id)
        ctx.verified_business = bool(biz and biz.verified and not biz.domain_mismatch)
        ctx.has_order_history = bool(rel and rel.has_active_relationship and any(
            k in why for k in ("order", "delivery", "grocery", "purchase", "shopping")
        ))
        ctx.has_booking_history = bool(rel and rel.has_active_relationship and any(
            k in why for k in ("booking", "appointment", "reservation", "ticket", "travel", "health")
        ))
        ctx.opted_out = bool(rel and rel.opted_out)
        ctx.dismissed_similar = "business_messages_dismissed" in bundle.drivers or bundle.repetition >= 0.5
        ctx.repeated_pattern = "repeated_pattern_rejected" in bundle.drivers or bundle.repetition >= 0.6
        ctx.trusted_sender = bool(sender_stats and sender_stats.seen >= 1 and sender_stats.affinity >= 0.2)
        ctx.unknown_sender = "unknown_sender" in bundle.drivers
        ctx.first_contact = "first_contact_sender" in safety.threats
        ctx.direct_request = content.has_direct_request
        ctx.mentions_user = content.mentions_user(message.user_id)
        ctx.same_day_deadline = "same_day_deadline" in bundle.drivers
        # Work context follows the person, not the channel. A colleague pinging
        # about a failing deployment is a work message whether it arrives in the
        # team group or as a direct chat.
        ctx.work_context = bool(group and group.group_type in WORK_TYPES) or self._shares_group_type(
            message, WORK_TYPES
        )
        ctx.school_context = bool(group and group.group_type in SCHOOL_TYPES)
        ctx.has_relationship = bool(rel and rel.has_active_relationship)
        ctx.has_link = bool(content.urls)
        ctx.local_language_urgency = bool(content.translit_urgency_hits)
        # A voice note is addressed to its recipient by nature; a text broadcast
        # to a group is not unless it names or asks something of them.
        ctx.directed_at_user = (
            ctx.mentions_user or ctx.direct_request or content.modality == "voice"
        )
        ctx.support_framing = bool(content.impersonation_hits) or any(
            k in content.low for k in ("support alert", "helpdesk", "support team", "customer care")
        )
        ctx.injection = safety.injection_score >= 0.5
        ctx.credential_request = "credential_request" in safety.threats
        ctx.account_threat = "account_threat_pressure" in safety.threats
        ctx.payment_risk = "payment_demand" in safety.threats or "payment_qr_risk" in safety.threats
        # Interest is evidenced either by a live business relationship or by the
        # user having consistently engaged with this sender's offers before.
        ctx.interest_match = bool(rel and rel.has_active_relationship and not ctx.opted_out) or bool(
            sender_stats and sender_stats.seen >= 2 and sender_stats.affinity >= 0.5
        )
        ctx.forwarded_chain = message.forwarded_count >= 3
        ctx.impersonation = "brand_impersonation" in safety.threats
        return ctx

    def _shares_group_type(self, message: Message, group_types: set[str]) -> bool:
        """Does the sender share a group of this kind with the recipient?"""
        if not message.sender_user_id:
            return False
        for (group_id, user_id), _ in self.ds.memberships.items():
            if user_id != message.sender_user_id:
                continue
            group = self.ds.groups.get(group_id)
            if group and group.group_type in group_types and self.ds.membership(group_id, message.user_id):
                return True
        return False

    def _evidence_for(self, message: Message, content: ContentView, action: str, rationale, mtype: str):
        """Retrieve citations that are coherent with the stated rationale.

        A reason that asserts "this is the first message from the sender" must
        not arrive with prior messages attached, or the explanation contradicts
        its own citation. Beyond that the ranking decides: an earlier attempt to
        force multi-citation onto a single counterparty measurably *lowered*
        agreement with the reference, because a repetition pattern can legitimately
        span senders (the same chain forward arriving from two different groups).
        """
        if rationale.evidence_policy == "none":
            return []
        return self.retriever.retrieve(
            message, content, evidence_intent(action, mtype), want=rationale.evidence_want
        )

    def route_one(self, message: Message, stats: RunStats) -> "RoutedMessage":
        content = build_content(message, self.media)
        # Cross-lingual fallback runs only when the lexicons found nothing or
        # the text is largely non-Latin, so the tuned English path is untouched.
        content.non_latin_ratio = non_latin_ratio(content.combined)
        if self.semantic.should_run(content.combined, content.lexicon_hit_count):
            content.semantic_intent, content.semantic_margin = self.semantic.verdict(
                content.combined
            )
        safety = self.safety.assess(message, content)

        similar = self.retriever.candidates(message, content)
        repetition, rep_drivers = self.memory.repetition_score(message, similar)

        bundle = self.engine.score(message, content, safety, repetition, rep_drivers)
        mtype, _ = self.engine.classifier.classify(message, content, safety, bundle.urgency)
        action, mtype = self.engine.decide(bundle, mtype, safety)

        ctx = self._rationale_context(message, content, safety, bundle, action, mtype)
        rationale = select(ctx)

        evidence = self._evidence_for(message, content, action, rationale, mtype)
        evidence_ids = [c.message.message_id for c in evidence]

        decision = Decision(
            message_id=message.message_id,
            action=action,
            message_type=mtype,
            reason=rationale.text,
            confidence=self.engine.confidence(action, rationale.confidence, bundle),
            evidence_message_ids=evidence_ids,
            priority_score=bundle.priority,
            signals=bundle.as_dict(),
            rationale_key=rationale.key,
            drivers=bundle.drivers,
            trace={
                "threats": safety.threats,
                "safety_signals": safety.signals,
                "modality": content.modality,
                "evidence_scores": [
                    {"id": c.message.message_id, "total": c.total, "outcome": c.outcome}
                    for c in evidence
                ],
            },
        )

        if self._is_gray_zone(bundle, safety):
            stats.gray_zone += 1
            decision.trace["gray_zone"] = True
            if self._llm is not None:
                decision = self._arbitrate(decision, message, content, safety, bundle, ctx, stats)

        return RoutedMessage(decision, message, content, safety, bundle, ctx)

    def _is_gray_zone(self, bundle: SignalBundle, safety: SafetyVerdict) -> bool:
        """Rows where the deterministic core is genuinely undecided.

        Spending LLM budget on confident decisions buys nothing, so arbitration
        is reserved for scores hugging a threshold, and never for messages the
        safety engine has already ruled on.
        """
        if safety.is_scam or safety.injection_score >= 0.5:
            return False
        half = self.settings.thresholds.gray_zone_halfwidth
        near_notify = abs(bundle.priority - self.settings.thresholds.notify_floor) <= half
        near_mute = abs(bundle.priority - self.settings.thresholds.mute_ceiling) <= half
        return near_notify or near_mute

    def _arbitrate(
        self,
        decision: Decision,
        message: Message,
        content: ContentView,
        safety: SafetyVerdict,
        bundle: SignalBundle,
        ctx: RationaleContext,
        stats: RunStats,
    ) -> Decision:
        verdict = self._llm.arbitrate(message, content, safety, bundle, decision)
        stats.llm_calls += 1
        if verdict is None:
            return decision
        if verdict.action == decision.action:
            decision.trace["llm"] = {"agreed": True, "note": verdict.note}
            return decision

        # The LLM may only move a decision one step along notify/digest/mute and
        # may never overrule the safety engine.
        order = {"mute": 0, "digest": 1, "notify": 2}
        if abs(order[verdict.action] - order[decision.action]) > 1:
            decision.trace["llm"] = {"agreed": False, "rejected": "jump_too_large"}
            return decision
        if verdict.action == "notify" and safety.risk >= self.settings.thresholds.safety_notify_veto:
            decision.trace["llm"] = {"agreed": False, "rejected": "safety_veto"}
            return decision

        stats.llm_overrides += 1
        decision.trace["llm"] = {
            "agreed": False, "from": decision.action, "to": verdict.action, "note": verdict.note,
        }
        decision.action = verdict.action
        if verdict.message_type:
            decision.message_type = verdict.message_type

        ctx.action = decision.action
        ctx.message_type = decision.message_type
        rationale = select(ctx)
        decision.rationale_key = rationale.key
        decision.reason = rationale.text
        decision.confidence = self.engine.confidence(decision.action, rationale.confidence, bundle)

        evidence = self._evidence_for(
            message, content, decision.action, rationale, decision.message_type
        )
        decision.evidence_message_ids = [c.message.message_id for c in evidence]
        return decision

    # ------------------------------------------------------------------
    def run(self, messages: list[Message] | None = None) -> tuple[list[Decision], RunStats, CriticReport]:
        targets = messages if messages is not None else self.ds.messages
        stats = RunStats(total=len(targets))
        report = CriticReport()
        started = time.time()

        decisions: list[Decision] = []
        for message in targets:
            try:
                routed = self.route_one(message, stats)
            except Exception as exc:  # noqa: BLE001 - one bad row must not sink the run
                log.exception("routing failed for %s: %s", message.message_id, exc)
                # Failing safe means deferring, never dropping: an unanalysable
                # message still reaches the user in their digest.
                decisions.append(
                    Decision(
                        message_id=message.message_id,
                        action="digest",
                        message_type="unknown",
                        reason="The message could not be fully analysed, so it is deferred rather than dropped.",
                        confidence=0.80,
                        rationale_key="digest_fallback",
                        trace={"error": str(exc)},
                    )
                )
                continue

            routed.ctx.action = routed.decision.action
            routed.ctx.message_type = routed.decision.message_type
            decisions.append(
                self.critic.review(
                    routed.decision, routed.message, routed.content,
                    routed.safety, routed.bundle, routed.ctx, report,
                )
            )

        for d in decisions:
            stats.by_action[d.action] = stats.by_action.get(d.action, 0) + 1
            stats.by_type[d.message_type] = stats.by_type.get(d.message_type, 0) + 1
        stats.elapsed_s = time.time() - started
        return decisions, stats, report
