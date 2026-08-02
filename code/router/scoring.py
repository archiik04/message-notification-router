"""Urgency estimation, type classification, and decision fusion.

The router does not classify messages directly into actions. It estimates a set
of interpretable intermediate quantities - urgency, trust, relationship,
actionability, risk - and fuses them into a single priority score that is then
thresholded. That indirection is what makes the system explainable: every
decision can be traced to the signals that moved it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import ConfidenceBands, FusionWeights, RoutingThresholds, Settings
from .content import ContentView
from .memory import MemoryStore
from .safety import SafetyVerdict
from .schema import Message

WORK_GROUP_TYPES = {"coworker", "college_faculty", "tech_community"}
SCHOOL_GROUP_TYPES = {"school_group"}
OPERATIONAL_GROUP_TYPES = {"society", "safety", "caregiving", "family", "extended_family"}


@dataclass
class SignalBundle:
    """All intermediate quantities behind one routing decision."""

    urgency: float = 0.0
    trust: float = 0.5
    relationship: float = 0.5
    actionability: float = 0.0
    topic_importance: float = 0.0

    repetition: float = 0.0
    fatigue: float = 0.0
    promo_pressure: float = 0.0
    quiet_hours: bool = False
    transactional: bool = False   # concerns something the user has in flight

    scam: float = 0.0
    spam: float = 0.0
    injection: float = 0.0

    priority: float = 0.0
    drivers: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, float]:
        return {
            "urgency": self.urgency,
            "trust": self.trust,
            "relationship": self.relationship,
            "actionability": self.actionability,
            "topic_importance": self.topic_importance,
            "repetition": self.repetition,
            "fatigue": self.fatigue,
            "promo_pressure": self.promo_pressure,
            "scam": self.scam,
            "spam": self.spam,
            "injection": self.injection,
            "priority": self.priority,
        }


class UrgencyEngine:
    """Estimates how time-critical a message is, and how much it asks of the user."""

    def __init__(self, dataset) -> None:
        self.ds = dataset

    def score(self, message: Message, content: ContentView) -> tuple[float, float, list[str]]:
        urgency, actionability, drivers = 0.0, 0.0, []

        explicit = len(content.urgency_hits)
        if explicit:
            urgency += min(0.34, 0.16 * explicit)
            drivers.append("explicit_urgency_language")

        if content.semantic_intent == "urgent":
            urgency += 0.34 + min(0.16, content.semantic_margin)
            drivers.append("multilingual_urgency_similarity")
        elif content.semantic_intent in ("promotion", "benign"):
            urgency -= 0.10

        if content.translit_urgency_hits:
            urgency += min(0.24, 0.12 * len(content.translit_urgency_hits))
            drivers.append("transliterated_urgency")

        if content.time_pressure_hits:
            same_day = any(
                h in ("today", "tonight", "now", "this morning", "this afternoon", "this evening")
                or h.startswith(("in ", "within ", "by ", "before ", "next "))
                for h in content.time_pressure_hits
            )
            urgency += 0.26 if same_day else 0.12
            if same_day:
                drivers.append("same_day_deadline")

        if content.has_direct_request and not content.feedback_hits:
            actionability += 0.42
            drivers.append("direct_request")
        elif content.feedback_hits:
            # Asking for a rating is a request in form only.
            actionability += 0.05
            drivers.append("feedback_request")
        if content.has_question:
            actionability += 0.14
        if content.mentions_user(message.user_id):
            actionability += 0.34
            urgency += 0.14
            drivers.append("direct_mention")

        # Voice notes carry prosody: a short, fast message is usually a real ask.
        if content.modality == "voice":
            if content.voice_rushed:
                urgency += 0.16
                drivers.append("rushed_speech")
            if 0 < content.voice_duration_s <= 12 and content.urgency_hits:
                urgency += 0.10
            if content.voice_duration_s >= 30:
                # Long monologues are overwhelmingly recorded marketing.
                urgency -= 0.12
                drivers.append("long_monologue")

        if message.group_id:
            group = self.ds.groups.get(message.group_id)
            if group:
                if group.group_type in WORK_GROUP_TYPES:
                    urgency += 0.10
                elif group.group_type in SCHOOL_GROUP_TYPES:
                    urgency += 0.12
                elif group.group_type in OPERATIONAL_GROUP_TYPES:
                    urgency += 0.06
            if self.ds.sender_is_admin(message.group_id, message.sender_user_id):
                urgency += 0.10
                drivers.append("sender_is_group_admin")

        # Promotional urgency ("hurry, offer ends today") is manufactured, not real.
        if len(content.promo_hits) >= 2:
            urgency -= 0.22
            drivers.append("manufactured_urgency")

        if content.greeting_hits and not content.has_direct_request:
            urgency -= 0.16
        if message.forwarded_count >= 3:
            urgency -= 0.14

        # The sender's own framing outranks keyword matches. "Call me tomorrow,
        # nothing urgent" contains a request and a time, but explicitly is not
        # an interruption - and only the sender can tell us that.
        if content.deescalation_hits:
            damping = min(0.85, 0.35 * len(content.deescalation_hits))
            urgency *= 1.0 - damping
            actionability *= 1.0 - damping * 0.6
            drivers.append("sender_deprioritised_message")
            if "same_day_deadline" in drivers and damping >= 0.30:
                drivers.remove("same_day_deadline")

        return (
            round(max(0.0, min(1.0, urgency)), 3),
            round(max(0.0, min(1.0, actionability)), 3),
            drivers,
        )


class TypeClassifier:
    """Assigns the best-fit message_type from the allowed vocabulary."""

    def __init__(self, dataset) -> None:
        self.ds = dataset

    def classify(
        self, message: Message, content: ContentView, safety: SafetyVerdict, urgency: float
    ) -> tuple[str, list[str]]:
        drivers: list[str] = []

        # Risk categories win outright: a scam poster is a scam, not a promotion.
        if safety.injection_score >= 0.5 or safety.is_scam:
            return "scam", ["fraud_signals"]
        if safety.is_spam:
            # Unwanted marketing is still a "promotion" unless it is a cold
            # solicitation - a recorded sales call or a pitch the user has
            # already written off - which is what the labels call spam.
            if self._is_cold_solicitation(message, content):
                return "spam", ["cold_solicitation"]
            return "promotion", ["unwanted_marketing"]

        biz = self.ds.businesses.get(message.business_id) if message.business_id else None
        group = self.ds.groups.get(message.group_id) if message.group_id else None
        low = content.low

        # "Good morning beta, call me later when free, nothing urgent" is a
        # greeting that happens to contain a verb. A request the sender has
        # already waved off does not turn well-wishing into an action item.
        if content.greeting_hits and content.word_count < 45 and (
            not content.has_direct_request or content.deescalation_hits
        ):
            return "greeting", ["greeting_language"]

        if message.forwarded_count >= 3 and (content.forward_hits or not content.has_direct_request):
            return "forward", ["forward_chain"]

        promo = len(content.caption_promo_hits)
        payment = len(content.caption_payment_hits)
        event = len(content.caption_event_hits)
        biz_update = len(content.caption_business_update_hits)

        # Peer-to-peer selling is an offer even though it arrives as chat. A
        # neighbour writing "recliner, discount today, cash or UPI, pickup" is
        # advertising, not talking.
        if not message.is_business and (
            content.caption_marketplace_hits
            or (promo and content.commerce_hits and not content.media_text)
        ):
            return "promotion", ["peer_to_peer_offer"]

        if promo >= 2 or (promo >= 1 and content.has_opt_out_footer):
            drivers.append("promotional_language")
            return "promotion", drivers
        if content.image_layout == "promotional_poster" and not biz_update:
            return "promotion", ["promotional_poster"]

        if payment >= 2 or (payment >= 1 and any(
            k in low for k in ("due", "pay by", "outstanding", "bill", "emi", "challan")
        )):
            return "payment", ["payment_language"]

        # School and society operations are scheduled activities even when they
        # arrive with same-day pressure, so they read as events, not alarms.
        operational_group = bool(group and group.group_type in SCHOOL_GROUP_TYPES)
        if operational_group and (event or content.time_pressure_hits):
            return "event", ["school_operations"]

        # An admin broadcasting a same-day operational problem (water, power,
        # safety) is urgent even though it asks nothing of the reader directly.
        if (
            group
            and group.group_type in OPERATIONAL_GROUP_TYPES
            and self.ds.sender_is_admin(message.group_id, message.sender_user_id)
            and content.time_pressure_hits
            and urgency >= 0.35
        ):
            return "urgent", ["admin_operational_alert"]

        if urgency >= 0.55 and (content.has_direct_request or content.urgency_hits):
            return "urgent", ["time_critical"]

        if event >= 1 and (content.time_pressure_hits or event >= 2):
            return "event", ["scheduled_activity"]

        if biz is not None or message.is_business:
            if biz_update >= 1:
                return "business_update", ["service_update"]
            if promo >= 1:
                return "promotion", ["promotional_language"]
            return "business_update", ["business_sender"]

        if message.is_personal or message.sender_user_id:
            if content.has_direct_request and urgency >= 0.45:
                return "urgent", ["direct_time_bound_request"]
            if content.is_empty:
                return "unknown", ["no_recoverable_content"]
            # An approach from someone with no shared history is not yet a
            # personal relationship; the reference labels call this unknown.
            if message.is_personal and self._is_stranger(message):
                return "unknown", ["no_prior_contact"]
            return "personal", ["interpersonal_message"]

        return "unknown", ["insufficient_signal"]

    def _is_cold_solicitation(self, message: Message, content: ContentView) -> bool:
        if content.modality == "voice" and content.voice_duration_s >= 20:
            return True
        rel = (
            self.ds.relationship(message.user_id, message.business_id)
            if message.business_id else None
        )
        if rel and rel.why_user_knows_account.lower().startswith(("ignored", "cold", "unknown")):
            return True
        return False

    def _is_stranger(self, message: Message) -> bool:
        if not message.sender_user_id:
            return False
        prior = self.ds.history_by_user.get(message.user_id, [])
        return not any(h.sender_user_id == message.sender_user_id for h in prior)


class RoutingEngine:
    """Fuses every signal into a final action, type, and calibrated confidence."""

    def __init__(self, settings: Settings, dataset, memory: MemoryStore) -> None:
        self.settings = settings
        self.ds = dataset
        self.memory = memory
        self.weights: FusionWeights = settings.weights
        self.thresholds: RoutingThresholds = settings.thresholds
        self.bands: ConfidenceBands = settings.confidence
        self.urgency_engine = UrgencyEngine(dataset)
        self.classifier = TypeClassifier(dataset)

    def trust_score(self, message: Message, content: ContentView, safety: SafetyVerdict) -> tuple[float, list[str]]:
        drivers: list[str] = []
        trust = 0.5

        if message.business_id:
            biz = self.ds.businesses.get(message.business_id)
            if biz:
                if biz.verified:
                    trust += 0.22
                    drivers.append("verified_business")
                else:
                    trust -= 0.12
                if biz.domain_mismatch:
                    trust -= 0.34
                    drivers.append("sender_domain_mismatch")
                if biz.account_age_days >= 730:
                    trust += 0.08
                elif biz.account_age_days < 120:
                    trust -= 0.10
                    drivers.append("new_business_account")
                if biz.user_reports_30d >= 10:
                    trust -= 0.14
                    drivers.append("business_widely_reported")
                rel = self.ds.relationship(message.user_id, message.business_id)
                if rel and rel.has_active_relationship:
                    trust += 0.16
                    drivers.append("known_counterparty")

        if message.group_id:
            if self.ds.sender_is_admin(message.group_id, message.sender_user_id):
                trust += 0.14
                drivers.append("group_admin_sender")
            membership = self.ds.membership(message.group_id, message.sender_user_id)
            if membership is None and message.sender_user_id:
                trust -= 0.08

        if message.sender_user_id:
            stats = self.memory.profile(message.user_id).sender(message.sender_user_id)
            if stats.seen >= 2:
                trust += 0.18 * stats.affinity

        trust -= 0.45 * safety.risk
        return round(max(0.0, min(1.0, trust)), 3), drivers

    def topic_importance(self, message: Message, content: ContentView, mtype: str) -> float:
        base = {
            "urgent": 0.92, "payment": 0.70, "event": 0.62, "business_update": 0.55,
            "personal": 0.50, "unknown": 0.35, "forward": 0.20, "greeting": 0.18,
            "promotion": 0.22, "spam": 0.05, "scam": 0.0,
        }.get(mtype, 0.35)
        if message.group_id:
            group = self.ds.groups.get(message.group_id)
            if group and group.group_type in SCHOOL_GROUP_TYPES | OPERATIONAL_GROUP_TYPES:
                base += 0.06

        # A message about something the user actually has in flight - a live
        # order, a booking they made - is consequential in a way that generic
        # brand mail is not, even though both come from the same account.
        if message.business_id and mtype in ("business_update", "event", "payment"):
            rel = self.ds.relationship(message.user_id, message.business_id)
            if rel and rel.has_active_relationship:
                base += 0.24
                if content.business_update_hits or content.event_hits:
                    base += 0.08
        return round(min(1.0, base), 3)

    def score(
        self,
        message: Message,
        content: ContentView,
        safety: SafetyVerdict,
        repetition: float,
        repetition_drivers: list[str],
    ) -> SignalBundle:
        bundle = SignalBundle()
        drivers: list[str] = []

        urgency, actionability, u_drivers = self.urgency_engine.score(message, content)
        bundle.urgency = urgency
        bundle.actionability = actionability
        drivers.extend(u_drivers)

        mtype, t_drivers = self.classifier.classify(message, content, safety, urgency)
        drivers.extend(t_drivers)

        trust, tr_drivers = self.trust_score(message, content, safety)
        bundle.trust = trust
        drivers.extend(tr_drivers)

        topic = {"promotion": "promotion", "spam": "promotion", "greeting": "greeting",
                 "forward": "forward", "payment": "payment", "event": "event",
                 "business_update": "logistics", "scam": "security_prompt"}.get(mtype, "other")
        relationship, r_drivers = self.memory.relationship_score(message, topic)
        bundle.relationship = relationship
        drivers.extend(r_drivers)

        bundle.topic_importance = self.topic_importance(message, content, mtype)
        bundle.transactional = self._is_transactional(message, content, mtype)
        bundle.repetition = repetition
        drivers.extend(repetition_drivers)
        bundle.fatigue = self.memory.fatigue_penalty(message.user_id)
        bundle.scam = safety.scam_score
        bundle.spam = safety.spam_score
        bundle.injection = safety.injection_score
        bundle.promo_pressure = min(1.0, len(content.promo_hits) / 4.0)

        user = self.ds.users.get(message.user_id)
        bundle.quiet_hours = bool(user and user.is_quiet_hour(message.created_at))
        if bundle.quiet_hours:
            drivers.append("inside_quiet_hours")

        w = self.weights
        priority = (
            w.urgency * bundle.urgency
            + w.trust * bundle.trust
            + w.relationship * bundle.relationship
            + w.actionability * bundle.actionability
            + w.topic_importance * bundle.topic_importance
        )
        priority -= w.repetition_penalty * bundle.repetition
        priority -= w.fatigue_penalty * bundle.fatigue
        priority -= w.promo_penalty * bundle.promo_pressure
        if bundle.quiet_hours:
            priority -= w.quiet_hours_penalty
        if message.forwarded_count >= 8:
            # Scales with how far the chain has travelled, capped so a genuine
            # forwarded alert from a trusted admin can still surface.
            priority -= w.forward_chain_penalty * min(1.0, message.forwarded_count / 20.0)
            drivers.append("mass_forwarded_chain")

        bundle.priority = round(max(0.0, min(1.0, priority)), 4)
        bundle.drivers = drivers
        return bundle

    def _is_transactional(self, message: Message, content: ContentView, mtype: str) -> bool:
        """The message concerns a specific thing this user already has in flight.

        A delivery status or booking reminder is worth an interruption; a
        satisfaction survey from the same verified brand is not, even though
        both are business updates from an account the user knows.
        """
        if mtype not in ("business_update", "payment", "event") or not message.business_id:
            return False
        rel = self.ds.relationship(message.user_id, message.business_id)
        if not (rel and rel.has_active_relationship):
            return False
        return bool(content.business_update_hits or content.payment_hits or content.event_hits)

    def _warrants_interruption(self, bundle: SignalBundle) -> bool:
        """An interruption needs a reason, not just a well liked sender.

        Liking someone is why their message is worth *reading*; it is not why it
        has to be read *now*. Requiring a time-bound element, a direct ask, or a
        live transaction keeps warm-but-idle chatter out of the notify lane.
        """
        th = self.thresholds
        return (
            bundle.urgency >= th.interrupt_urgency_gate
            or bundle.actionability >= th.interrupt_action_gate
            or bundle.transactional
        )

    def decide(self, bundle: SignalBundle, mtype: str, safety: SafetyVerdict) -> tuple[str, str]:
        """Map scores to an action, with safety holding an absolute veto."""
        th = self.thresholds

        if bundle.injection >= th.injection_hard_mute:
            return "mute", "scam"
        if safety.scam_score >= th.scam_hard_mute:
            return "mute", "scam"
        if safety.spam_score >= th.spam_hard_mute and mtype in ("promotion", "spam", "business_update"):
            return "mute", "spam" if mtype == "spam" else mtype

        if bundle.priority >= th.notify_floor and self._warrants_interruption(bundle):
            # Even a wanted, urgent-looking message cannot interrupt if the
            # safety engine sees meaningful risk in it.
            if safety.risk >= th.safety_notify_veto:
                return "digest", mtype
            return "notify", mtype

        if bundle.priority <= th.mute_ceiling:
            return "mute", mtype
        return "digest", mtype

    def confidence(self, action: str, base: float, bundle: SignalBundle) -> float:
        """Nudge the rationale's base confidence within its action band.

        Confidence reflects how cleanly the evidence separated, so a decision
        made far from its threshold earns a little more than a borderline one.
        """
        low, high = getattr(self.bands, action)
        th = self.thresholds
        if action == "notify":
            margin = bundle.priority - th.notify_floor
        elif action == "mute":
            margin = max(bundle.scam, bundle.injection, bundle.spam, th.mute_ceiling - bundle.priority)
        else:
            distance = min(abs(bundle.priority - th.notify_floor), abs(bundle.priority - th.mute_ceiling))
            margin = distance
        adjust = max(-0.02, min(0.02, (margin - 0.12) * 0.12))
        return round(max(low, min(high, base + adjust)), 2)
