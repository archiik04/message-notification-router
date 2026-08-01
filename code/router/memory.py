"""User memory and personalization.

The router's core claim is that routing is a property of the *pair* (message,
user), not of the message alone. This module builds the user side of that pair:
a behavioural profile assembled from how the person has actually treated
similar senders, groups and businesses in the past.

Every score here is derived from observed events, never from a hand-written
preference list, so the model adapts to a user we have never seen before.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .schema import Message, MessageEvent


def _rate(numerator: float, denominator: float, prior: float = 0.5, strength: float = 3.0) -> float:
    """Smoothed rate that stays sane on tiny sample sizes.

    A sender with one opened message should not read as a perfectly trusted
    contact, so observations are blended toward a neutral prior until enough
    evidence accumulates.
    """
    return (numerator + prior * strength) / (denominator + strength)


@dataclass
class EngagementStats:
    seen: int = 0
    opened: int = 0
    replied: int = 0
    dismissed: int = 0
    muted: int = 0
    reported: int = 0
    fast_reactions: int = 0

    def add(self, event: MessageEvent) -> None:
        self.seen += 1
        self.opened += int(event.message_opened)
        self.replied += int(event.message_replied)
        self.dismissed += int(event.notification_dismissed)
        self.muted += int(event.muted_after_message)
        self.reported += int(event.message_reported)
        if 0 <= event.reaction_time_minutes <= 5:
            self.fast_reactions += 1

    @property
    def open_rate(self) -> float:
        return _rate(self.opened, self.seen)

    @property
    def reply_rate(self) -> float:
        return _rate(self.replied, self.seen, prior=0.25)

    @property
    def rejection_rate(self) -> float:
        return _rate(self.dismissed + self.muted + self.reported, self.seen, prior=0.35)

    @property
    def affinity(self) -> float:
        """-1 (actively rejected) .. +1 (actively engaged)."""
        if self.seen == 0:
            return 0.0
        positive = self.opened + 1.5 * self.replied + 0.5 * self.fast_reactions
        negative = self.dismissed + 1.5 * self.muted + 3.0 * self.reported
        total = positive + negative
        if total == 0:
            return 0.0
        raw = (positive - negative) / total
        # Confidence grows with sample size so one observation cannot swing the
        # profile to an extreme.
        return round(raw * (1.0 - math.exp(-self.seen / 2.5)), 3)


@dataclass
class UserProfile:
    """Everything learned about how one person handles notifications."""

    user_id: str
    overall: EngagementStats = field(default_factory=EngagementStats)
    by_sender: dict[str, EngagementStats] = field(default_factory=dict)
    by_group: dict[str, EngagementStats] = field(default_factory=dict)
    by_business: dict[str, EngagementStats] = field(default_factory=dict)
    by_topic: dict[str, EngagementStats] = field(default_factory=dict)

    notification_load: float = 0.0     # mean notifications per day
    dismissal_ratio: float = 0.0       # dismissed / sent across the summary window
    report_propensity: float = 0.0     # how readily this user reports messages
    engagement_baseline: float = 0.5   # open-rate baseline from users.csv

    def sender(self, sender_id: str) -> EngagementStats:
        return self.by_sender.get(sender_id, EngagementStats())

    def group(self, group_id: str) -> EngagementStats:
        return self.by_group.get(group_id, EngagementStats())

    def business(self, business_id: str) -> EngagementStats:
        return self.by_business.get(business_id, EngagementStats())

    def topic(self, topic: str) -> EngagementStats:
        return self.by_topic.get(topic, EngagementStats())

    @property
    def is_fatigued(self) -> bool:
        """A user drowning in dismissed notifications deserves a higher bar."""
        return self.dismissal_ratio >= 0.5 and self.notification_load >= 2.0


def _topic_of(message: Message) -> str:
    """Coarse topic bucket used to generalise across senders.

    Deliberately crude: its only job is to let engagement with one promotional
    message inform the handling of the next one.
    """
    text = (message.message_text or "").lower()
    if any(k in text for k in ("otp", "verify", "blocked", "kyc", "password", "pin")):
        return "security_prompt"
    if any(k in text for k in ("off", "sale", "offer", "discount", "cashback", "deal", "coupon")):
        return "promotion"
    if any(k in text for k in ("order", "delivery", "shipped", "tracking", "dispatch")):
        return "logistics"
    if any(k in text for k in ("meeting", "appointment", "booking", "circular", "trip", "event", "class")):
        return "event"
    if any(k in text for k in ("good morning", "blessed", "stay positive", "god bless")):
        return "greeting"
    if any(k in text for k in ("bill", "due", "payment", "emi", "statement", "maintenance")):
        return "payment"
    if message.forwarded_count >= 3:
        return "forward"
    return "other"


class MemoryStore:
    """Builds and serves per-user behavioural profiles."""

    def __init__(self, dataset) -> None:
        self.ds = dataset
        self.profiles: dict[str, UserProfile] = {}
        self._build()

    def _build(self) -> None:
        for user_id in self.ds.users:
            self.profiles[user_id] = UserProfile(user_id=user_id)

        for (user_id, message_id), event in self.ds.events.items():
            profile = self.profiles.setdefault(user_id, UserProfile(user_id=user_id))
            hist = self.ds.history_by_id.get(message_id)
            profile.overall.add(event)
            if hist is None:
                continue
            if hist.sender_user_id:
                profile.by_sender.setdefault(hist.sender_user_id, EngagementStats()).add(event)
            if hist.group_id:
                profile.by_group.setdefault(hist.group_id, EngagementStats()).add(event)
            if hist.business_id:
                profile.by_business.setdefault(hist.business_id, EngagementStats()).add(event)
            profile.by_topic.setdefault(_topic_of(hist), EngagementStats()).add(event)

        for user_id, profile in self.profiles.items():
            loads = self.ds.daily_load.get(user_id, [])
            if loads:
                sent = sum(d.notifications_sent for d in loads)
                dismissed = sum(d.notifications_dismissed for d in loads)
                profile.notification_load = round(sent / len(loads), 3)
                profile.dismissal_ratio = round(dismissed / sent, 3) if sent else 0.0

            user = self.ds.users.get(user_id)
            if user:
                total = user.messages_opened_30d + user.notifications_dismissed_30d
                profile.engagement_baseline = (
                    round(user.messages_opened_30d / total, 3) if total else 0.5
                )
                profile.report_propensity = (
                    round(user.messages_reported_30d / max(1, user.messages_opened_30d), 4)
                )

    def profile(self, user_id: str) -> UserProfile:
        return self.profiles.setdefault(user_id, UserProfile(user_id=user_id))

    # ------------------------------------------------------------------
    def relationship_score(self, message: Message, topic: str) -> tuple[float, list[str]]:
        """How much this user cares about this sender, on a 0..1 scale.

        Returns the score plus the human-readable drivers behind it so the
        explanation layer can cite concrete behaviour rather than a bare number.
        """
        profile = self.profile(message.user_id)
        drivers: list[str] = []
        score = 0.5

        if message.is_group and message.group_id:
            membership = self.ds.membership(message.group_id, message.user_id)
            group = self.ds.groups.get(message.group_id)
            gstats = profile.group(message.group_id)

            if membership:
                if membership.group_muted_by_user:
                    score -= 0.20
                    drivers.append("user_muted_group")
                read_ratio = _rate(
                    membership.messages_read_30d,
                    max(membership.messages_read_30d + membership.notifications_dismissed_30d, 1),
                )
                score += 0.24 * (read_ratio - 0.5) * 2
                if membership.replies_sent_30d >= 3:
                    score += 0.10
                    drivers.append("user_replies_in_group")
                if membership.notifications_dismissed_30d >= 5 and membership.replies_sent_30d == 0:
                    score -= 0.14
                    drivers.append("user_dismisses_group")
                if membership.is_admin:
                    score += 0.04

            if gstats.seen:
                score += 0.18 * gstats.affinity

            # Group purpose is a real prior: a school or society channel carries
            # operational consequences that a hobby group does not.
            if group:
                score += {
                    "family": 0.10, "school_group": 0.12, "coworker": 0.10, "society": 0.06,
                    "caregiving": 0.12, "safety": 0.10, "extended_family": 0.02,
                    "investment_tips": -0.16, "marketplace": -0.08, "local_food": -0.06,
                    "tech_community": -0.04, "book_club": -0.04, "alumni": -0.04,
                }.get(group.group_type, 0.0)

        if message.sender_user_id:
            sstats = profile.sender(message.sender_user_id)
            if sstats.seen:
                score += 0.30 * sstats.affinity
                if sstats.affinity <= -0.4:
                    drivers.append("sender_usually_ignored")
                elif sstats.affinity >= 0.4:
                    drivers.append("sender_usually_engaged")
            else:
                drivers.append("unknown_sender")
                score -= 0.06

        if message.business_id:
            rel = self.ds.relationship(message.user_id, message.business_id)
            bstats = profile.business(message.business_id)
            if rel:
                if rel.has_active_relationship:
                    score += 0.18
                    drivers.append("active_business_relationship")
                # A marketing opt-out is consent withdrawn for marketing only.
                # It must not suppress the delivery, booking or payment updates
                # the same account sends about things the user actually bought.
                if rel.opted_out and topic in ("promotion", "greeting"):
                    score -= 0.26
                    drivers.append("opted_out_of_promotions")
                if rel.messages_dismissed_30d >= 3 and rel.messages_opened_30d <= 1:
                    score -= 0.18
                    drivers.append("business_messages_dismissed")
                if rel.messages_replied_30d >= 1:
                    score += 0.10
            else:
                score -= 0.12
                drivers.append("no_business_relationship")
            if bstats.seen:
                score += 0.16 * bstats.affinity

        tstats = profile.topic(topic)
        if tstats.seen >= 2:
            score += 0.12 * tstats.affinity
            if tstats.affinity <= -0.45:
                drivers.append(f"user_rejects_{topic}")

        return round(max(0.0, min(1.0, score)), 3), drivers

    def fatigue_penalty(self, user_id: str) -> float:
        """Extra restraint for users who dismiss most of what they receive."""
        profile = self.profile(user_id)
        if not profile.is_fatigued:
            return 0.0
        return round(min(1.0, (profile.dismissal_ratio - 0.5) * 2.0), 3)

    def repetition_score(
        self, message: Message, similar: list[tuple[Message, float]]
    ) -> tuple[float, list[str]]:
        """How strongly this looks like something the user has already rejected.

        `similar` is the retrieval layer's ranked history, so this measures
        rejection of *near-duplicate* content rather than of the sender overall.
        """
        if not similar:
            return 0.0, []
        rejected, engaged, drivers = 0, 0, []
        for hist, sim in similar:
            event = self.ds.event_for(message.user_id, hist.message_id)
            if event is None or sim < 0.30:
                continue
            if event.negative:
                rejected += 1
            elif event.positive:
                engaged += 1
        total = rejected + engaged
        if total == 0:
            return 0.0, []
        score = rejected / total
        if rejected >= 2:
            drivers.append("repeated_pattern_rejected")
        elif rejected == 1 and engaged == 0:
            drivers.append("similar_message_rejected")
        return round(score, 3), drivers
