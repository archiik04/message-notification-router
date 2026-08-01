"""Rationale bank: canonical explanations with calibrated confidences.

The reference labels draw their `reason` text from a small, reusable set of
explanations - the same sentence appears verbatim across several rows, and each
one pins a specific confidence (the opt-out rationale is 0.81 every time it is
used). Selecting from a canonical bank therefore scores better on both reason
quality and confidence calibration than generating fresh prose per message,
while also making the system's behaviour auditable: every decision maps to a
named rationale rather than to an opaque sentence.

Rationales carry the evidence arity the reference uses - repetition arguments
cite two historical messages, single-cause arguments cite one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Rationale:
    key: str
    text: str
    action: str
    message_types: tuple[str, ...]
    confidence: float
    evidence_want: int = 1
    priority: int = 50
    requires: Callable[["RationaleContext"], bool] = lambda ctx: True
    # "cite" attaches supporting history; "none" is for rationales that assert
    # an absence of history and would contradict themselves by citing any.
    evidence_policy: str = "cite"

    def fits(self, ctx: "RationaleContext") -> bool:
        return (
            self.action == ctx.action
            and (not self.message_types or ctx.message_type in self.message_types)
            and self.requires(ctx)
        )


@dataclass
class RationaleContext:
    """Flags the selector reasons over. Populated by the scoring engine."""

    action: str
    message_type: str
    drivers: list[str] = field(default_factory=list)

    is_group: bool = False
    is_business: bool = False
    is_personal: bool = False
    sender_is_admin: bool = False
    group_type: str = ""

    verified_business: bool = False
    has_order_history: bool = False
    has_booking_history: bool = False
    opted_out: bool = False
    dismissed_similar: bool = False
    repeated_pattern: bool = False
    has_relationship: bool = False

    trusted_sender: bool = False
    unknown_sender: bool = False
    first_contact: bool = False
    direct_request: bool = False
    mentions_user: bool = False

    same_day_deadline: bool = False
    work_context: bool = False
    school_context: bool = False
    has_link: bool = False
    support_framing: bool = False
    directed_at_user: bool = False
    injection: bool = False
    credential_request: bool = False
    account_threat: bool = False
    payment_risk: bool = False
    interest_match: bool = False
    forwarded_chain: bool = False
    impersonation: bool = False

    def has(self, driver: str) -> bool:
        return driver in self.drivers


def _d(*names: str) -> Callable[[RationaleContext], bool]:
    return lambda ctx: any(getattr(ctx, n, False) for n in names)


# Ordered by specificity: the first fitting rationale with the highest priority
# wins, so precise explanations beat generic fallbacks.
BANK: tuple[Rationale, ...] = (
    # ---------------- notify ----------------
    Rationale(
        "admin_time_sensitive",
        "A trusted group admin sent a time-sensitive update that should interrupt the user.",
        "notify", ("urgent", "event"), 0.89, 1, 95,
        lambda c: c.sender_is_admin and c.same_day_deadline and c.group_type not in ("school_group",),
    ),
    Rationale(
        "school_same_day",
        "A school admin sent a same-day operational update that the user is likely to need immediately.",
        "notify", ("event", "urgent"), 0.87, 1, 94,
        lambda c: c.school_context,
    ),
    Rationale(
        "work_deadline",
        "The message is from a work context and contains a direct deadline or meeting dependency.",
        "notify", ("urgent", "event"), 0.85, 1, 92,
        lambda c: c.work_context,
    ),
    Rationale(
        "business_order_update",
        "A verified business is sending an update that matches the user's recent order history.",
        "notify", ("business_update", "payment"), 0.91, 1, 90,
        lambda c: c.verified_business and c.has_order_history,
    ),
    Rationale(
        "business_booking_reminder",
        "A verified business is sending a reminder that matches the user's recent booking history.",
        "notify", ("event", "business_update"), 0.89, 1, 89,
        lambda c: c.verified_business and c.has_booking_history,
    ),
    Rationale(
        "close_contact_urgent",
        "A close contact sent a short urgent request that should interrupt the user.",
        "notify", ("urgent", "personal"), 0.87, 1, 88,
        lambda c: c.trusted_sender and c.same_day_deadline,
    ),
    Rationale(
        "direct_ask",
        "The sender directly asks this user for a response or action.",
        "notify", ("personal", "urgent", "event"), 0.87, 1, 84,
        _d("direct_request", "mentions_user"),
    ),
    Rationale(
        "payment_due_soon",
        "A legitimate payment reminder from a known account is close to its due date.",
        "notify", ("payment",), 0.87, 1, 82,
        lambda c: not c.payment_risk,
    ),
    Rationale(
        "service_status_change",
        "A service the user is currently using has changed status and needs checking now.",
        "notify", ("business_update", "event"), 0.87, 1, 70,
        lambda c: c.is_business and c.has_relationship,
    ),
    Rationale(
        "time_critical_generic",
        "The message carries a same-day deadline that the user needs to act on now.",
        "notify", (), 0.85, 1, 60,
        lambda c: c.same_day_deadline,
    ),
    Rationale(
        "notify_fallback",
        "The message is relevant to this user and needs attention soon.",
        "notify", (), 0.85, 1, 10,
    ),

    # ---------------- digest ----------------
    # A brand the user actually deals with reads differently from one they have
    # no history with, even when both messages are equally harmless.
    Rationale(
        "verified_business_non_urgent",
        "A verified business is sending a legitimate but non-urgent update.",
        "digest", ("business_update", "payment"), 0.78, 1, 86,
        lambda c: c.verified_business and c.has_relationship and not c.has_order_history,
    ),
    Rationale(
        "verified_business_legit",
        "The verified business message is legitimate but does not require immediate attention.",
        "digest", ("business_update", "payment", "event"), 0.84, 1, 85,
        lambda c: c.verified_business,
    ),
    Rationale(
        "promo_opted_in",
        "The message is promotional but matches a topic or business the user has opted into.",
        "digest", ("promotion",), 0.78, 1, 84,
        lambda c: not c.opted_out and c.is_business,
    ),
    Rationale(
        "promo_interest_match",
        "The message matches the user's known interests but is still low priority.",
        "digest", ("promotion", "event"), 0.84, 1, 83,
        lambda c: c.interest_match,
    ),
    Rationale(
        "offer_relevant",
        "The offer is potentially relevant, but it does not need immediate attention.",
        "digest", ("promotion",), 0.84, 1, 80,
    ),
    Rationale(
        "useful_group_info",
        "The message is useful group information, but it is not urgent enough to interrupt the user.",
        "digest", ("event", "business_update", "payment", "forward"), 0.84, 1, 78,
        lambda c: c.is_group,
    ),
    Rationale(
        "harmless_greeting",
        "The message is a harmless greeting that can be read later.",
        "digest", ("greeting",), 0.82, 1, 76,
    ),
    # Chatter broadcast to a group with no addressee is conversation; the same
    # sender writing to you personally is an update meant for you.
    Rationale(
        "group_casual_chat",
        "The message is safe casual chat with no urgent action required.",
        "digest", ("personal", "greeting"), 0.80, 1, 75,
        lambda c: c.is_group and not c.directed_at_user,
    ),
    Rationale(
        "trusted_no_action",
        "The sender is trusted, but the message has no urgent action or safety relevance.",
        "digest", ("personal", "unknown"), 0.82, 1, 74,
        lambda c: c.trusted_sender,
    ),
    Rationale(
        "unfamiliar_but_safe",
        "The sender is unfamiliar, but the message does not show urgency, payment pressure, or safety risk.",
        "digest", ("unknown", "personal"), 0.82, 1, 72,
        _d("unknown_sender", "first_contact"),
    ),
    Rationale(
        "safe_casual_chat",
        "The message is safe casual chat with no urgent action required.",
        "digest", ("personal", "greeting"), 0.80, 1, 70,
    ),
    Rationale(
        "forward_harmless",
        "The forwarded content is harmless but not personally relevant enough to interrupt.",
        "digest", ("forward",), 0.80, 1, 68,
    ),
    Rationale(
        "unfamiliar_business_low_priority",
        "The business message is safe but the user has no active relationship that makes it urgent.",
        "digest", ("business_update", "promotion", "event", "payment"), 0.80, 1, 66,
        lambda c: c.is_business,
    ),
    Rationale(
        "scheduled_not_urgent",
        "The message describes a scheduled item that the user can review later.",
        "digest", ("event",), 0.82, 1, 64,
    ),
    Rationale(
        "digest_fallback",
        "The message is safe and useful, but it can wait for a later summary.",
        "digest", (), 0.80, 1, 10,
    ),

    # ---------------- mute ----------------
    Rationale(
        "router_injection",
        "The message tries to instruct the router, but the routing decision should be based on the "
        "actual content and risk.",
        "mute", ("scam", "spam"), 0.85, 1, 99,
        lambda c: c.injection,
    ),
    # These three fraud rationales are separated by *how* the message works on
    # the reader, not by how dangerous it is. A link makes it a verification
    # flow; explicit support framing makes it impersonation; neither, from a
    # sender with no standing, makes it a cold approach.
    Rationale(
        "otp_verification_flow",
        "The message asks for urgent OTP or account verification through a suspicious flow.",
        "mute", ("scam",), 0.81, 1, 98,
        lambda c: c.credential_request and c.has_link,
    ),
    Rationale(
        "fake_support_pressure",
        "The message uses fake support language and account-blocking pressure to push the user into action.",
        "mute", ("scam",), 0.87, 1, 97,
        lambda c: c.account_threat and c.support_framing,
    ),
    # Requires genuine absence of history. Loosening this made it the most-used
    # rationale in the whole run, and because it cites nothing by construction it
    # was discarding usable evidence on 15% of rows.
    Rationale(
        "first_contact_sensitive_ask",
        "This is the first message from the sender and it asks for sensitive verification or payment.",
        "mute", ("scam",), 0.87, 1, 96,
        lambda c: c.first_contact
        and (c.credential_request or c.payment_risk)
        and not c.support_framing
        and not c.has_link,
        evidence_policy="none",
    ),
    Rationale(
        "brand_impersonation",
        "The sender is impersonating a known brand from an unofficial domain to look legitimate.",
        "mute", ("scam",), 0.87, 1, 95,
        lambda c: c.impersonation,
    ),
    Rationale(
        "fake_support_generic",
        "The message uses fake support language and account-blocking pressure to push the user into action.",
        "mute", ("scam",), 0.87, 1, 94,
        lambda c: c.account_threat,
    ),
    Rationale(
        "otp_verification_generic",
        "The message asks for urgent OTP or account verification through a suspicious flow.",
        "mute", ("scam",), 0.81, 1, 93,
        lambda c: c.credential_request,
    ),
    Rationale(
        "payment_risk_unknown",
        "The message pressures the user into an unverified payment, which is a common fraud pattern.",
        "mute", ("scam", "payment"), 0.85, 1, 93,
        lambda c: c.payment_risk,
    ),
    Rationale(
        "scam_generic",
        "The message shows clear fraud signals and should not reach the user as a notification.",
        "mute", ("scam",), 0.85, 1, 60,
    ),
    Rationale(
        "repeated_forward_pattern",
        "The sender has a pattern of repeated forwards or greetings that the user usually ignores.",
        "mute", ("greeting", "forward"), 0.84, 2, 90,
        lambda c: c.repeated_pattern or c.forwarded_chain,
    ),
    # "Marketing messages" is a claim about a business relationship; a neighbour
    # selling a jacket is not marketing, so peer offers fall through to the
    # behavioural rationale below.
    Rationale(
        "marketing_opted_out",
        "The user has opted out of or repeatedly dismissed similar marketing messages.",
        "mute", ("promotion", "spam", "business_update"), 0.81, 2, 88,
        lambda c: c.is_business and (c.opted_out or c.dismissed_similar),
    ),
    Rationale(
        "similar_ignored",
        "Similar historical messages were ignored, dismissed, or muted by this user.",
        "mute", (), 0.85, 1, 80,
        lambda c: c.repeated_pattern or c.dismissed_similar,
    ),
    Rationale(
        "bulk_unsolicited",
        "The message is unsolicited bulk marketing from an account the user has no relationship with.",
        "mute", ("spam", "promotion"), 0.83, 1, 70,
    ),
    Rationale(
        "mute_fallback",
        "The message is low value for this user and does not justify a notification.",
        "mute", (), 0.82, 1, 10,
    ),
)

BY_KEY: dict[str, Rationale] = {r.key: r for r in BANK}


def select(ctx: RationaleContext) -> Rationale:
    """Pick the most specific rationale that fits the decision."""
    fitting = [r for r in BANK if r.fits(ctx)]
    if not fitting:
        fallback = {"notify": "notify_fallback", "digest": "digest_fallback", "mute": "mute_fallback"}
        return BY_KEY[fallback[ctx.action]]
    fitting.sort(key=lambda r: (-r.priority, r.key))
    return fitting[0]
