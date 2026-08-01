"""Typed records for every dataset entity plus the decision payload.

Typed access keeps the engines honest: a missing column fails loudly at load
time rather than silently producing an empty-string feature at scoring time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    return str(value).strip() in {"1", "true", "True", "yes"}


def _dt(value: Any) -> datetime | None:
    raw = (str(value) or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


@dataclass
class Message:
    message_id: str
    user_id: str
    conversation_type: str
    group_id: str
    business_id: str
    sender_user_id: str
    created_at: datetime | None
    message_text: str
    media_type: str
    media_id: str
    forwarded_count: int

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "Message":
        return cls(
            message_id=row["message_id"].strip(),
            user_id=row["user_id"].strip(),
            conversation_type=row["conversation_type"].strip(),
            group_id=row.get("group_id", "").strip(),
            business_id=row.get("business_id", "").strip(),
            sender_user_id=row.get("sender_user_id", "").strip(),
            created_at=_dt(row.get("created_at")),
            message_text=row.get("message_text", "") or "",
            media_type=row.get("media_type", "").strip(),
            media_id=row.get("media_id", "").strip(),
            forwarded_count=_int(row.get("forwarded_count")),
        )

    @property
    def is_group(self) -> bool:
        return self.conversation_type == "group"

    @property
    def is_business(self) -> bool:
        return self.conversation_type == "business"

    @property
    def is_personal(self) -> bool:
        return self.conversation_type == "personal"

    @property
    def has_media(self) -> bool:
        return bool(self.media_type and self.media_id)


@dataclass
class LabelledMessage(Message):
    """A sample row that already carries the reference decision."""

    action: str = ""
    message_type: str = ""
    reason: str = ""
    confidence: float = 0.0
    evidence_message_ids: str = ""

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "LabelledMessage":
        base = Message.from_row(row)
        return cls(
            **base.__dict__,
            action=row.get("action", "").strip(),
            message_type=row.get("message_type", "").strip(),
            reason=row.get("reason", "").strip(),
            confidence=_float(row.get("confidence")),
            evidence_message_ids=row.get("evidence_message_ids", "").strip(),
        )

    def evidence_ids(self) -> list[str]:
        raw = self.evidence_message_ids.strip()
        if not raw or raw.lower() == "none":
            return []
        return [x.strip() for x in raw.split(";") if x.strip()]


@dataclass
class User:
    user_id: str
    do_not_disturb_window: str
    messages_opened_30d: int
    messages_replied_30d: int
    notifications_dismissed_30d: int
    messages_reported_30d: int

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "User":
        return cls(
            user_id=row["user_id"].strip(),
            do_not_disturb_window=row.get("do_not_disturb_window", "").strip(),
            messages_opened_30d=_int(row.get("messages_opened_30d")),
            messages_replied_30d=_int(row.get("messages_replied_30d")),
            notifications_dismissed_30d=_int(row.get("notifications_dismissed_30d")),
            messages_reported_30d=_int(row.get("messages_reported_30d")),
        )

    def quiet_window(self) -> tuple[time, time] | None:
        raw = self.do_not_disturb_window
        if not raw or "-" not in raw:
            return None
        start_s, _, end_s = raw.partition("-")
        try:
            sh, sm = (int(x) for x in start_s.strip().split(":"))
            eh, em = (int(x) for x in end_s.strip().split(":"))
        except ValueError:
            return None
        return time(sh, sm), time(eh, em)

    def is_quiet_hour(self, when: datetime | None) -> bool:
        window = self.quiet_window()
        if window is None or when is None:
            return False
        start, end = window
        t = when.time()
        if start <= end:
            return start <= t < end
        # Window wraps past midnight (for example 22:00-07:00).
        return t >= start or t < end


@dataclass
class Group:
    group_id: str
    group_name: str
    group_type: str
    member_count: int
    admin_count: int
    created_at: datetime | None
    messages_30d: int

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "Group":
        return cls(
            group_id=row["group_id"].strip(),
            group_name=row.get("group_name", "").strip(),
            group_type=row.get("group_type", "").strip(),
            member_count=_int(row.get("member_count")),
            admin_count=_int(row.get("admin_count")),
            created_at=_dt(row.get("created_at")),
            messages_30d=_int(row.get("messages_30d")),
        )


@dataclass
class GroupMembership:
    group_id: str
    user_id: str
    role: str
    joined_at: datetime | None
    messages_sent_30d: int
    messages_read_30d: int
    replies_sent_30d: int
    notifications_dismissed_30d: int
    group_muted_by_user: bool

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "GroupMembership":
        return cls(
            group_id=row["group_id"].strip(),
            user_id=row["user_id"].strip(),
            role=row.get("role", "").strip(),
            joined_at=_dt(row.get("joined_at")),
            messages_sent_30d=_int(row.get("messages_sent_30d")),
            messages_read_30d=_int(row.get("messages_read_30d")),
            replies_sent_30d=_int(row.get("replies_sent_30d")),
            notifications_dismissed_30d=_int(row.get("notifications_dismissed_30d")),
            group_muted_by_user=_bool(row.get("group_muted_by_user")),
        )

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


@dataclass
class BusinessAccount:
    business_id: str
    display_name: str
    brand_name: str
    category: str
    verified: bool
    official_domain: str
    domain_used_by_sender: str
    account_age_days: int
    messages_sent_30d: int
    user_reports_30d: int
    domain_used_by_sender_age_days: int

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "BusinessAccount":
        return cls(
            business_id=row["business_id"].strip(),
            display_name=row.get("display_name", "").strip(),
            brand_name=row.get("brand_name", "").strip(),
            category=row.get("category", "").strip(),
            verified=_bool(row.get("verified")),
            official_domain=row.get("official_domain", "").strip().lower(),
            domain_used_by_sender=row.get("domain_used_by_sender", "").strip().lower(),
            account_age_days=_int(row.get("account_age_days")),
            messages_sent_30d=_int(row.get("messages_sent_30d")),
            user_reports_30d=_int(row.get("user_reports_30d")),
            domain_used_by_sender_age_days=_int(row.get("domain_used_by_sender_age_days")),
        )

    @property
    def domain_mismatch(self) -> bool:
        """True when the sending domain is not the brand's official domain."""
        if not self.official_domain or not self.domain_used_by_sender:
            return False
        return self.official_domain != self.domain_used_by_sender

    @property
    def brand_root(self) -> str:
        """The distinctive label of the official domain (amazon.in -> amazon)."""
        return self.official_domain.split(".")[0] if self.official_domain else ""

    @property
    def brand_lookalike_domain(self) -> bool:
        """The sender reuses the brand's name on a domain the brand does not own.

        This separates the two very different populations hiding behind a plain
        domain mismatch. `amazon.in -> amazonpay-delivery.in` borrows the brand
        to look legitimate and is impersonation. `thrillophilia.com ->
        link.wame.pro` is an established brand sending through a link service,
        which is ordinary marketing infrastructure. Treating both as fraud
        would mute a large share of legitimate business mail.
        """
        if not self.domain_mismatch:
            return False
        root = self.brand_root
        if len(root) < 3:
            return False
        return root in self.domain_used_by_sender.replace("-", "").replace("_", "")


@dataclass
class UserBusinessHistory:
    user_id: str
    business_id: str
    why_user_knows_account: str
    last_activity_at: datetime | None
    allows_promotions: bool
    promotions_opted_out_at: datetime | None
    activity_count_180d: int
    messages_opened_30d: int
    messages_dismissed_30d: int
    messages_replied_30d: int
    last_reply_at: datetime | None

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "UserBusinessHistory":
        return cls(
            user_id=row["user_id"].strip(),
            business_id=row["business_id"].strip(),
            why_user_knows_account=row.get("why_user_knows_account", "").strip(),
            last_activity_at=_dt(row.get("last_activity_at")),
            allows_promotions=_bool(row.get("allows_promotions")),
            promotions_opted_out_at=_dt(row.get("promotions_opted_out_at")),
            activity_count_180d=_int(row.get("activity_count_180d")),
            messages_opened_30d=_int(row.get("messages_opened_30d")),
            messages_dismissed_30d=_int(row.get("messages_dismissed_30d")),
            messages_replied_30d=_int(row.get("messages_replied_30d")),
            last_reply_at=_dt(row.get("last_reply_at")),
        )

    @property
    def opted_out(self) -> bool:
        return self.promotions_opted_out_at is not None or not self.allows_promotions

    @property
    def has_active_relationship(self) -> bool:
        return self.activity_count_180d > 0 or self.messages_replied_30d > 0


@dataclass
class MessageEvent:
    user_id: str
    message_id: str
    message_opened: bool
    message_replied: bool
    reaction_time_minutes: float
    notification_dismissed: bool
    muted_after_message: bool
    message_reported: bool

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "MessageEvent":
        return cls(
            user_id=row["user_id"].strip(),
            message_id=row["message_id"].strip(),
            message_opened=_bool(row.get("message_opened")),
            message_replied=_bool(row.get("message_replied")),
            reaction_time_minutes=_float(row.get("reaction_time_minutes"), -1.0),
            notification_dismissed=_bool(row.get("notification_dismissed")),
            muted_after_message=_bool(row.get("muted_after_message")),
            message_reported=_bool(row.get("message_reported")),
        )

    @property
    def positive(self) -> bool:
        """User engaged: evidence that a similar message deserved attention."""
        return self.message_opened or self.message_replied

    @property
    def negative(self) -> bool:
        """User rejected: evidence that a similar message should be suppressed."""
        return self.notification_dismissed or self.muted_after_message or self.message_reported


@dataclass
class DailyNotificationLoad:
    user_id: str
    date: str
    notifications_sent: int
    notifications_dismissed: int

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "DailyNotificationLoad":
        return cls(
            user_id=row["user_id"].strip(),
            date=row.get("date", "").strip(),
            notifications_sent=_int(row.get("notifications_sent")),
            notifications_dismissed=_int(row.get("notifications_dismissed")),
        )


@dataclass
class MediaAsset:
    media_id: str
    file_path: str


@dataclass
class Decision:
    """Final routing decision plus the full audit trail behind it."""

    message_id: str
    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: list[str] = field(default_factory=list)

    priority_score: float = 0.0
    signals: dict[str, float] = field(default_factory=dict)
    rationale_key: str = ""
    drivers: list[str] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)

    def evidence_field(self) -> str:
        return ";".join(self.evidence_message_ids) if self.evidence_message_ids else "none"

    def to_output_row(self) -> dict[str, str]:
        return {
            "message_id": self.message_id,
            "action": self.action,
            "message_type": self.message_type,
            "reason": self.reason,
            "confidence": f"{self.confidence:.2f}",
            "evidence_message_ids": self.evidence_field(),
        }
