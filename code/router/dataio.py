"""Dataset loading, indexing, and submission writing."""

from __future__ import annotations

import csv
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .config import ACTIONS, MESSAGE_TYPES, OUTPUT_COLUMNS, Settings
from .schema import (
    BusinessAccount,
    DailyNotificationLoad,
    Decision,
    Group,
    GroupMembership,
    LabelledMessage,
    MediaAsset,
    Message,
    MessageEvent,
    User,
    UserBusinessHistory,
)

log = logging.getLogger(__name__)

# Several message_text fields embed newlines and quotes, so the whole file has
# to go through the csv module rather than any line-oriented shortcut.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


@dataclass
class Dataset:
    """All participant-facing data, loaded once and indexed for O(1) lookup."""

    messages: list[Message] = field(default_factory=list)
    samples: list[LabelledMessage] = field(default_factory=list)
    history: list[Message] = field(default_factory=list)

    users: dict[str, User] = field(default_factory=dict)
    groups: dict[str, Group] = field(default_factory=dict)
    businesses: dict[str, BusinessAccount] = field(default_factory=dict)

    memberships: dict[tuple[str, str], GroupMembership] = field(default_factory=dict)
    business_history: dict[tuple[str, str], UserBusinessHistory] = field(default_factory=dict)
    events: dict[tuple[str, str], MessageEvent] = field(default_factory=dict)

    images: dict[str, MediaAsset] = field(default_factory=dict)
    voice_notes: dict[str, MediaAsset] = field(default_factory=dict)
    daily_load: dict[str, list[DailyNotificationLoad]] = field(default_factory=dict)

    history_by_id: dict[str, Message] = field(default_factory=dict)
    history_by_user: dict[str, list[Message]] = field(default_factory=dict)

    def group_members(self, group_id: str) -> list[GroupMembership]:
        return [m for (gid, _), m in self.memberships.items() if gid == group_id]

    def membership(self, group_id: str, user_id: str) -> GroupMembership | None:
        return self.memberships.get((group_id, user_id))

    def relationship(self, user_id: str, business_id: str) -> UserBusinessHistory | None:
        return self.business_history.get((user_id, business_id))

    def event_for(self, user_id: str, message_id: str) -> MessageEvent | None:
        return self.events.get((user_id, message_id))

    def media_path(self, settings: Settings, media_id: str) -> Path | None:
        asset = self.images.get(media_id) or self.voice_notes.get(media_id)
        if asset is None:
            return None
        path = settings.dataset_dir / asset.file_path
        return path if path.exists() else None

    def sender_is_admin(self, group_id: str, sender_user_id: str) -> bool:
        membership = self.membership(group_id, sender_user_id)
        return bool(membership and membership.is_admin)


def load_dataset(settings: Settings) -> Dataset:
    root = settings.dataset_dir
    if not root.exists():
        raise FileNotFoundError(f"dataset directory not found: {root}")

    ds = Dataset()
    ds.messages = [Message.from_row(r) for r in read_csv(root / "messages.csv")]
    ds.history = [Message.from_row(r) for r in read_csv(root / "message_history.csv")]

    sample_path = root / "sample_messages.csv"
    if sample_path.exists():
        ds.samples = [LabelledMessage.from_row(r) for r in read_csv(sample_path)]

    ds.users = {u.user_id: u for u in (User.from_row(r) for r in read_csv(root / "users.csv"))}
    ds.groups = {g.group_id: g for g in (Group.from_row(r) for r in read_csv(root / "groups.csv"))}
    ds.businesses = {
        b.business_id: b
        for b in (BusinessAccount.from_row(r) for r in read_csv(root / "business_accounts.csv"))
    }

    for row in read_csv(root / "group_members.csv"):
        m = GroupMembership.from_row(row)
        ds.memberships[(m.group_id, m.user_id)] = m

    for row in read_csv(root / "user_business_history.csv"):
        h = UserBusinessHistory.from_row(row)
        ds.business_history[(h.user_id, h.business_id)] = h

    for row in read_csv(root / "message_events.csv"):
        e = MessageEvent.from_row(row)
        ds.events[(e.user_id, e.message_id)] = e

    for row in read_csv(root / "images.csv"):
        ds.images[row["image_id"].strip()] = MediaAsset(row["image_id"].strip(), row["file_path"].strip())

    for row in read_csv(root / "voice_notes.csv"):
        ds.voice_notes[row["voice_note_id"].strip()] = MediaAsset(
            row["voice_note_id"].strip(), row["file_path"].strip()
        )

    for row in read_csv(root / "daily_notification_summary.csv"):
        d = DailyNotificationLoad.from_row(row)
        ds.daily_load.setdefault(d.user_id, []).append(d)

    ds.history_by_id = {m.message_id: m for m in ds.history}
    for m in ds.history:
        ds.history_by_user.setdefault(m.user_id, []).append(m)

    log.info(
        "loaded dataset: %d messages, %d history, %d samples, %d users, %d groups, %d businesses",
        len(ds.messages),
        len(ds.history),
        len(ds.samples),
        len(ds.users),
        len(ds.groups),
        len(ds.businesses),
    )
    return ds


def write_output(path: Path, decisions: Iterable[Decision]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [d.to_output_row() for d in decisions]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = ["VALID" if self.ok else "INVALID"]
        lines += [f"  ERROR   {e}" for e in self.errors]
        lines += [f"  WARNING {w}" for w in self.warnings]
        return "\n".join(lines)


def validate_output(path: Path, expected_ids: list[str]) -> ValidationReport:
    """Enforce the submission contract before anything is shipped."""
    errors: list[str] = []
    warnings: list[str] = []

    if not path.exists():
        return ValidationReport(False, [f"output file missing: {path}"])

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return ValidationReport(False, ["output file is empty"])
        rows = list(reader)

    if tuple(h.strip() for h in header) != OUTPUT_COLUMNS:
        errors.append(f"header mismatch: got {header}, want {list(OUTPUT_COLUMNS)}")

    got_ids = [r[0] for r in rows if r]
    if len(got_ids) != len(expected_ids):
        errors.append(f"row count {len(got_ids)} != expected {len(expected_ids)}")

    missing = set(expected_ids) - set(got_ids)
    extra = set(got_ids) - set(expected_ids)
    if missing:
        errors.append(f"{len(missing)} message_id(s) missing, e.g. {sorted(missing)[:5]}")
    if extra:
        errors.append(f"{len(extra)} unexpected message_id(s), e.g. {sorted(extra)[:5]}")
    if len(got_ids) != len(set(got_ids)):
        errors.append("duplicate message_id rows present")
    if got_ids != expected_ids:
        warnings.append("row order differs from messages.csv")

    for row in rows:
        if len(row) != len(OUTPUT_COLUMNS):
            errors.append(f"row {row[:1]} has {len(row)} fields, want {len(OUTPUT_COLUMNS)}")
            continue
        mid, action, mtype, reason, conf, evidence = row
        if action not in ACTIONS:
            errors.append(f"{mid}: bad action {action!r}")
        if mtype not in MESSAGE_TYPES:
            errors.append(f"{mid}: bad message_type {mtype!r}")
        if not reason.strip():
            errors.append(f"{mid}: empty reason")
        try:
            c = float(conf)
            if not 0.0 <= c <= 1.0:
                errors.append(f"{mid}: confidence {c} out of range")
        except ValueError:
            errors.append(f"{mid}: non-numeric confidence {conf!r}")
        if not evidence.strip():
            errors.append(f"{mid}: evidence must be ids or 'none'")

    return ValidationReport(not errors, errors, warnings)
