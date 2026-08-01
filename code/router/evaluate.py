"""Evaluation against the labelled sample rows.

Reports more than accuracy, because accuracy alone hides the errors that matter.
A router that mutes an urgent message and one that digests a promotion can post
the same score while being very differently wrong, so safety-critical error
classes are counted separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import ACTIONS, MESSAGE_TYPES
from .schema import Decision, LabelledMessage

# Errors weighted by real-world cost rather than by frequency.
CRITICAL_ERRORS = {
    ("notify", "mute"): "urgent_message_muted",
    ("mute", "notify"): "unsafe_message_notified",
}


@dataclass
class EvalResult:
    n: int = 0
    action_correct: int = 0
    type_correct: int = 0
    both_correct: int = 0
    adjacent: int = 0                 # off by one step on the notify/digest/mute scale

    action_confusion: dict[tuple[str, str], int] = field(default_factory=dict)
    type_confusion: dict[tuple[str, str], int] = field(default_factory=dict)
    critical: dict[str, int] = field(default_factory=dict)

    evidence_expected: int = 0
    evidence_exact: int = 0
    evidence_overlap: int = 0
    evidence_none_correct: int = 0

    confidence_error: float = 0.0
    reason_exact: int = 0
    mistakes: list[str] = field(default_factory=list)

    @property
    def action_accuracy(self) -> float:
        return self.action_correct / self.n if self.n else 0.0

    @property
    def type_accuracy(self) -> float:
        return self.type_correct / self.n if self.n else 0.0

    @property
    def joint_accuracy(self) -> float:
        return self.both_correct / self.n if self.n else 0.0

    @property
    def evidence_recall(self) -> float:
        return self.evidence_overlap / self.evidence_expected if self.evidence_expected else 0.0

    @property
    def mean_confidence_error(self) -> float:
        return self.confidence_error / self.n if self.n else 0.0

    @property
    def critical_error_count(self) -> int:
        return sum(self.critical.values())

    def render(self, verbose: bool = False) -> str:
        lines = [
            f"n = {self.n}",
            f"  action accuracy      {self.action_accuracy:6.1%}  ({self.action_correct}/{self.n})",
            f"  message_type accuracy{self.type_accuracy:6.1%}  ({self.type_correct}/{self.n})",
            f"  joint accuracy       {self.joint_accuracy:6.1%}  ({self.both_correct}/{self.n})",
            f"  adjacent-only errors {self.adjacent}",
            f"  CRITICAL errors      {self.critical_error_count}"
            + (f"  {dict(self.critical)}" if self.critical else ""),
            f"  reason exact match   {self.reason_exact}/{self.n}",
            f"  evidence recall      {self.evidence_recall:6.1%}"
            f"  (exact {self.evidence_exact}/{self.evidence_expected},"
            f" none-correct {self.evidence_none_correct})",
            f"  mean |conf error|    {self.mean_confidence_error:.3f}",
        ]
        lines.append("  action confusion (true -> pred):")
        for true in ACTIONS:
            row = "    " + f"{true:7s}"
            for pred in ACTIONS:
                row += f"  {pred[:3]}={self.action_confusion.get((true, pred), 0):3d}"
            lines.append(row)
        if verbose and self.mistakes:
            lines.append("  mistakes:")
            lines.extend(f"    {m}" for m in self.mistakes)
        return "\n".join(lines)


def evaluate(decisions: list[Decision], truth: list[LabelledMessage], verbose: bool = False) -> EvalResult:
    by_id = {d.message_id: d for d in decisions}
    result = EvalResult()
    order = {"mute": 0, "digest": 1, "notify": 2}

    for gold in truth:
        pred = by_id.get(gold.message_id)
        if pred is None:
            continue
        result.n += 1

        action_ok = pred.action == gold.action
        type_ok = pred.message_type == gold.message_type
        result.action_correct += int(action_ok)
        result.type_correct += int(type_ok)
        result.both_correct += int(action_ok and type_ok)

        result.action_confusion[(gold.action, pred.action)] = (
            result.action_confusion.get((gold.action, pred.action), 0) + 1
        )
        result.type_confusion[(gold.message_type, pred.message_type)] = (
            result.type_confusion.get((gold.message_type, pred.message_type), 0) + 1
        )

        if not action_ok:
            if abs(order[gold.action] - order[pred.action]) == 1:
                result.adjacent += 1
            label = CRITICAL_ERRORS.get((gold.action, pred.action))
            if label:
                result.critical[label] = result.critical.get(label, 0) + 1
            result.mistakes.append(
                f"{gold.message_id}: action {gold.action}->{pred.action} "
                f"type {gold.message_type}->{pred.message_type} "
                f"[{pred.rationale_key}] prio={pred.priority_score:.2f}"
            )
        elif not type_ok:
            result.mistakes.append(
                f"{gold.message_id}: type {gold.message_type}->{pred.message_type} "
                f"(action {gold.action} ok) [{pred.rationale_key}]"
            )

        gold_ev = set(gold.evidence_ids())
        pred_ev = set(pred.evidence_message_ids)
        if gold_ev:
            result.evidence_expected += len(gold_ev)
            result.evidence_overlap += len(gold_ev & pred_ev)
            if gold_ev == pred_ev:
                result.evidence_exact += 1
        elif not pred_ev:
            result.evidence_none_correct += 1

        result.confidence_error += abs(pred.confidence - gold.confidence)
        result.reason_exact += int(pred.reason.strip() == gold.reason.strip())

    return result


def calibration_report(decisions: list[Decision], truth: list[LabelledMessage], bins: int = 5) -> str:
    """Reliability table: does stated confidence track observed accuracy?"""
    by_id = {d.message_id: d for d in decisions}
    buckets: dict[int, list[tuple[float, bool]]] = {}
    for gold in truth:
        pred = by_id.get(gold.message_id)
        if pred is None:
            continue
        idx = min(bins - 1, int(pred.confidence * bins))
        buckets.setdefault(idx, []).append((pred.confidence, pred.action == gold.action))

    lines = ["confidence calibration:"]
    ece = 0.0
    total = sum(len(v) for v in buckets.values()) or 1
    for idx in sorted(buckets):
        rows = buckets[idx]
        mean_conf = sum(c for c, _ in rows) / len(rows)
        acc = sum(1 for _, ok in rows if ok) / len(rows)
        ece += len(rows) / total * abs(mean_conf - acc)
        lo, hi = idx / bins, (idx + 1) / bins
        lines.append(f"  [{lo:.1f},{hi:.1f})  n={len(rows):3d}  conf={mean_conf:.3f}  acc={acc:.3f}")
    lines.append(f"  expected calibration error: {ece:.4f}")
    return "\n".join(lines)
