"""Language-agnostic fallback for text the lexicons cannot read.

The keyword engines are English-first with romanised-Hindi coverage. That is
enough for the sample data but leaves a real hole: a Bengali or Tamil message
saying "father is in hospital, come now" scores exactly zero on every lexicon,
so an emergency degrades to `digest`.

This module closes that hole with a multilingual sentence encoder, comparing the
message against a handful of exemplar intents in English. The encoder maps all
50+ supported languages into one space, so the exemplars do not need
translating.

Deliberately scoped as a *fallback*: it only runs when the lexicons found
nothing, or when the text is largely non-Latin. The English path is tuned and
measured, and this must not perturb it. Loading is lazy, so a run whose messages
are all handled lexically never pays the model-load cost.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Exemplars are written in English on purpose: the encoder is cross-lingual, so
# one set covers every language it supports.
EXEMPLARS: dict[str, tuple[str, ...]] = {
    "urgent": (
        "my father is in hospital, please come immediately",
        "call me right now, it is an emergency",
        "the deadline is today, you must act now",
        "come quickly, something has happened",
    ),
    "fraud": (
        "share your OTP or your account will be blocked",
        "verify your bank details at this link immediately",
        "you have won a prize, pay a fee to claim it",
        "your account is suspended, confirm your PIN now",
    ),
    "promotion": (
        "big sale today, flat fifty percent discount, shop now",
        "special offer just for you, limited time only",
    ),
    "benign": (
        "good morning, have a nice day",
        "just sharing some photos from the weekend",
        "thanks, see you later",
        "no hurry, whenever you get time",
    ),
}

_LATIN_RE = re.compile(r"[A-Za-z]")
_WORDY_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def non_latin_ratio(text: str) -> float:
    """Share of letters that are not Latin - a cheap script detector."""
    letters = _WORDY_RE.findall(text)
    if not letters:
        return 0.0
    latin = sum(1 for c in letters if _LATIN_RE.match(c))
    return round(1.0 - latin / len(letters), 3)


class SemanticIntentScorer:
    """Cross-lingual intent similarity, loaded on first genuine need."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._model = None
        self._exemplars = None
        self._failed = False

    def _load(self):
        if self._model is not None or self._failed or not self.enabled:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(MODEL_NAME)
            self._exemplars = {
                intent: self._model.encode(list(texts), convert_to_numpy=True,
                                           normalize_embeddings=True)
                for intent, texts in EXEMPLARS.items()
            }
            log.info("loaded multilingual intent model %s", MODEL_NAME)
        except Exception as exc:  # noqa: BLE001 - optional capability
            log.warning("multilingual scorer unavailable: %s", exc)
            self._failed = True
        return self._model

    def should_run(self, text: str, lexicon_hit_count: int) -> bool:
        """Only for text the keyword layer could not interpret."""
        if not self.enabled or self._failed:
            return False
        if len(text.split()) < 3:
            return False
        return non_latin_ratio(text) >= 0.3 or lexicon_hit_count == 0

    def score(self, text: str) -> dict[str, float]:
        """Similarity to each intent, or an empty dict if unavailable."""
        model = self._load()
        if model is None:
            return {}
        try:
            import numpy as np

            q = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
            return {
                intent: round(float(np.max(q @ emb.T)), 3)
                for intent, emb in self._exemplars.items()
            }
        except Exception as exc:  # noqa: BLE001
            log.warning("semantic scoring failed: %s", exc)
            return {}

    def verdict(self, text: str, margin: float = 0.12) -> tuple[str, float]:
        """Winning intent and its margin over the runner-up.

        A margin requirement keeps genuinely ambiguous text out: three
        near-equal similarities carry no information and should not move any
        decision.
        """
        scores = self.score(text)
        if not scores:
            return "", 0.0
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        (top, top_score), (_, second) = ranked[0], ranked[1]
        gap = top_score - second
        if gap < margin or top_score < 0.35:
            return "", round(gap, 3)
        return top, round(gap, 3)
