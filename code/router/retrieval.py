"""Evidence retrieval over the user's message history.

Evidence is not decoration: the reference labels cite a historical message in 28
of 30 cases, and always one belonging to the *same* user and the *same*
counterparty. Critically, the cited row's recorded outcome agrees with the
decision - `notify` evidence was opened and replied to, `mute` evidence was
dismissed or muted. Retrieval therefore ranks on three axes at once:

  semantic   - is this about the same thing?
  structural - is it the same relationship?
  behavioural- does its outcome actually support the decision being made?

A purely semantic retriever would happily cite a message the user loved as
justification for muting, which is exactly the failure this design avoids.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

from .config import RetrievalConfig, Settings
from .content import ContentView
from .multimodal import MediaIndex
from .schema import Message

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_STOP = frozenset(
    """a an the and or but if then than that this these those is are was were be been being am
    to of in on at by for with from as it its it's you your yours we our us they them their he
    she his her i me my will would can could should shall may might must do does did done have
    has had not no so just now also very please pls thanks thank ok okay""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1]


SAFETY_TYPES = frozenset({"scam", "spam"})


def evidence_intent(action: str, message_type: str) -> str:
    """Which recorded outcomes count as support for this decision.

    Behavioural mutes and safety mutes need opposite evidence. Muting a chain
    forward is justified by the user having dismissed the last one, so the
    citation must show rejection. Muting a scam is justified by the message
    resembling a known threat - and users frequently *did* open the earlier
    scam, which is why it is dangerous. Demanding rejection there would discard
    the most relevant citation available.
    """
    if action == "notify":
        return "promote"
    if action == "mute":
        return "neutral" if message_type in SAFETY_TYPES else "suppress"
    return "neutral"


@dataclass
class EvidenceCandidate:
    message: Message
    semantic: float
    structural: float
    behavioural: float
    total: float
    outcome: str  # positive | negative | neutral | unknown


class EvidenceRetriever:
    """Hybrid lexical/semantic retriever with behavioural re-ranking."""

    def __init__(self, settings: Settings, dataset, media: MediaIndex) -> None:
        self.cfg: RetrievalConfig = settings.retrieval
        self.settings = settings
        self.ds = dataset
        self.media = media
        self._encoder = None
        self._encoder_failed = False
        self._hist_text: dict[str, str] = {}
        self._hist_tokens: dict[str, set[str]] = {}
        self._idf: dict[str, float] = {}
        self._embeddings = None
        self._emb_ids: list[str] = []
        self._prepare()

    def _prepare(self) -> None:
        for msg in self.ds.history:
            text = msg.message_text or ""
            if msg.media_id:
                text = f"{text}\n{self.media.text_for(msg.media_id)}".strip()
            self._hist_text[msg.message_id] = text
            self._hist_tokens[msg.message_id] = set(tokenize(text))

        # Inverse document frequency so shared boilerplate ("dear customer")
        # contributes far less than a distinctive term ("tanker", "OTP").
        n = max(1, len(self._hist_tokens))
        df: dict[str, int] = {}
        for tokens in self._hist_tokens.values():
            for t in tokens:
                df[t] = df.get(t, 0) + 1
        self._idf = {t: math.log(1.0 + n / (1.0 + c)) for t, c in df.items()}

        if self.settings.use_embeddings:
            self._build_embeddings()

    def _build_embeddings(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.cfg.embedding_model)
            self._emb_ids = [m.message_id for m in self.ds.history]
            texts = [self._hist_text[i] or " " for i in self._emb_ids]
            self._embeddings = self._encoder.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True,
                batch_size=64, show_progress_bar=False,
            )
            log.info("built history embeddings: %s", self._embeddings.shape)
        except Exception as exc:  # noqa: BLE001 - retrieval degrades to lexical
            log.warning("embeddings unavailable, using lexical retrieval only: %s", exc)
            self._encoder = None
            self._encoder_failed = True

    # ------------------------------------------------------------------
    def _lexical(self, query_tokens: set[str], hist_id: str) -> float:
        other = self._hist_tokens.get(hist_id, set())
        if not query_tokens or not other:
            return 0.0
        shared = query_tokens & other
        if not shared:
            return 0.0
        num = sum(self._idf.get(t, 1.0) for t in shared)
        denom = math.sqrt(
            sum(self._idf.get(t, 1.0) for t in query_tokens)
            * sum(self._idf.get(t, 1.0) for t in other)
        )
        return num / denom if denom else 0.0

    def _semantic_scores(self, query: str, candidate_ids: list[str]) -> dict[str, float]:
        if self._encoder is None or self._embeddings is None:
            return {}
        try:
            import numpy as np

            q = self._encoder.encode([query or " "], convert_to_numpy=True, normalize_embeddings=True)
            index = {mid: i for i, mid in enumerate(self._emb_ids)}
            rows = [index[c] for c in candidate_ids if c in index]
            if not rows:
                return {}
            sims = (self._embeddings[rows] @ q.T).ravel()
            ids = [c for c in candidate_ids if c in index]
            return {mid: float(s) for mid, s in zip(ids, sims)}
        except Exception as exc:  # noqa: BLE001
            log.warning("semantic scoring failed: %s", exc)
            return {}

    @staticmethod
    def _structural(message: Message, hist: Message) -> float:
        """Relationship overlap between the incoming message and a history row."""
        score = 0.0
        if message.business_id and hist.business_id == message.business_id:
            score += 0.62
        if message.sender_user_id and hist.sender_user_id == message.sender_user_id:
            score += 0.58
        if message.group_id and hist.group_id == message.group_id:
            score += 0.30
        if hist.conversation_type == message.conversation_type:
            score += 0.12
        if message.media_type and hist.media_type == message.media_type:
            score += 0.06
        if message.forwarded_count >= 3 and hist.forwarded_count >= 3:
            score += 0.08
        return min(1.0, score)

    def _outcome(self, user_id: str, hist_id: str) -> str:
        event = self.ds.event_for(user_id, hist_id)
        if event is None:
            return "unknown"
        if event.negative:
            return "negative"
        if event.positive:
            return "positive"
        return "neutral"

    @staticmethod
    def _behavioural(outcome: str, intent: str) -> float:
        """Reward history whose recorded outcome supports the pending decision."""
        if intent == "suppress":
            return {"negative": 1.0, "neutral": 0.45, "unknown": 0.3, "positive": 0.05}[outcome]
        if intent == "promote":
            return {"positive": 1.0, "neutral": 0.45, "unknown": 0.3, "negative": 0.05}[outcome]
        return {"positive": 0.6, "negative": 0.6, "neutral": 0.5, "unknown": 0.35}[outcome]

    # ------------------------------------------------------------------
    def candidates(self, message: Message, content: ContentView, limit: int = 40) -> list[tuple[Message, float]]:
        """Content-similar history for this user, ignoring the pending decision.

        Used by the repetition detector, which must not be biased by the action
        the router is leaning toward.
        """
        pool = self.ds.history_by_user.get(message.user_id, [])
        if not pool:
            return []
        query = content.combined or message.message_text or ""
        qtokens = set(tokenize(query))
        ids = [h.message_id for h in pool]
        sem = self._semantic_scores(query, ids)

        scored: list[tuple[Message, float]] = []
        for hist in pool:
            if hist.message_id == message.message_id:
                continue
            s = sem.get(hist.message_id)
            if s is None:
                s = self._lexical(qtokens, hist.message_id)
            structural = self._structural(message, hist)
            scored.append((hist, round(0.65 * max(0.0, s) + 0.35 * structural, 4)))
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    def retrieve(
        self,
        message: Message,
        content: ContentView,
        intent: str,
        want: int = 1,
    ) -> list[EvidenceCandidate]:
        """Rank history for citation given the decision the router is making.

        `intent` is `promote` for notify, `suppress` for mute, `neutral` for
        digest; it steers which recorded outcomes count as supporting evidence.
        """
        pool = self.ds.history_by_user.get(message.user_id, [])
        if not pool:
            return []

        query = content.combined or message.message_text or ""
        qtokens = set(tokenize(query))
        ids = [h.message_id for h in pool]
        sem = self._semantic_scores(query, ids)

        out: list[EvidenceCandidate] = []
        for hist in pool:
            if hist.message_id == message.message_id:
                continue
            semantic = sem.get(hist.message_id)
            if semantic is None:
                semantic = self._lexical(qtokens, hist.message_id)
            semantic = max(0.0, min(1.0, semantic))
            structural = self._structural(message, hist)
            outcome = self._outcome(message.user_id, hist.message_id)
            behavioural = self._behavioural(outcome, intent)

            # A history row from an unrelated counterparty is not evidence about
            # this relationship no matter how similar the wording.
            if structural < 0.12 and semantic < 0.45:
                continue

            total = (
                self.cfg.w_semantic * semantic
                + self.cfg.w_structural * structural
                + self.cfg.w_behavioural * behavioural
            )
            out.append(
                EvidenceCandidate(hist, round(semantic, 4), round(structural, 4),
                                  round(behavioural, 4), round(total, 4), outcome)
            )

        out.sort(key=lambda c: (-c.total, c.message.message_id))
        selected = [c for c in out if c.total >= self.cfg.min_semantic_score][: max(1, want)]
        return selected
