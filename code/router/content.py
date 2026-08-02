"""Unified content view over text, OCR output and voice transcripts.

Every downstream engine reads this one structure, so a scam poster, a scam voice
note and a scam text message are all reasoned about through the same features.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .multimodal import AMOUNT_RE, MediaIndex, URL_RE
from .schema import Message

# --------------------------------------------------------------------------
# Lexicons. Grouped by the concept they evidence rather than by surface form so
# that the scoring layer can reason about *why* a phrase mattered.
# --------------------------------------------------------------------------

CREDENTIAL_TERMS = (
    "otp", "one time password", "one-time password", "login code", "verification code",
    "security code", "6 digit", "six digit", "cvv", "card number", "card details", "atm pin",
    "upi pin", "mpin", "password", "net banking password", "seed phrase", "private key",
    "wallet pin", "wallet details", "bank details", "account details", "confirm your pin",
    "share the otp", "share otp", "enter otp", "confirm password", "login credentials",
    "account number", "sharing your account", "share your account", "ifsc", "aadhaar number",
    "send details", "send your details",
)

ACCOUNT_THREAT_TERMS = (
    "will be blocked", "will be suspended", "account blocked", "account suspended",
    "profile will be blocked", "temporarily blocked", "access will expire", "expire today",
    "deactivated", "will be deactivated", "kyc pending", "kyc incomplete", "complete kyc",
    "verify now", "verify immediately", "reactivate", "restore access", "keep access active",
    "may be blocked", "failed verification", "verification failed", "suspicious login",
    # Service-termination pressure: the same coercion without the word "account".
    "service stops", "service will stop", "will be stopped", "will be cancelled",
    "will be disconnected", "processing will close", "will lapse", "final notice",
    "last warning", "before midnight", "or your connection", "avoid disconnection",
    "account closure", "avoid account closure", "account may get locked", "account check",
    "pending verification", "complete verification", "unless you login", "login now",
    "account lock", "avoid account lock", "service reactivation", "reactivation",
    "sim blocked", "sim will be blocked", "number blocked", "connection blocked",
    "card blocked", "will be barred", "access blocked", "be blocked",
    "approval window closes", "window closes today",
)

# An unexpected windfall is the setup half of an advance-fee scam; the payment
# demand is the sting. Either alone is weak, together they are conclusive.
WINDFALL_TERMS = (
    "loan approved", "amount will be released", "credit approved", "sanctioned",
    "you are approved", "claim approved", "payout approved", "funds are ready",
    "you have been selected", "you are selected", "selected for the role",
    "pre-approved", "your claim is approved", "eligible to receive",
)

PAYMENT_DEMAND_TERMS = (
    "pay now", "small fee", "reattempt fee", "delivery fee", "processing fee", "clearance fee",
    "customs fee", "release payment", "transfer amount", "send money", "pay to release",
    "pay a refundable", "convenience fee", "penalty amount", "outstanding amount pay",
    "scan and pay", "scan the qr", "scan this qr", "pay pending", "pending charge",
    "clearance amount", "pay immediately", "pay to continue", "complete the payment",
    "settle the amount", "pay the balance", "recharge now",
    "reactivation fee", "activation fee", "fee pending", "fee is pending", "send screenshot",
    "send the screenshot", "share screenshot", "pay today",
)

# A QR named in text is a payment instrument even when no image is attached, and
# "scan and send me the screenshot" is the classic peer-to-peer payment fraud.
PAY_AMOUNT_RE = re.compile(
    r"\b(pay|send|transfer|deposit|remit)\b[^.!?;\n]{0,25}?"
    r"(?:rs\.?|inr|₹|\$)?\s?\d{2,}(?:[,.]\d+)*",
    re.I,
)

QR_PAYMENT_RE = re.compile(
    r"\b(scan|scanning)\b[^.!?;\n]{0,40}?\b(qr|code|barcode)\b|\bqr\b[^.!?;\n]{0,30}?\b(pay|payment|send)\b",
    re.I,
)

# Refund bait: an unexpected windfall used to justify collecting card or wallet
# details. Legitimate refunds never require re-entering payment credentials.
REFUND_BAIT_TERMS = (
    "refund approved", "refund initiated", "claim your refund", "refund processing",
    "refund pending", "eligible for refund", "cashback credited", "amount will be reversed",
)

# Anti-fraud education mentions every credential keyword a scam does, but with
# the polarity reversed. Without this, a bank's "we will never ask for your OTP"
# advisory scores as credential harvesting - muting precisely the messages that
# protect users, which is worse than the noise it was meant to remove.
NEGATED_CREDENTIAL_RE = re.compile(
    r"\b(never|do not|don'?t|will not|won'?t|no ?one (?:will|should)|nobody|avoid|refrain from)\b"
    r"[^.!?]{0,60}?\b(ask|share|disclose|reveal|give|send|provide|tell)\b"
    r"[^.!?]{0,60}?\b(otp|pin|password|cvv|card|credential|details)\b",
    re.I,
)

# Reassurance rather than advice: "no payment or OTP is required for this
# delivery". A legitimate courier says this precisely to inoculate the customer
# against the scam version of the same message, so it is evidence of good faith
# and must not be read as a credential request.
NO_CREDENTIAL_NEEDED_RE = re.compile(
    r"\b(no|not|never|without)\b[^.!?;\n]{0,45}?"
    r"\b(otp|pin|password|cvv|payment|card details|bank details|verification)\b"
    r"[^.!?;\n]{0,35}?\b(required|needed|necessary|asked|requested|involved)\b",
    re.I,
)

# An imperative aimed at the reader. Advisory framing describes what a sender
# will never do; a demand tells the reader to do something now. Only the latter
# is an attack, and an attacker can freely prepend the former as camouflage.
CRED_DEMAND_RE = re.compile(
    r"\b(share|send|provide|confirm|enter|give|tell|reply with|forward|submit|ask for|verify with)\b"
    r"[^.!?;\n]{0,50}?\b(otp|pin|password|cvv|card number|card details|bank details|"
    r"account details|login code|verification code|security code|credential)",
    re.I,
)
NEGATION_TOKEN_RE = re.compile(
    r"\b(never|not|don'?t|won'?t|cannot|can'?t|no ?one|nobody|avoid|refrain)\b", re.I
)
CLAUSE_SPLIT_RE = re.compile(r"[.!?;\n]+")

ADVISORY_TERMS = (
    "safety advisory", "fraud awareness", "scam alert", "fraud alert", "beware of",
    "stay alert", "report fraud", "cyber safety", "security tips", "awareness campaign",
    "never ask for", "never share", "do not share", "protect yourself", "stay safe online",
    "get bowled by scammers", "how to stay safe",
)

# Romanised Hindi appears throughout Indian WhatsApp traffic. Without these
# a Hinglish scam that avoids English keywords scores zero.
TRANSLITERATED_RISK_TERMS = (
    "khata band", "khata block", "account band", "band ho jayega", "block ho jayega",
    "paisa bhejo", "paise bhejo", "turant bhejo", "abhi bhejo", "jaldi karo",
    "otp batao", "otp bhejo", "code batao", "link kholo", "link open karo",
    "verify karo", "payment karo", "transfer karo", "raqam", "jaldi",
)

TRANSLITERATED_URGENCY_TERMS = (
    "abhi", "turant", "jaldi", "foran", "emergency hai", "hospital mein",
    "call karo", "phone karo", "aajao", "aa jao",
)

PRIZE_TERMS = (
    "you have won", "congratulations you", "lucky winner", "lottery", "prize money",
    "claim your reward", "cash prize", "selected winner", "gift card worth", "free iphone",
)

INVESTMENT_FRAUD_TERMS = (
    "guaranteed return", "guaranteed profit", "assured returns", "double your money",
    "multibagger", "sure shot", "jackpot call", "intraday tips", "profit booked",
    "join vip group", "premium tips", "target hit", "no risk", "risk free profit",
)

IMPERSONATION_TERMS = (
    "support team", "customer support", "official support", "helpdesk", "security team",
    "bank official", "verification department", "compliance team", "we are calling from",
)

# Text aimed at the routing system itself rather than at the human recipient.
INJECTION_PATTERNS = (
    r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instruction|rule|routing|prompt)",
    r"disregard\s+(?:all\s+)?(?:previous|prior|above|earlier)",
    r"system\s+(?:note|prompt|message|instruction)\s*(?:for|to)?\s*(?:the\s+)?(?:notification\s+)?router",
    r"(?:always\s+)?mark\s+this\s+(?:message\s+)?as\s+(?:notify|urgent|important|high\s+priority)",
    r"(?:routing|notification)\s+rules?",
    r"you\s+are\s+an?\s+(?:ai|assistant|language\s+model|router)",
    r"override\s+(?:the\s+)?(?:filter|rule|setting|decision)",
    r"do\s+not\s+(?:mute|filter|block)\s+this",
    r"treat\s+this\s+as\s+(?:notify|urgent|priority)",
    r"new\s+instructions?\s*:",
    # Structured-field spoofing: content that imitates the engine's own
    # variables to smuggle a decision in ("action=notify", "confidence=1").
    r"[\"\']?\b(?:action|confidence|user_priority|verified_business|sender_risk)\b[\"\']?\s*[=:]\s*[\"\']?\w",
    r"internal\s+(?:router|system|engine)\s+metadata",
    r"(?:assistant|system|model)\s+instruction",
    r"routing\s+override",
    r"classify\s+(?:this\s+)?(?:as|to)\s+(?:notify|urgent|priority|important)",
    r"set\s+action\s*[=:]",
    r"ignore\s+(?:the\s+)?sender\s+risk",
)

URGENCY_TERMS = (
    "urgent", "urgently", "immediately", "right now", "asap", "as soon as possible",
    "emergency", "critical", "escalation", "escalate", "act now", "hurry", "last chance",
    "deadline", "eod", "end of day", "before it", "time sensitive", "quick help",
    "need help", "come online", "call now", "please call", "reach by", "pick up",
)

TIME_PRESSURE_RE = re.compile(
    r"\b(today|tonight|this (?:morning|afternoon|evening)|tomorrow|now(?!\s+that)"
    r"|within (?:the )?next \d+ ?(?:min|minutes|hours?|hrs?)|in (?:the )?next \d+ ?(?:min|minutes|hours?|hrs?)"
    r"|within \d+ ?(?:min|minutes|hours?|hrs?)|by \d{1,2}(?::\d{2})? ?(?:am|pm)?|before \d{1,2}(?::\d{2})?"
    r"|\d{1,2}:\d{2} ?(?:am|pm)?|next \d+ ?(?:min|minutes|hours?)|in the next"
    r"|(?:next |this )?(?:mon|tues|wednes|thurs|fri|satur|sun)day|next week|this weekend)\b",
    re.I,
)

# Senders routinely mark their own message as low priority. Honouring that is
# one of the strongest and least-used signals in the dataset: it separates a
# real request from a courteous update that merely mentions a time.
DEESCALATION_TERMS = (
    "nothing urgent", "not urgent", "no rush", "no hurry", "no pressure", "no need to reply",
    "no need to respond", "no need to answer", "whenever you get time", "whenever you can",
    "when you get a chance", "at your convenience", "just fyi", "just informing", "for your info",
    "don't call", "dont call", "no action needed", "nothing to do", "just sharing",
    "will update later", "we can talk tomorrow", "take your time", "if you are free",
    "only if", "join only if", "no need to", "nothing dramatic", "no reply needed",
    "nothing blocking", "not blocking", "nothing critical", "if you get time",
    "if you get a chance", "read it before", "no hurry at all", "not blocking anything",
)

# A satisfaction survey is grammatically a request but never an interruption.
# Without this a "can you fill a quick review?" outscores a real deadline.
FEEDBACK_REQUEST_TERMS = (
    "fill a quick review", "quick review", "share your feedback", "give your feedback",
    "rate your experience", "how was your experience", "hear about your experience",
    "take a short survey", "valuable feedback", "rate us", "review your order",
    "tell us how", "your opinion matters",
)

# Peer-to-peer selling inside a group. Reads as conversation but is really an
# offer, which is how the reference labels treat it.
MARKETPLACE_TERMS = (
    "selling", "for sale", "up for sale", "dm if interested", "message me if", "pickup near",
    "pickup is", "pick up near", "barely used", "no damage", "bought last year", "size m",
    "medium size", "brand new", "negotiable", "best price", "interested people", "kept aside",
    "share pics", "photos are attached", "photos for the", "warehouse pickup",
    "buyer cancelled", "clear it fast", "only if serious", "kept aside", "still want it",
    "price is final", "first come first", "resale", "reselling",
)

# Words that mark a message as a commercial transaction rather than conversation.
# Paired with an offer term, these turn peer chat into a peer-to-peer offer.
COMMERCE_TERMS = (
    "pickup", "pick up", "cash or upi", "upi", "cash", "price", "rs.", "rs ", "₹",
    "buyer", "seller", "delivery charge", "shipping", "in stock", "available", "booking amount",
)

PROMO_TERMS = (
    "% off", "discount", "flat off", "sale", "offer", "coupon", "promo code", "cashback",
    "buy now", "shop now", "order now", "book now", "limited time", "hurry", "deal",
    "free delivery", "lowest price", "unbeatable price", "starting at", "upto", "up to",
    "exclusive", "t&c apply", "terms and conditions", "unsubscribe", "reply stop",
    "new arrivals", "mega sale", "best price", "special price", "save big",
)

GREETING_TERMS = (
    "good morning", "good night", "good evening", "have a blessed", "stay positive",
    "keep smiling", "god bless", "have a nice day", "happy sunday", "warm wishes",
    "hope today is", "good vibes", "blessings", "stay happy", "shubh",
)

FORWARD_TERMS = (
    "fwd", "forwarded as received", "fwd as received", "as received", "pls forward",
    "please forward", "share with", "forward to", "sharing here", "received on whatsapp",
    "copy paste", "must read", "very useful", "share maximum",
)

# Scheduled obligations only. Ambiguous social words ("match", "practice") are
# excluded because casual chat about a match is not an event notification.
EVENT_TERMS = (
    "meeting", "appointment", "booking", "reservation", "schedule", "scheduled", "rsvp",
    "venue", "circular", "field trip", "seminar", "webinar", "workshop", "reminder",
    "ceremony", "rehearsal", "assembly", "parents meeting", "consent", "form is open",
    "register by", "check-in", "itinerary", "cultural night", "annual day", "sports day",
    "last date", "deadline to submit", "submit by", "timing", "agenda",
)

PAYMENT_LEGIT_TERMS = (
    "bill", "due date", "statement", "invoice", "emi", "premium due", "payment received",
    "transaction", "debited", "credited", "receipt", "maintenance", "dues", "outstanding",
    "recharge", "balance", "auto-pay", "autopay", "installment", "challan", "fee payment",
)

BUSINESS_UPDATE_TERMS = (
    "order", "delivery", "shipped", "dispatched", "out for delivery", "tracking",
    "arriving", "your package", "service request", "ticket", "complaint", "refund processed",
    "appointment confirmed", "booking confirmed", "policy", "renewal", "account statement",
    "update is ready", "status", "confirmed",
)

DIRECT_REQUEST_RE = re.compile(
    r"\b(can you|could you|please (?:call|send|share|confirm|check|join|reply|come|bring|pay)"
    r"|pls (?:call|send|share|confirm|check|join|reply|come)|let me know|need you to|waiting for your"
    r"|are you (?:free|available|around|still)|do you have|would you|will you|reply once|confirm before"
    r"|send me|call me|ping me|join the|revert)\b",
    re.I,
)

QUESTION_RE = re.compile(r"\?")
MENTION_RE = re.compile(r"@(u_\d+|\w+)")
OPT_OUT_RE = re.compile(r"\b(reply stop|unsubscribe|opt out|stop to unsubscribe)\b", re.I)

SUSPICIOUS_TLDS = (".xyz", ".top", ".click", ".link", ".buzz", ".icu", ".rest", ".online", ".site")
URL_SHORTENERS = ("bit.ly", "tinyurl", "t.co", "rb.gy", "cutt.ly", "is.gd", "shorturl", "rebrand.ly")


def normalize_keep_spacing(text: str) -> str:
    """NFKC + control-character strip, but WITHOUT collapsing runs of spaces.

    Wider gaps are the only thing separating one obfuscated word from the next
    ("O T P  n o w"), so the matching view must see them.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("​", "").replace(" ", " ")
    return re.sub(r"[​-‏‪-‮⁦-⁩]", "", text).strip()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("​", "").replace("\xa0", " ")
    # Bidi overrides and zero-width joiners are used to disguise payloads.
    text = re.sub(r"[​-‏‪-‮⁦-⁩]", "", text)
    return re.sub(r"[ \t]+", " ", text).strip()


_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_SPACED_RE = re.compile(r"\b(?:[A-Za-z]\s){2,}[A-Za-z]\b")
_INNER_DIGIT_RE = re.compile(r"(?<=[A-Za-z])[013457@$]+(?=[A-Za-z])|\b[013457@$]+(?=[A-Za-z]{2,})")


def normalize_for_matching(text: str) -> str:
    """A lexicon-facing view that survives cheap obfuscation.

    Fraud text routinely evades keyword filters with substitutions ("0TP",
    "acc0unt") and letter spacing ("S H A R E"). Matching against a normalised
    copy defeats both without touching the text shown to a human. Digits are
    only de-leeted when adjacent to letters, so "pay 499" and "4 PM" survive
    intact.
    """
    lowered = text.lower()
    collapsed = _SPACED_RE.sub(lambda m: m.group(0).replace(" ", ""), lowered)
    return _INNER_DIGIT_RE.sub(lambda m: m.group(0).translate(_LEET), collapsed)


_TERM_RE_CACHE: dict[str, re.Pattern] = {}


def _term_pattern(term: str) -> re.Pattern:
    """Compile a lexicon term into a whole-word matcher.

    Plain substring search silently mis-fires: "emi" matches inside "supervisor
    email", typing a university deadline notice as a payment. A word boundary
    stops that, while an optional trailing inflection keeps the plural and
    participle forms substring matching handled for free ("order" -> "orders").

    A boundary only anchors next to a word character, so terms starting or
    ending in punctuation ("% off", "rs.") take a bare escape on that side.
    """
    pattern = _TERM_RE_CACHE.get(term)
    if pattern is None:
        left = "\\b" if term[:1].isalnum() else ""
        right = "(?:s|es|ed|ing)?\\b" if term[-1:].isalnum() else ""
        pattern = re.compile(left + re.escape(term) + right, re.I)
        _TERM_RE_CACHE[term] = pattern
    return pattern


def count_hits(low_text: str, terms: tuple[str, ...]) -> list[str]:
    """Return the terms that occur as whole words in the text."""
    return [t for t in terms if _term_pattern(t).search(low_text)]


def has_active_credential_demand(text: str) -> bool:
    """True when some clause actually asks the reader for a secret.

    Two subtleties, both found by adversarial testing:

    * Evaluated per clause, so a disclaimer in one sentence cannot excuse a live
      demand in the next ("We never ask for OTP. Now share your OTP.").
    * Negation only counts when it *precedes* the demand verb. A scam's own
      "share your OTP to avoid account closure" contains a negation word after
      the ask, which a whole-clause test would misread as a disclaimer.
    """
    for clause in CLAUSE_SPLIT_RE.split(text):
        for match in CRED_DEMAND_RE.finditer(clause):
            prefix = clause[max(0, match.start() - 30) : match.start()]
            if not NEGATION_TOKEN_RE.search(prefix):
                return True
    return False


@dataclass
class ContentView:
    """Everything the engines need to know about *what was said*."""

    message_id: str
    text: str = ""              # original message text, normalized
    media_text: str = ""        # OCR output or voice transcript
    combined: str = ""          # both, for lexical and semantic analysis
    low: str = ""               # lowercase combined

    modality: str = "text"      # text | image | voice
    char_count: int = 0
    word_count: int = 0

    urls: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    amounts: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)

    credential_hits: list[str] = field(default_factory=list)
    account_threat_hits: list[str] = field(default_factory=list)
    payment_demand_hits: list[str] = field(default_factory=list)
    prize_hits: list[str] = field(default_factory=list)
    investment_hits: list[str] = field(default_factory=list)
    impersonation_hits: list[str] = field(default_factory=list)
    injection_hits: list[str] = field(default_factory=list)
    urgency_hits: list[str] = field(default_factory=list)
    time_pressure_hits: list[str] = field(default_factory=list)
    promo_hits: list[str] = field(default_factory=list)
    greeting_hits: list[str] = field(default_factory=list)
    forward_hits: list[str] = field(default_factory=list)
    event_hits: list[str] = field(default_factory=list)
    payment_hits: list[str] = field(default_factory=list)
    business_update_hits: list[str] = field(default_factory=list)
    deescalation_hits: list[str] = field(default_factory=list)
    marketplace_hits: list[str] = field(default_factory=list)
    refund_bait_hits: list[str] = field(default_factory=list)
    commerce_hits: list[str] = field(default_factory=list)
    windfall_hits: list[str] = field(default_factory=list)
    translit_risk_hits: list[str] = field(default_factory=list)
    translit_urgency_hits: list[str] = field(default_factory=list)
    demands_amount: bool = False
    lexicon_hit_count: int = 0
    semantic_intent: str = ""      # cross-lingual fallback verdict
    semantic_margin: float = 0.0
    non_latin_ratio: float = 0.0
    feedback_hits: list[str] = field(default_factory=list)
    caption_marketplace_hits: list[str] = field(default_factory=list)
    caption_promo_hits: list[str] = field(default_factory=list)
    caption_payment_hits: list[str] = field(default_factory=list)
    caption_event_hits: list[str] = field(default_factory=list)
    caption_business_update_hits: list[str] = field(default_factory=list)
    advisory_hits: list[str] = field(default_factory=list)
    is_safety_advisory: bool = False
    has_active_credential_demand: bool = False

    has_direct_request: bool = False
    has_question: bool = False
    has_opt_out_footer: bool = False
    has_qr: bool = False          # QR detected in an attached image
    mentions_qr_payment: bool = False  # QR payment described in the text itself
    suspicious_url: bool = False
    shortened_url: bool = False

    # Media-derived extras.
    image_layout: str = ""
    scene_tags: list[str] = field(default_factory=list)
    voice_duration_s: float = 0.0
    voice_wpm: float = 0.0
    voice_rushed: bool = False
    media_failed: bool = False

    def mentions_user(self, user_id: str) -> bool:
        return any(m.lower() == user_id.lower() for m in self.mentions)

    @property
    def is_empty(self) -> bool:
        return not self.combined.strip()


def build_content(message: Message, media: MediaIndex) -> ContentView:
    text = normalize(message.message_text)
    media_text = ""
    cv = ContentView(message_id=message.message_id, text=text)

    if message.media_type == "image" and message.media_id in media.images:
        img = media.images[message.media_id]
        media_text = img.as_text()
        cv.modality = "image"
        cv.image_layout = img.layout
        cv.scene_tags = list(img.scene_tags)
        cv.has_qr = img.has_qr
        cv.media_failed = not img.ok
    elif message.media_type == "voice" and message.media_id in media.voices:
        vn = media.voices[message.media_id]
        media_text = vn.as_text()
        cv.modality = "voice"
        cv.voice_duration_s = vn.duration_s
        cv.voice_wpm = vn.words_per_minute
        cv.voice_rushed = vn.is_rushed
        cv.media_failed = not vn.ok
    elif message.media_type:
        cv.modality = message.media_type
        cv.media_failed = True

    cv.media_text = normalize(media_text)
    cv.combined = "\n".join(p for p in (cv.text, cv.media_text) if p).strip()
    # Lexicons match against the de-obfuscated view; `low` stays the plain
    # lowercase text for anything that needs the literal wording.
    spaced_source = "\n".join(
        p for p in (normalize_keep_spacing(message.message_text), cv.media_text) if p
    )
    low = normalize_for_matching(spaced_source)
    cv.low = low
    cv.char_count = len(cv.combined)
    cv.word_count = len(cv.combined.split())

    cv.urls = sorted({m.group(1).lower().rstrip(".,") for m in URL_RE.finditer(cv.combined)})[:10]
    cv.domains = sorted({_domain_of(u) for u in cv.urls if _domain_of(u)})
    cv.amounts = sorted({m.group(0) for m in AMOUNT_RE.finditer(cv.combined)})[:8]
    cv.mentions = sorted({m.group(1) for m in MENTION_RE.finditer(cv.combined)})[:10]

    cv.credential_hits = count_hits(low, CREDENTIAL_TERMS)
    cv.account_threat_hits = count_hits(low, ACCOUNT_THREAT_TERMS)
    cv.payment_demand_hits = count_hits(low, PAYMENT_DEMAND_TERMS)
    cv.prize_hits = count_hits(low, PRIZE_TERMS)
    cv.investment_hits = count_hits(low, INVESTMENT_FRAUD_TERMS)
    cv.impersonation_hits = count_hits(low, IMPERSONATION_TERMS)
    cv.urgency_hits = count_hits(low, URGENCY_TERMS)
    cv.promo_hits = count_hits(low, PROMO_TERMS)
    cv.greeting_hits = count_hits(low, GREETING_TERMS)
    cv.forward_hits = count_hits(low, FORWARD_TERMS)
    cv.event_hits = count_hits(low, EVENT_TERMS)
    cv.payment_hits = count_hits(low, PAYMENT_LEGIT_TERMS)
    cv.business_update_hits = count_hits(low, BUSINESS_UPDATE_TERMS)
    cv.deescalation_hits = count_hits(low, DEESCALATION_TERMS)
    cv.marketplace_hits = count_hits(low, MARKETPLACE_TERMS)
    cv.refund_bait_hits = count_hits(low, REFUND_BAIT_TERMS)
    cv.commerce_hits = count_hits(low, COMMERCE_TERMS)
    cv.windfall_hits = count_hits(low, WINDFALL_TERMS)
    cv.mentions_qr_payment = bool(QR_PAYMENT_RE.search(cv.combined))
    cv.demands_amount = bool(PAY_AMOUNT_RE.search(cv.combined))
    cv.translit_risk_hits = count_hits(low, TRANSLITERATED_RISK_TERMS)
    cv.translit_urgency_hits = count_hits(low, TRANSLITERATED_URGENCY_TERMS)
    cv.feedback_hits = count_hits(low, FEEDBACK_REQUEST_TERMS)
    # Peer-selling is judged on what the sender actually wrote, not on OCR of an
    # attached poster: a university internship flyer contains plenty of
    # offer-like vocabulary without the sender selling anything.
    # What the sender wrote outranks what the attachment says. A one-line
    # caption over a 300-word university flyer is still the message; letting
    # poster OCR vote on message_type lets the attachment hijack the label.
    # Safety still reads the full combined text - a scam poster stays dangerous.
    caption = cv.text.lower() if cv.text.strip() else low
    cv.caption_marketplace_hits = count_hits(caption, MARKETPLACE_TERMS)
    cv.caption_promo_hits = count_hits(caption, PROMO_TERMS)
    cv.caption_payment_hits = count_hits(caption, PAYMENT_LEGIT_TERMS)
    cv.caption_event_hits = count_hits(caption, EVENT_TERMS)
    cv.caption_business_update_hits = count_hits(caption, BUSINESS_UPDATE_TERMS)
    cv.advisory_hits = count_hits(low, ADVISORY_TERMS)
    # Educational framing plus reversed polarity: the message warns about the
    # request rather than making it.
    cv.has_active_credential_demand = has_active_credential_demand(cv.combined)
    cv.is_safety_advisory = bool(
        (
            NEGATED_CREDENTIAL_RE.search(cv.combined)
            or NO_CREDENTIAL_NEEDED_RE.search(cv.combined)
            or cv.advisory_hits
        )
        and not cv.has_active_credential_demand
        and not cv.payment_demand_hits
    )

    cv.injection_hits = [p for p in INJECTION_PATTERNS if re.search(p, low)]
    cv.time_pressure_hits = sorted({m.group(0).lower() for m in TIME_PRESSURE_RE.finditer(cv.combined)})[:8]
    cv.has_direct_request = bool(DIRECT_REQUEST_RE.search(cv.combined))
    cv.has_question = bool(QUESTION_RE.search(cv.combined))
    cv.has_opt_out_footer = bool(OPT_OUT_RE.search(cv.combined))
    cv.lexicon_hit_count = sum(len(x) for x in (
        cv.credential_hits, cv.account_threat_hits, cv.payment_demand_hits, cv.prize_hits,
        cv.investment_hits, cv.urgency_hits, cv.promo_hits, cv.greeting_hits,
        cv.event_hits, cv.payment_hits, cv.business_update_hits, cv.translit_risk_hits,
        cv.translit_urgency_hits, cv.time_pressure_hits,
    ))
    cv.suspicious_url = any(d.endswith(SUSPICIOUS_TLDS) for d in cv.domains)
    cv.shortened_url = any(s in d for d in cv.domains for s in URL_SHORTENERS)
    return cv


def _domain_of(url: str) -> str:
    u = re.sub(r"^https?://", "", url.strip().lower())
    return u.split("/")[0].strip()
