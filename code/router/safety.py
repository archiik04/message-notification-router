"""Independent safety engine.

This layer deliberately knows nothing about whether the user *likes* a sender.
Its verdict is computed from content and sender provenance alone, and the
routing layer treats it as a veto: personalisation can never promote a message
that this engine considers dangerous.

Threat model covered:
  * credential harvesting (OTP / PIN / password / CVV requests)
  * account-suspension pressure and fake support desks
  * brand impersonation via lookalike sending domains
  * advance-fee and fake-payment demands
  * prize / lottery bait and investment-return fraud
  * malicious QR codes and suspicious or shortened links
  * prompt injection aimed at the router itself
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .content import ContentView
from .schema import BusinessAccount, Message

# A brand named in the body is a claim of identity; if the sending domain does
# not back it up, the message is impersonating that brand.
KNOWN_BRAND_TOKENS = (
    "amazon", "flipkart", "paytm", "phonepe", "gpay", "google pay", "hdfc", "icici", "sbi",
    "axis", "kotak", "bank of america", "chase", "paypal", "netflix", "swiggy", "zomato",
    "uber", "ola", "irctc", "jio", "airtel", "vodafone", "vi ", "myntra", "ajio", "nykaa",
    "whatsapp", "meta", "instagram", "microsoft", "apple", "dbs", "citi", "yes bank",
)

# Legitimate-looking hosts that are really lookalikes: a brand token glued to
# extra words in the domain label (amazonpay-delivery.in, account-login.in).
LOOKALIKE_RE = re.compile(
    r"\b([a-z0-9]*(?:amazon|paytm|phonepe|hdfc|icici|sbi|axis|netflix|swiggy|zomato|irctc|jio|airtel"
    r"|paypal|apple|microsoft|whatsapp|flipkart|myntra|kotak|account|secure|verify|login|wallet|kyc|pay)"
    r"[a-z0-9\-]*)\.(?:[a-z]{2,24})\b",
    re.I,
)

GENERIC_SALUTATION_RE = re.compile(
    r"\b(dear (?:customer|user|sir/madam|member)|hi customer|hello user|dear valued)\b", re.I
)


@dataclass
class SafetyVerdict:
    """Structured risk assessment for one message."""

    scam_score: float = 0.0
    spam_score: float = 0.0
    injection_score: float = 0.0
    impersonation_score: float = 0.0

    threats: list[str] = field(default_factory=list)
    signals: dict[str, float] = field(default_factory=dict)
    primary_threat: str = ""

    @property
    def risk(self) -> float:
        """Overall unsafety, dominated by the strongest individual threat."""
        return max(self.scam_score, self.injection_score, self.impersonation_score)

    @property
    def is_scam(self) -> bool:
        return self.risk >= 0.55

    @property
    def is_spam(self) -> bool:
        return self.spam_score >= 0.62 and not self.is_scam


class SafetyEngine:
    def __init__(self, dataset) -> None:
        self.ds = dataset

    def assess(self, message: Message, content: ContentView) -> SafetyVerdict:
        v = SafetyVerdict()
        biz = self.ds.businesses.get(message.business_id) if message.business_id else None

        scam = 0.0
        threats: list[str] = []

        # Anti-fraud education quotes the very phrases fraud uses. Scoring it as
        # fraud would suppress the warnings users most need to see, so advisory
        # framing withdraws the keyword-driven threats while leaving the
        # behavioural ones (links, payment demands) intact.
        advisory = content.is_safety_advisory
        if advisory:
            v.signals["safety_advisory"] = 1.0

        # ---- Credential harvesting -------------------------------------
        # Asking a person to hand over an OTP, PIN or password is close to
        # definitionally fraud: no legitimate sender ever needs one. This is
        # weighted to stand on its own rather than to need corroboration.
        if content.credential_hits and not advisory:
            weight = 0.62 if len(content.credential_hits) == 1 else 0.74
            scam += weight
            threats.append("credential_request")
            v.signals["credential"] = weight

        # ---- Account-suspension / fake support pressure ----------------
        if content.account_threat_hits and not advisory:
            bump = 0.34 + 0.08 * min(2, len(content.account_threat_hits) - 1)
            scam += bump
            threats.append("account_threat_pressure")
            v.signals["account_threat"] = bump

        if content.impersonation_hits and not advisory and (
            content.credential_hits or content.account_threat_hits
        ):
            scam += 0.14
            threats.append("fake_support_identity")
            v.signals["fake_support"] = 0.14

        # ---- Advance-fee / payment extraction --------------------------
        if content.payment_demand_hits or content.demands_amount:
            bump = 0.34
            if content.credential_hits or content.account_threat_hits:
                bump += 0.14
            # "Loan approved, just pay the processing fee" - the windfall exists
            # only to make the fee sound reasonable.
            if content.windfall_hits or content.refund_bait_hits or content.prize_hits:
                bump += 0.30
                threats.append("advance_fee_pattern")
            scam += bump
            threats.append("payment_demand")
            v.signals["payment_demand"] = bump

        # ---- Romanised-Hindi fraud vocabulary ---------------------------
        if content.translit_risk_hits:
            bump = 0.34 + 0.12 * min(2, len(content.translit_risk_hits) - 1)
            scam += bump
            threats.append("transliterated_fraud_language")
            v.signals["translit_risk"] = bump

        # ---- Refund bait -----------------------------------------------
        # An unprompted refund that asks the user to "verify" payment details is
        # collection dressed as reimbursement; a real refund needs nothing back.
        if content.refund_bait_hits:
            bump = 0.20
            if content.credential_hits or content.payment_demand_hits:
                bump += 0.30
            scam += bump
            threats.append("refund_bait")
            v.signals["refund_bait"] = bump

        # ---- Prize bait and investment fraud ---------------------------
        if content.prize_hits:
            # Nobody legitimately opens with "you have won 25 lakh".
            scam += 0.62
            threats.append("prize_bait")
            v.signals["prize"] = 0.62
        if content.investment_hits:
            bump = 0.58 + (0.12 if len(content.investment_hits) > 1 else 0.0)
            scam += bump
            threats.append("investment_fraud")
            v.signals["investment"] = bump

        # ---- Link and QR risk ------------------------------------------
        link_risk, link_threats = self._link_risk(content, biz)
        scam += link_risk
        threats.extend(link_threats)
        if link_risk:
            v.signals["link_risk"] = round(link_risk, 3)

        # ---- Brand impersonation ---------------------------------------
        imp = self._impersonation(content, biz)
        v.impersonation_score = imp
        if imp >= 0.4:
            threats.append("brand_impersonation")
            scam += 0.18
            v.signals["impersonation"] = imp

        # ---- Unknown-sender amplification ------------------------------
        # The same demand is far more dangerous from a stranger than from a
        # counterparty the user already transacts with.
        if scam > 0 and self._is_cold_sender(message):
            scam += 0.12
            threats.append("first_contact_sender")
            v.signals["cold_sender"] = 0.12

        # ---- Prompt injection ------------------------------------------
        if content.injection_hits:
            v.injection_score = min(1.0, 0.62 + 0.12 * (len(content.injection_hits) - 1))
            threats.append("router_prompt_injection")
            v.signals["injection"] = v.injection_score
            # Content that tries to steer the router is itself hostile intent.
            scam = max(scam, 0.60)

        v.scam_score = round(min(1.0, scam), 3)
        v.spam_score = self._spam_score(message, content, biz)
        v.threats = threats
        v.primary_threat = threats[0] if threats else ""
        return v

    # ------------------------------------------------------------------
    def _link_risk(self, content: ContentView, biz: BusinessAccount | None) -> tuple[float, list[str]]:
        risk, threats = 0.0, []
        if content.suspicious_url:
            risk += 0.20
            threats.append("suspicious_tld")
        if content.shortened_url:
            risk += 0.16
            threats.append("shortened_link")

        # A link paired with a credential or account-threat ask is the classic
        # phishing shape, regardless of the domain's reputation.
        if content.urls and (content.credential_hits or content.account_threat_hits):
            risk += 0.28
            threats.append("phishing_link_flow")

        # Being told to pay at a link is the payment-fraud equivalent of a
        # credential phish; the destination is what makes the demand actionable.
        if content.urls and content.payment_demand_hits:
            risk += 0.24
            threats.append("payment_link_flow")

        # A hidden destination behind time pressure is the same shape with the
        # domain concealed, which is worse rather than better.
        if content.shortened_url and (content.account_threat_hits or content.urgency_hits):
            risk += 0.24
            threats.append("obscured_link_pressure")

        qr_present = content.has_qr or content.mentions_qr_payment
        if qr_present and (
            content.payment_demand_hits or content.account_threat_hits or content.demands_amount
        ):
            risk += 0.26
            threats.append("payment_qr_risk")

        if biz is not None and biz.domain_mismatch:
            for dom in content.domains:
                if biz.official_domain and biz.official_domain not in dom:
                    risk += 0.10
                    threats.append("offbrand_link")
                    break
        return min(0.5, risk), threats

    def _impersonation(self, content: ContentView, biz: BusinessAccount | None) -> float:
        score = 0.0
        if biz is not None and biz.domain_mismatch:
            if biz.brand_lookalike_domain:
                # The brand's own name on a domain the brand does not own.
                score += 0.72
            elif not biz.verified:
                # An unrelated third-party domain is only suspicious when the
                # account behind it was never verified.
                score += 0.18

            if not biz.verified:
                score += 0.10
            if biz.account_age_days < 90:
                score += 0.20
            if 0 < biz.domain_used_by_sender_age_days < 60:
                score += 0.15
            if biz.user_reports_30d >= 25:
                score += 0.15

            # An established, verified brand routing through a link service is
            # normal marketing plumbing, not impersonation.
            if biz.verified and biz.account_age_days > 730 and not biz.brand_lookalike_domain:
                score = min(score, 0.12)

        if biz is not None and not biz.verified and biz.user_reports_30d >= 5:
            score += 0.10

        # A domain built out of security vocabulary (verify / secure / kyc /
        # login / alert) is bait regardless of whether we hold a business record
        # for the sender - most impersonation has no record at all.
        bait_domain = any(
            re.search(r"(verify|secure|kyc|login|alert|otp|update|refund|reward)", d)
            for d in content.domains
        )
        if bait_domain and (
            content.credential_hits or content.account_threat_hits
            or content.payment_demand_hits or content.demands_amount
        ):
            score += 0.45

        # Brand claimed in the body while the visible link is a lookalike host.
        claims_brand = any(b in content.low for b in KNOWN_BRAND_TOKENS)
        lookalike = any(LOOKALIKE_RE.fullmatch(d) or LOOKALIKE_RE.match(d) for d in content.domains)
        if claims_brand and lookalike and biz is None:
            score += 0.45
        elif lookalike and (content.credential_hits or content.account_threat_hits):
            score += 0.30

        if GENERIC_SALUTATION_RE.search(content.combined) and (
            content.credential_hits or content.account_threat_hits
        ):
            score += 0.10
        return round(min(1.0, score), 3)

    def _is_cold_sender(self, message: Message) -> bool:
        """True when this user has no prior history with the sender at all."""
        prior = self.ds.history_by_user.get(message.user_id, [])
        if message.sender_user_id:
            return not any(h.sender_user_id == message.sender_user_id for h in prior)
        if message.business_id:
            rel = self.ds.relationship(message.user_id, message.business_id)
            if rel and rel.has_active_relationship:
                return False
            return not any(h.business_id == message.business_id for h in prior)
        return False

    def _spam_score(
        self, message: Message, content: ContentView, biz: BusinessAccount | None
    ) -> float:
        """Unsolicited bulk marketing, distinct from outright fraud."""
        score = 0.0
        promo_strength = min(1.0, len(content.promo_hits) / 4.0)
        if promo_strength:
            score += 0.34 * promo_strength
        if content.has_opt_out_footer:
            score += 0.16

        if biz is not None:
            rel = self.ds.relationship(message.user_id, biz.business_id)
            if rel is None:
                score += 0.22  # bulk marketing with no relationship at all
            else:
                if rel.opted_out:
                    score += 0.34
                if rel.messages_dismissed_30d >= 3 and rel.messages_opened_30d <= 1:
                    score += 0.20
            if biz.user_reports_30d >= 8:
                score += 0.12
            if biz.messages_sent_30d >= 2000:
                score += 0.06

        # Chain forwards of promotional material.
        if message.forwarded_count >= 3 and (content.promo_hits or content.forward_hits):
            score += 0.18
        # A message forwarded this many times is chain content by definition,
        # whatever it claims to be about.
        if message.forwarded_count >= 12:
            score += 0.22
        return round(min(1.0, score), 3)
