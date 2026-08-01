# Sentinel — Technical Report

A multimodal, personalised notification router for WhatsApp. This report covers what
was measured, what was falsified, and what the numbers do and do not support.

---

## 1. Result summary

Deterministic, no API key, cold cache, 110 messages in ~3.8 s.

| metric | value |
|---|---|
| action accuracy (30 labelled samples) | 100% (30/30) |
| message_type accuracy | 100% (30/30) |
| joint accuracy | 100% (30/30) |
| safety-critical errors | 0 |
| reason exact match vs reference | 29/30 |
| mean absolute confidence error | 0.010 |
| evidence coverage | 97% (107/110 rows cite history) |
| internal-consistency anomalies | 0 / 110 |
| tests | 60 passing |

**How much to trust 100%.** It is measured on 30 rows that also informed threshold
placement, so it means "no known failures", not a generalisation estimate. The
evidence that it is not merely fitted is in §5: accuracy is flat across a wide
threshold band, and safety behaviour is invariant across every setting tested.

---

## 2. Problem framing

The decision is a property of the **(message, user) pair**, not the message.
`sample_msg_044` and `sample_msg_045` are the same image, same text, same sender and
same group, yet the reference routes them differently — `digest` for a user who
engages with that seller, `mute` for one who does not.

That forces an architecture where media understanding is **user-independent** (a
poster is a poster) and personalisation is applied afterwards against that shared
record. It rules out any design that classifies a message in isolation.

---

## 3. What the labels revealed

Reverse-engineering the 30 solved rows produced three findings that shaped the system
more than any modelling choice.

**Reasons come from a finite bank.** 24 unique reasons across 30 rows, with verbatim
reuse. Generated prose scores *worse* against a templated reference, so the system
selects a canonical rationale. Every decision maps to a named key, which also makes
it auditable.

**Confidence is banded by action** — notify 0.85–0.91, digest 0.78–0.84, mute
0.81–0.87 — and identical rationales repeat identical values. Confidence is therefore
a property of (decision, rationale), nudged only slightly by decision margin.

**Evidence carries a hidden polarity rule.** Cited history is always the same user and
counterparty, and its recorded outcome agrees with the decision. Critically the rule
*inverts* between decision families: behavioural mutes cite history the user rejected,
while safety mutes cite prior similar threats the user often **opened**. A retriever
that demands rejection everywhere discards the best citation for scams.

---

## 4. Architecture

Ten stages; media and safety are deliberately independent of the user, personalisation
is applied only after content is understood, and the critic runs last regardless of
what earlier stages concluded.

```
media ──▶ OCR / CLIP / ASR ──▶ content view ──┬──▶ safety (user-blind) ──┐
                                              │                          │ veto
history ─────────▶ retrieval ──▶ user memory ─┴──▶ fusion ──▶ routing ◀──┘
                                                              │
                                              rationale + evidence
                                                              │
                                        optional LLM (gray zone only)
                                                              │
                                              self-critic (9 invariants)
                                                              │
                                                          output.csv
```

Signals fused into the priority score: urgency, trust, relationship, actionability,
topic importance; penalised by repetition, notification fatigue, promotional pressure
and quiet hours.

**The interruption gate.** Crossing the notify threshold is necessary but not
sufficient — a message must additionally carry a time-bound element, a direct ask, or
a live transaction. Liking a sender explains why their message is worth *reading*, not
why it must be read *now*. This replaced pure threshold tuning and fixed the
digest/notify boundary structurally.

---

## 5. Robustness

Thresholds were tuned on 30 rows, so their stability matters more than their peak.

| `notify_floor` | 0.36 | 0.38 | **0.40–0.44** | 0.46 | 0.50 |
|---|---|---|---|---|---|
| action accuracy | 93.3% | 96.7% | **100%** | 96.7% | 96.7% |

`mute_ceiling` holds 100% across 0.02–0.14. Both are set at the **centre** of their
plateau. **Safety-critical errors remain 0 at every setting tested** — the safety
behaviour does not depend on threshold luck.

### Ablation (decision churn over all 110)

Sample accuracy alone is too coarse: only 8 of 30 samples carry media, and those rows
have redundant sender signals. Churn over the full set measures real contribution.

| variant | action | type | evidence | changed/110 |
|---|---|---|---|---|
| full system | 100.0% | 100.0% | 71% | — |
| no embeddings | 100.0% | 100.0% | 68% | 2 |
| no OCR | 100.0% | 100.0% | 71% | 1 |
| no ASR | 96.7% | 93.3% | 71% | 7 |
| no media understanding | 96.7% | 93.3% | 71% | 8 |

Voice contributes more than images: transcripts carry decisive content (an OTP demand,
a family emergency), whereas image messages usually have an informative caption beside
them.

---

## 6. Safety

Threat model: credential harvesting, account-suspension and fake-support pressure,
brand impersonation via lookalike domains, advance-fee and refund-bait fraud, prize and
investment fraud, malicious QR and obscured links, and prompt injection aimed at the
router itself.

**Personalisation can suppress but never promote.** `risk ≥ 0.35` blocks `notify`
outright, and the critic re-checks the invariant afterwards.

### Injection defence is layered, not a single rule

The test set contains **five** distinct injection phrasings, including structured-field
spoofing (`action=notify`, `verified_business=true`). All five are muted as scam.

Disabling the injection detector **entirely** still leaves all five muted — they fall
through to `otp_verification_flow` and `first_contact_sensitive_ask` on their
underlying fraud signals. A novel phrasing nobody anticipated is still caught.

### Polarity: the hardest safety problem here

Anti-fraud advisories quote fraud verbatim. A bank's "we will never ask for your OTP"
and a courier's "no payment or OTP is required" both trip a naive credential detector,
and muting them suppresses exactly the messages that protect users.

Adding an advisory exemption then created a **new** hole: prefixing "we never ask for
OTP" cloaked a real demand. The fix requires negation to precede the demand verb
*within its own clause*. Both the advisory and two cloaking attacks are permanent
tests.

---

## 7. The LLM experiment

Built, run live against `gpt-4o-mini`, and measured.

| configuration | sample action accuracy | overrides on 110 |
|---|---|---|
| deterministic core | **100%** | — |
| + LLM, stale thresholds | 96.7% | 7, all `notify → digest` |
| + LLM, corrected prompt | **100%** | 5, four of them `digest → notify` |

**The first measurement was wrong, and the cause was a bug in my own prompt.** The
facts block described the engine's thresholds with a hardcoded string —
`notify floor 0.62, mute ceiling 0.30` — written before they were tuned to 0.42 and
0.08. Borderline rows score 0.43–0.46, so the model was told every one sat below the
notify bar and reasoned accordingly. Sourcing thresholds from live config inverted the
direction of the bias, which is the clearest possible evidence of what caused it.

**Corrected, the model agrees with the deterministic engine on every labelled row**
(zero overrides on the 30 samples). It still makes 5 changes on unlabelled rows that
cannot be verified. Buying no measurable accuracy at the price of network dependence,
per-run variance and cost is a bad trade, so the deterministic path ships.

**Where it earned its keep was offline, as a gap finder.** Two early overrides were
correct and exposed real holes — `reactivation fee` was absent from the payment
lexicon, QR risk required a *detected image* QR rather than one named in prose, and
`account number` was missing as a credential. All are now handled deterministically and
pinned by tests, so the capability does not depend on an API key.

---

## 8. Falsified hypotheses

Kept deliberately, because they are the honest part of the record.

| hypothesis | expected | measured | outcome |
|---|---|---|---|
| penalise near-duplicate evidence | better recall | 71% → 55% | discarded |
| force multi-citation onto one counterparty | more coherent | recall dropped; reference cites across senders | discarded |
| LLM arbitration adds judgement | higher accuracy | 100% → 100%, zero overrides once the prompt bug was fixed | off by default: no measurable gain, loses determinism |
| *(my own)* LLM is a net negative | held for a day | caused by a stale hardcoded threshold in my prompt | conclusion retracted and documented |
| loosening first-contact to match reason strings | better reason score | became most-used reason, cited no evidence on 15% of rows | reverted |

The last one is the most instructive: optimising a proxy (exact string match on 30
rows) actively damaged the real objective (evidence coverage on 110 rows). Restoring
correctness cost one sample match and raised evidence coverage from 82% to 97%.

---

## 9. Engineering

- **Zero configuration** — no API key, no network at inference.
- **Deterministic** — asserted by test.
- **Cached** — content-addressed OCR/ASR; enable flags are part of the cache key, a bug
  found while building the ablation when disabled-OCR runs silently reused cached text.
- **Fails safe** — an unanalysable message is *deferred*, never dropped.
- **Degrades gracefully** — missing Tesseract, Whisper, embeddings or LLM each disable
  one capability while the rest continues.
- **Observable** — `run_metrics.json` records latency, throughput, cache hit rate,
  media coverage, safety interventions and critic repairs.
- **Auditable** — `--trace` emits every signal and threat per decision; `explain <id>`
  reconstructs one decision end to end.

### Test suite (60 tests)

| group | what it protects |
|---|---|
| data | schema, media resolution, quiet-hours midnight wrap |
| safety | fraud recall, five injection variants, impersonation vs link-shortener |
| polarity | advisories, courier reassurance, and two cloaking attacks |
| lexicon | whole-word matching (`emi` must not match inside `email`) |
| routing | accuracy floors, zero critical errors, per-user divergence, confidence bands |
| grounding | media is genuinely read, not inferred from captions |
| output | contract validity, allowed vocabulary, determinism |
| LLM | verdict parsing, safety veto, graceful failure |

---

## 10. Limitations

- Thresholds are tuned on 30 labelled rows. The plateau is wide, but a larger
  validation set would justify tighter bands.
- Lexicons are English-first. Hinglish scams in the test set are caught largely because
  they retain English tokens (`OTP`, `link`, `verification`) — that is partly luck, and
  a multilingual embedding plus a code-mixed lexicon is the highest-value next step.
- Repetition is per-user. A cross-user "this exact forward is circulating now" signal
  would catch chain content on first sight rather than second.
- Topic buckets in the memory model are coarse; learned interest embeddings would
  personalise promotions better than keyword buckets.
- Evidence recall sits at a measured ceiling of 71% for exact-ID match. The residual
  cases cite defensible alternatives, but a learned reranker over (semantic,
  structural, behavioural) features could plausibly beat the hand-set weights.
