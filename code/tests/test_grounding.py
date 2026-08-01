"""Grounding tests: proof the system reads the media, not just the caption.

A multimodal router can score well while being quietly blind - if every image
message carries a descriptive caption, a text-only system looks identical to one
that actually performs OCR. These tests close that hole by asserting that the
media pipeline recovers information the caption does not contain, that specific
known content is genuinely extracted, and that decisions move when the media is
taken away.

Run:  python -m pytest code/tests/test_grounding.py -q
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.config import DEFAULT_SETTINGS  # noqa: E402
from router.content import build_content  # noqa: E402
from router.dataio import load_dataset  # noqa: E402
from router.multimodal import build_media_index  # noqa: E402
from router.pipeline import NotificationRouter  # noqa: E402


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DEFAULT_SETTINGS)


@pytest.fixture(scope="module")
def media(dataset):
    return build_media_index(DEFAULT_SETTINGS, dataset)


def _tokens(text: str) -> set[str]:
    return {t.strip(".,:;!?()[]\"'").lower() for t in text.split() if len(t) > 3}


# ---------------------------------------------------------------- OCR grounding


def test_ocr_recovers_text_that_is_not_in_the_caption(dataset, media):
    """At least some image text must be information the caption never gave us."""
    novel_total = 0
    for message in dataset.messages:
        if message.media_type != "image" or message.media_id not in media.images:
            continue
        img = media.images[message.media_id]
        if not img.ocr_text:
            continue
        novel = _tokens(img.ocr_text) - _tokens(message.message_text or "")
        novel_total += len(novel)
    assert novel_total > 100, (
        f"only {novel_total} novel tokens recovered from images - "
        "OCR may not be contributing beyond the captions"
    )


@pytest.mark.parametrize(
    "media_id, expected_any",
    [
        ("img_011", ("consent", "trip", "permission")),   # school consent form
        ("img_016", ("hdfc", "bank", "account")),         # bank statement
        ("img_012", ("bombay", "research", "internship")),  # university notice
        ("img_010", ("amazon", "prime", "cashback")),     # retail promotion
    ],
)
def test_specific_image_content_is_actually_extracted(media, media_id, expected_any):
    """Named documents must yield their own vocabulary, not a generic guess."""
    text = media.images[media_id].ocr_text.lower()
    assert any(word in text for word in expected_any), (
        f"{media_id}: none of {expected_any} found in OCR output {text[:120]!r}"
    )


def test_image_files_with_wrong_extensions_still_parse(media):
    """One asset is AVIF and another PNG, both named .jpg."""
    assert media.images["img_020"].ok, "AVIF-mislabelled image failed to parse"
    assert media.images["img_023"].ok, "PNG-mislabelled image failed to parse"


# ---------------------------------------------------------------- ASR grounding


def test_voice_notes_are_genuinely_transcribed(media):
    """Voice messages carry no text at all, so ASR is the only signal source."""
    transcribed = [v for v in media.voices.values() if v.ok and v.word_count >= 5]
    assert len(transcribed) >= 10, (
        f"only {len(transcribed)} voice notes produced usable transcripts"
    )


@pytest.mark.parametrize(
    "media_id, expected_any",
    [
        ("vn_008", ("otp", "bank", "blocked")),        # OTP fraud call
        ("vn_004", ("school", "gate", "pickup")),      # school transport
        ("vn_002", ("call", "clinic", "dad")),         # family urgency
        ("vn_009", ("airport", "pickup", "driver")),   # travel update
    ],
)
def test_specific_voice_content_is_actually_transcribed(media, media_id, expected_any):
    text = media.voices[media_id].transcript.lower()
    assert any(word in text for word in expected_any), (
        f"{media_id}: none of {expected_any} in transcript {text[:120]!r}"
    )


def test_speaking_rate_separates_urgent_from_calm_speech(media):
    """Prosody is a real signal, not a decorative field."""
    urgent = media.voices["vn_002"]     # "Please call now, Dad is unwell"
    calm = media.voices["vn_001"]       # "Had dinner, call when free, nothing urgent"
    assert urgent.words_per_minute > calm.words_per_minute, (
        f"urgent={urgent.words_per_minute} calm={calm.words_per_minute}"
    )


# ---------------------------------------------------------------- decision grounding


def test_removing_media_changes_real_decisions(dataset, media):
    """The strongest proof: blind the system and watch the answers move."""
    seeing = NotificationRouter(DEFAULT_SETTINGS, dataset, media)
    blind_settings = replace(
        DEFAULT_SETTINGS,
        multimodal=replace(DEFAULT_SETTINGS.multimodal, enable_ocr=False, enable_asr=False),
    )
    blind_media = build_media_index(blind_settings, dataset)
    blind = NotificationRouter(blind_settings, dataset, blind_media)

    seeing_decisions, _, _ = seeing.run()
    blind_decisions, _, _ = blind.run()

    before = {d.message_id: (d.action, d.message_type) for d in seeing_decisions}
    after = {d.message_id: (d.action, d.message_type) for d in blind_decisions}
    changed = [mid for mid in before if before[mid] != after[mid]]

    assert changed, "decisions are identical without media - the pipeline is decorative"


def test_voice_only_messages_are_not_routed_blindly(dataset, media):
    """A voice note has no text; without ASR it cannot be typed meaningfully."""
    router = NotificationRouter(DEFAULT_SETTINGS, dataset, media)
    voice_messages = [m for m in dataset.messages if m.media_type == "voice"]
    assert voice_messages

    typed = 0
    for message in voice_messages:
        content = build_content(message, media)
        # The content the router reasons over must come from the audio itself.
        if content.combined.strip():
            typed += 1
    assert typed == len(voice_messages), (
        f"{len(voice_messages) - typed} voice messages reached routing with no content"
    )


def test_scam_hidden_in_audio_is_still_caught(dataset, media):
    """vn_008 is an OTP scam delivered by voice - text alone would miss it."""
    router = NotificationRouter(DEFAULT_SETTINGS, dataset, media)
    target = next(
        (m for m in dataset.messages if m.media_id == "vn_008"), None
    )
    if target is None:
        pytest.skip("vn_008 not referenced by any routed message")

    from router.pipeline import RunStats

    routed = router.route_one(target, RunStats())
    assert routed.decision.action == "mute"
    assert routed.decision.message_type == "scam"
