"""Multimodal understanding: OCR for images, ASR for voice notes.

Both pipelines write into a shared on-disk JSON cache keyed by content hash, so
the expensive work happens once and every later run is fast and reproducible.

Design note: media understanding produces a *content* record that is deliberately
user-independent. The same poster can be a useful promotion for one user and
noise for another, so personalisation is applied downstream against this shared
record rather than being baked into extraction.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import MultimodalConfig, Settings

log = logging.getLogger(__name__)

# Requires an alphabetic TLD so that version strings and decimals such as
# "11.2" are not mistaken for domains.
URL_RE = re.compile(
    r"\b((?:https?://)?(?:[a-z0-9][a-z0-9\-]{0,62}\.)+[a-z]{2,24}(?:/[^\s]*)?)\b", re.I
)
WORDLIKE_RE = re.compile(r"^[A-Za-z][A-Za-z'\-]{2,}$")
AMOUNT_RE = re.compile(r"(?:rs\.?|inr|₹|\$)\s?[\d,]+(?:\.\d{1,2})?", re.I)
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?\d{10}\b")
DATE_RE = re.compile(
    r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2})\b",
    re.I,
)

def _open_image(path: Path):
    """Open an image by content, not by file extension.

    The dataset ships an AVIF file and a PNG both named `.jpg`, so anything that
    trusts the extension silently loses those assets.
    """
    from PIL import Image

    try:
        import pillow_avif  # noqa: F401  (registers the AVIF decoder with Pillow)
    except ImportError:
        pass
    return Image.open(path)


def _load_bgr(path: Path):
    """Load an image as an OpenCV BGR array, tolerating exotic encodings."""
    import cv2
    import numpy as np

    img = cv2.imread(str(path))
    if img is not None:
        return img
    try:
        with _open_image(path) as im:
            rgb = np.array(im.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:  # noqa: BLE001
        return None


# Zero-shot vision labels. CLIP scores the image against these captions so that
# text-free photos (a rack of clothes, a scenic poster) still get a semantic tag.
CLIP_PROMPTS: dict[str, str] = {
    "promotional_poster": "a marketing advertisement poster with a discount offer",
    "event_flyer": "a poster announcing an event with a date and venue",
    "retail_product_photo": "a photograph of clothes or products for sale in a shop",
    "official_document": "a scanned official document, form or bank statement",
    "app_screenshot": "a screenshot of a mobile app or computer screen",
    "payment_receipt": "a payment receipt, invoice or transaction confirmation screenshot",
    "chart_or_report": "a financial chart, graph or stock trading screen",
    "notice_or_circular": "a printed notice, circular or announcement letter",
    "personal_photo": "a casual personal photograph of people or a place",
    "identity_or_alert": "a missing person notice or public safety alert",
}


@dataclass
class ImageUnderstanding:
    media_id: str
    ok: bool = False
    ocr_text: str = ""
    ocr_char_count: int = 0
    ocr_word_count: int = 0
    ocr_confidence: float = 0.0
    text_density: float = 0.0
    layout: str = "unknown"
    scene_tags: list[str] = field(default_factory=list)
    scene_scores: dict[str, float] = field(default_factory=dict)
    urls: list[str] = field(default_factory=list)
    amounts: list[str] = field(default_factory=list)
    phones: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    has_qr: bool = False
    qr_payloads: list[str] = field(default_factory=list)
    width: int = 0
    height: int = 0
    error: str = ""

    def as_text(self) -> str:
        """A compact textual rendering used by the text engines and retrieval."""
        parts: list[str] = []
        if self.ocr_text.strip():
            parts.append(self.ocr_text.strip())
        if self.scene_tags:
            parts.append("[visual: " + ", ".join(self.scene_tags) + "]")
        if self.has_qr:
            parts.append("[contains QR code]")
        return "\n".join(parts)


@dataclass
class VoiceUnderstanding:
    media_id: str
    ok: bool = False
    transcript: str = ""
    language: str = ""
    language_confidence: float = 0.0
    duration_s: float = 0.0
    word_count: int = 0
    words_per_minute: float = 0.0
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    speech_ratio: float = 0.0
    error: str = ""

    def as_text(self) -> str:
        return self.transcript.strip()

    @property
    def is_rushed(self) -> bool:
        """Fast speech is a weak but real proxy for urgency or stress."""
        return self.words_per_minute >= 165.0

    @property
    def is_short_burst(self) -> bool:
        return 0 < self.duration_s <= 12.0


class MediaCache:
    """Content-addressed JSON cache so reruns never redo OCR or ASR."""

    def __init__(self, cache_dir: Path, namespace: str) -> None:
        self.dir = cache_dir / namespace
        self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _digest(path: Path, salt: str) -> str:
        h = hashlib.sha256()
        h.update(salt.encode("utf-8"))
        h.update(path.name.encode("utf-8"))
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:24]

    def get(self, path: Path, salt: str) -> dict | None:
        f = self.dir / f"{self._digest(path, salt)}.json"
        if not f.exists():
            return None
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def put(self, path: Path, salt: str, payload: dict) -> None:
        f = self.dir / f"{self._digest(path, salt)}.json"
        try:
            f.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError as exc:
            log.warning("cache write failed for %s: %s", path.name, exc)


def _locate_tesseract(cfg: MultimodalConfig) -> str | None:
    if cfg.tesseract_cmd:
        return cfg.tesseract_cmd
    env = os.environ.get("TESSERACT_CMD")
    if env and Path(env).exists():
        return env
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract",
    ):
        if Path(candidate).exists():
            return candidate
    return None


class ImageAnalyzer:
    """OCR + QR + zero-shot scene understanding for image messages."""

    def __init__(self, settings: Settings) -> None:
        self.cfg = settings.multimodal
        self.cache = MediaCache(settings.cache_dir, "images")
        self._tess = _locate_tesseract(self.cfg)
        self._clip = None
        self._clip_text_emb = None
        self._clip_failed = False
        if self.cfg.enable_ocr and not self._tess:
            log.warning("tesseract not found - image OCR degraded to metadata only")

    def _load_clip(self):
        if self._clip is not None or self._clip_failed:
            return self._clip
        try:
            from sentence_transformers import SentenceTransformer

            self._clip = SentenceTransformer("clip-ViT-B-32")
            self._clip_text_emb = self._clip.encode(
                list(CLIP_PROMPTS.values()), convert_to_numpy=True, normalize_embeddings=True
            )
        except Exception as exc:  # noqa: BLE001 - optional dependency path
            log.warning("CLIP unavailable, scene tagging disabled: %s", exc)
            self._clip_failed = True
            self._clip = None
        return self._clip

    def _ocr(self, path: Path) -> tuple[str, float]:
        """Run OCR over several preprocessings and keep the best-scoring pass."""
        if not self._tess:
            return "", 0.0
        try:
            import cv2
            import numpy as np
            import pytesseract

            pytesseract.pytesseract.tesseract_cmd = self._tess
            raw = _load_bgr(path)
            if raw is None:
                return "", 0.0

            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            if max(h, w) < 1000:  # upscale small images so glyphs survive binarisation
                scale = 1000.0 / max(h, w)
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            variants = [
                gray,
                cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1],
                cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
                ),
            ]

            best_text, best_conf = "", 0.0
            for variant in variants:
                data = pytesseract.image_to_data(
                    variant,
                    lang=self.cfg.ocr_languages,
                    config="--oem 3 --psm 6",
                    output_type=pytesseract.Output.DICT,
                )
                words, confs = [], []
                for token, conf in zip(data["text"], data["conf"]):
                    token = (token or "").strip()
                    try:
                        c = float(conf)
                    except (TypeError, ValueError):
                        continue
                    if token and c >= 40:
                        words.append(token)
                        confs.append(c)
                if not words:
                    continue
                mean_conf = sum(confs) / len(confs) / 100.0
                # Prefer passes that recover more readable text, not just high
                # confidence on a couple of tokens.
                score = mean_conf * (len(words) ** 0.4)
                if score > best_conf:
                    best_conf, best_text = score, " ".join(words)

            text = re.sub(r"[ \t]+", " ", best_text).strip()
            conf = min(1.0, best_conf / 3.0) if text else 0.0
            return text, round(conf, 3)
        except Exception as exc:  # noqa: BLE001 - OCR must never break the run
            log.warning("OCR failed for %s: %s", path.name, exc)
            return "", 0.0

    def _detect_qr(self, path: Path) -> tuple[bool, list[str]]:
        try:
            import cv2

            img = _load_bgr(path)
            if img is None:
                return False, []
            detector = cv2.QRCodeDetector()
            ok, decoded, points, _ = detector.detectAndDecodeMulti(img)
            if ok and points is not None:
                payloads = [d for d in decoded if d]
                return True, payloads
            return False, []
        except Exception:  # noqa: BLE001
            return False, []

    def _scene_tags(self, path: Path) -> tuple[list[str], dict[str, float]]:
        model = self._load_clip()
        if model is None:
            return [], {}
        try:
            import numpy as np
            from PIL import Image

            with _open_image(path) as im:
                emb = model.encode([im.convert("RGB")], convert_to_numpy=True, normalize_embeddings=True)
            sims = (emb @ self._clip_text_emb.T)[0]
            # Softmax over CLIP similarities gives a comparable distribution.
            exp = np.exp((sims - sims.max()) * 100.0)
            probs = exp / exp.sum()
            keys = list(CLIP_PROMPTS.keys())
            scores = {k: round(float(p), 4) for k, p in zip(keys, probs)}
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            tags = [k for k, v in ranked[:2] if v >= 0.12]
            return tags, scores
        except Exception as exc:  # noqa: BLE001
            log.warning("scene tagging failed for %s: %s", path.name, exc)
            return [], {}

    @staticmethod
    def _clean_ocr(text: str) -> tuple[str, int]:
        """Drop glyph noise token by token, keeping words, numbers and symbols.

        Whole-text rejection loses sparse but meaningful posters (a missing
        person notice reading only "Missing from 4-11-2025"), while keeping raw
        output lets a photograph's stray marks leak into every text engine.
        Filtering per token preserves the former and removes the latter.
        """
        kept: list[str] = []
        for raw in text.split():
            tok = raw.strip(".,:;!?()[]{}<>\"'|\\/*_~`^=+")
            if not tok:
                continue
            if WORDLIKE_RE.match(tok):
                kept.append(tok)
            elif re.fullmatch(r"[₹$%#@&]?[\d][\d,.:/\-]*%?", tok):
                kept.append(tok)
            elif re.fullmatch(r"[A-Z]{2,}", tok):
                kept.append(tok)
        words = sum(1 for t in kept if WORDLIKE_RE.match(t))
        return " ".join(kept), words

    @staticmethod
    def _classify_layout(text: str, tags: list[str], words: int) -> str:
        """Classify layout from recovered text plus the zero-shot scene tags.

        Word count is used rather than character density because dense-but-noisy
        OCR on a photograph would otherwise masquerade as a text-heavy poster.
        """
        low = text.lower()
        if words >= 4:
            if any(
                k in low
                for k in ("statement", "consent form", "certificate", "page no", "account no", "permission")
            ):
                return "document"
            if any(k in low for k in ("receipt", "invoice", "transaction id", "txn", "paid to")):
                return "receipt"
            if any(k in low for k in ("inbox", "meeting notes", "calendar", "webinar")):
                return "screenshot"
            if any(
                k in low
                for k in ("off", "sale", "offer", "book now", "discount", "cashback", "unbeatable", "price")
            ):
                return "promotional_poster"
        if "app_screenshot" in tags:
            return "screenshot"
        if "official_document" in tags and words >= 4:
            return "document"
        if words >= 8:
            return "poster"
        if "retail_product_photo" in tags or "personal_photo" in tags:
            return "photo"
        return "unknown"

    def analyze(self, media_id: str, path: Path) -> ImageUnderstanding:
        # Enable flags belong in the key: otherwise an OCR-disabled run would
        # silently reuse cached OCR output and look identical to a full run.
        salt = f"img-v2-{self.cfg.ocr_languages}-ocr{int(self.cfg.enable_ocr)}"
        cached = self.cache.get(path, salt)
        if cached:
            return ImageUnderstanding(**cached)

        out = ImageUnderstanding(media_id=media_id)
        try:
            with _open_image(path) as im:
                out.width, out.height = im.size
        except Exception as exc:  # noqa: BLE001
            out.error = f"open failed: {exc}"

        raw_text, conf = self._ocr(path) if self.cfg.enable_ocr else ("", 0.0)
        text, words = self._clean_ocr(raw_text)
        if words < 2:
            text, conf, words = "", 0.0, 0

        out.ocr_text = text
        out.ocr_confidence = conf
        out.ocr_char_count = len(text)
        out.ocr_word_count = words
        area = max(1, out.width * out.height)
        out.text_density = round(len(text) / (area / 10000.0), 5) if area else 0.0

        out.has_qr, out.qr_payloads = self._detect_qr(path)
        out.scene_tags, out.scene_scores = self._scene_tags(path)
        out.layout = self._classify_layout(text, out.scene_tags, words)

        blob = f"{text} {' '.join(out.qr_payloads)}"
        out.urls = sorted({m.group(1).lower() for m in URL_RE.finditer(blob)})[:8]
        out.amounts = sorted({m.group(0) for m in AMOUNT_RE.finditer(blob)})[:8]
        out.phones = sorted({m.group(0) for m in PHONE_RE.finditer(blob)})[:5]
        out.dates = sorted({m.group(1) for m in DATE_RE.finditer(blob)})[:6]
        out.ok = bool(text or out.scene_tags)

        self.cache.put(path, salt, asdict(out))
        return out


class VoiceAnalyzer:
    """Local speech-to-text plus lightweight prosody features."""

    def __init__(self, settings: Settings) -> None:
        self.cfg = settings.multimodal
        self.cache = MediaCache(settings.cache_dir, "voice")
        self._model = None
        self._failed = False

    def _load(self):
        if self._model is not None or self._failed:
            return self._model
        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.cfg.whisper_model, device="cpu", compute_type=self.cfg.whisper_compute_type
            )
            log.info("loaded whisper model '%s'", self.cfg.whisper_model)
        except Exception as exc:  # noqa: BLE001 - ASR is optional
            log.warning("faster-whisper unavailable, voice transcription disabled: %s", exc)
            self._failed = True
        return self._model

    @staticmethod
    def _duration(path: Path) -> float:
        if path.suffix.lower() == ".wav":
            try:
                with wave.open(str(path), "rb") as w:
                    return w.getnframes() / float(w.getframerate() or 1)
            except Exception:  # noqa: BLE001
                return 0.0
        ffprobe = shutil.which("ffprobe")
        if ffprobe:
            try:
                res = subprocess.run(
                    [ffprobe, "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=nw=1:nk=1", str(path)],
                    capture_output=True, text=True, timeout=30, check=False,
                )
                return float(res.stdout.strip() or 0.0)
            except Exception:  # noqa: BLE001
                return 0.0
        return 0.0

    def analyze(self, media_id: str, path: Path) -> VoiceUnderstanding:
        salt = f"voice-v2-{self.cfg.whisper_model}-asr{int(self.cfg.enable_asr)}"
        cached = self.cache.get(path, salt)
        if cached:
            return VoiceUnderstanding(**cached)

        out = VoiceUnderstanding(media_id=media_id)
        model = self._load() if self.cfg.enable_asr else None
        if model is None:
            out.duration_s = self._duration(path)
            out.error = "asr_unavailable"
            self.cache.put(path, salt, asdict(out))
            return out

        try:
            segments, info = model.transcribe(
                str(path),
                beam_size=self.cfg.whisper_beam_size,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            texts, logprobs, no_speech, voiced = [], [], [], 0.0
            for seg in segments:
                texts.append(seg.text.strip())
                logprobs.append(getattr(seg, "avg_logprob", 0.0))
                no_speech.append(getattr(seg, "no_speech_prob", 0.0))
                voiced += max(0.0, seg.end - seg.start)

            out.transcript = re.sub(r"\s+", " ", " ".join(texts)).strip()
            out.language = getattr(info, "language", "") or ""
            out.language_confidence = round(float(getattr(info, "language_probability", 0.0)), 3)
            out.duration_s = round(float(getattr(info, "duration", 0.0) or self._duration(path)), 2)
            out.word_count = len(out.transcript.split())
            out.avg_logprob = round(sum(logprobs) / len(logprobs), 3) if logprobs else 0.0
            out.no_speech_prob = round(sum(no_speech) / len(no_speech), 3) if no_speech else 0.0
            out.speech_ratio = round(voiced / out.duration_s, 3) if out.duration_s > 0 else 0.0
            out.words_per_minute = (
                round(out.word_count / (voiced / 60.0), 1) if voiced > 1.0 else 0.0
            )
            out.ok = bool(out.transcript)
        except Exception as exc:  # noqa: BLE001
            out.error = f"transcribe failed: {exc}"
            log.warning("ASR failed for %s: %s", path.name, exc)

        self.cache.put(path, salt, asdict(out))
        return out


@dataclass
class MediaIndex:
    """Resolved understanding for every media asset referenced by a message."""

    images: dict[str, ImageUnderstanding] = field(default_factory=dict)
    voices: dict[str, VoiceUnderstanding] = field(default_factory=dict)

    def text_for(self, media_id: str) -> str:
        if media_id in self.images:
            return self.images[media_id].as_text()
        if media_id in self.voices:
            return self.voices[media_id].as_text()
        return ""


def build_media_index(settings: Settings, dataset, media_ids: set[str] | None = None) -> MediaIndex:
    """Analyze every referenced media asset, reusing cache where possible."""
    index = MediaIndex()
    img_analyzer = ImageAnalyzer(settings)
    voice_analyzer = VoiceAnalyzer(settings)

    wanted_images = [
        (mid, a) for mid, a in dataset.images.items() if media_ids is None or mid in media_ids
    ]
    wanted_voices = [
        (mid, a) for mid, a in dataset.voice_notes.items() if media_ids is None or mid in media_ids
    ]

    for mid, asset in wanted_images:
        path = settings.dataset_dir / asset.file_path
        if not path.exists():
            log.warning("missing image file: %s", path)
            continue
        index.images[mid] = img_analyzer.analyze(mid, path)

    for mid, asset in wanted_voices:
        path = settings.dataset_dir / asset.file_path
        if not path.exists():
            log.warning("missing audio file: %s", path)
            continue
        index.voices[mid] = voice_analyzer.analyze(mid, path)

    log.info("media index: %d images, %d voice notes", len(index.images), len(index.voices))
    return index
