"""Pytest configuration.

Puts `code/` on the import path so tests can `import router.*` regardless of the
directory pytest is invoked from, and quiets the model libraries whose progress
output would otherwise bury the test results.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

for noisy in (
    "sentence_transformers",
    "transformers",
    "faster_whisper",
    "urllib3",
    "httpx",
    "PIL",
):
    logging.getLogger(noisy).setLevel(logging.ERROR)
