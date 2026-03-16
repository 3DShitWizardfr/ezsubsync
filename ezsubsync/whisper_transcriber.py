"""Speech recognition using the local open-source Whisper model."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[int, int, str], None]]


class _ProgressWriter:
    """Intercept Whisper's ``verbose=True`` stdout and forward to a callback."""

    def __init__(self, original_stdout: Any, progress_cb: Callable[[int, int, str], None]) -> None:
        self._original = original_stdout
        self._cb = progress_cb
        self._buf = ""
        self._count = 0

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            stripped = line.strip()
            if stripped:
                self._count += 1
                self._cb(
                    min(10 + self._count, 99), 100,
                    f"Whisper: {stripped}",
                )
        return len(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


def _load_whisper_model(model_name: str = "base") -> Any:
    """Load a Whisper model, raising a helpful error if not installed."""
    try:
        import whisper  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "The 'openai-whisper' package is required for transcription. "
            "Install it with: pip install openai-whisper"
        ) from exc
    logger.info("Loading Whisper model '%s'…", model_name)
    return whisper.load_model(model_name)


def transcribe(
    audio_path: str | Path,
    model_name: str = "base",
    language: Optional[str] = None,
    progress_cb: ProgressCallback = None,
) -> List[Dict[str, object]]:
    """Transcribe an audio file using a local Whisper model.

    Args:
        audio_path: Path to a WAV audio file.
        model_name: Whisper model size (``tiny``, ``base``, ``small``,
            ``medium``, ``large``).
        language: Optional ISO language code to hint the language.
        progress_cb: Optional callback ``(current, total, message)``.

    Returns:
        A list of segment dicts, each containing ``"start"`` (float seconds),
        ``"end"`` (float seconds), and ``"text"`` (str).

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ImportError: If the ``whisper`` package is not installed.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if progress_cb:
        progress_cb(0, 100, f"Loading Whisper model '{model_name}'…")

    model = _load_whisper_model(model_name)

    if progress_cb:
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        progress_cb(10, 100, f"Transcribing {audio_path.name} ({file_size_mb:.1f} MB) — this may take a while…")

    options: Dict[str, Any] = {"verbose": True}
    if language:
        options["language"] = language

    # Capture Whisper's verbose stdout and forward each decoded line
    # to the progress callback so the user sees real-time transcription.
    if progress_cb:
        writer = _ProgressWriter(sys.stdout, progress_cb)
        old_stdout = sys.stdout
        sys.stdout = writer  # type: ignore[assignment]
        try:
            result = model.transcribe(str(audio_path), **options)
        finally:
            sys.stdout = old_stdout
    else:
        result = model.transcribe(str(audio_path), **options)

    segments: List[Dict[str, object]] = []
    raw_segments = result.get("segments", [])

    for seg in raw_segments:
        segments.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": str(seg.get("text", "")).strip(),
        })

    if progress_cb:
        progress_cb(100, 100, f"Transcription complete — {len(segments)} segments")

    logger.info("Transcribed %d segments from %s", len(segments), audio_path)
    return segments
