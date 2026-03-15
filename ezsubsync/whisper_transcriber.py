"""Speech recognition using the local open-source Whisper model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[int, int, str], None]]


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
        progress_cb(10, 100, "Transcribing audio…")

    options: Dict[str, Any] = {}
    if language:
        options["language"] = language

    result = model.transcribe(str(audio_path), **options)

    segments: List[Dict[str, object]] = []
    raw_segments = result.get("segments", [])
    total_seg = len(raw_segments)

    for i, seg in enumerate(raw_segments):
        if progress_cb:
            pct = 10 + int(90 * (i + 1) / max(total_seg, 1))
            progress_cb(pct, 100, f"Processing segment {i + 1}/{total_seg}")

        segments.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": str(seg.get("text", "")).strip(),
        })

    if progress_cb:
        progress_cb(100, 100, "Transcription complete")

    logger.info("Transcribed %d segments from %s", len(segments), audio_path)
    return segments
