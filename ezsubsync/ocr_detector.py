"""Optical character recognition for on-screen text detection in video frames."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[int, int, str], None]]


def _check_dependencies() -> None:
    """Verify that OpenCV and Tesseract are available."""
    try:
        import cv2  # noqa: F401  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "The 'opencv-python' package is required for OCR. "
            "Install it with: pip install opencv-python"
        ) from exc
    try:
        import pytesseract  # noqa: F401  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "The 'pytesseract' package is required for OCR. "
            "Install it with: pip install pytesseract"
        ) from exc


def detect_onscreen_text(
    video_path: str | Path,
    interval_sec: float = 2.0,
    language: str = "eng",
    progress_cb: ProgressCallback = None,
) -> List[Dict[str, Any]]:
    """Detect on-screen text in a video at regular intervals.

    Extracts frames at the given interval, applies pre-processing, and
    runs Tesseract OCR to find titles, location names, character names,
    or other burned-in text.

    Args:
        video_path: Path to the video file.
        interval_sec: Seconds between sampled frames.
        language: Tesseract language code (e.g. ``eng``, ``fra``).
        progress_cb: Optional callback ``(current, total, message)``.

    Returns:
        A list of dicts with ``"time_ms"`` (int) and ``"text"`` (str).

    Raises:
        FileNotFoundError: If the video file does not exist.
        ImportError: If required packages are missing.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    _check_dependencies()

    import cv2  # type: ignore[import-untyped]
    import pytesseract  # type: ignore[import-untyped]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps else 0
    frame_interval = int(fps * interval_sec)

    if progress_cb:
        progress_cb(0, 100, "Starting OCR scan…")

    detections: List[Dict[str, Any]] = []
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                time_ms = int(frame_idx / fps * 1000)

                # Convert to grey-scale and threshold for better OCR
                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(grey, 150, 255, cv2.THRESH_BINARY)

                text = pytesseract.image_to_string(
                    thresh, lang=language
                ).strip()

                if text:
                    detections.append({"time_ms": time_ms, "text": text})
                    logger.debug("OCR @ %d ms: %s", time_ms, text[:80])

                if progress_cb and duration_sec > 0:
                    pct = min(100, int(100 * (time_ms / 1000) / duration_sec))
                    progress_cb(pct, 100, f"OCR scanning at {time_ms / 1000:.1f}s")

            frame_idx += 1
    finally:
        cap.release()

    if progress_cb:
        progress_cb(100, 100, "OCR scan complete")

    logger.info("OCR detected text in %d frames from %s", len(detections), video_path)
    return detections
