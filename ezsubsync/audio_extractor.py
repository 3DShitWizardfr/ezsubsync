"""Audio extraction from video files using ffmpeg."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[int, int, str], None]]


def check_ffmpeg() -> bool:
    """Return *True* if ``ffmpeg`` is available on the system PATH."""
    return shutil.which("ffmpeg") is not None


def extract_audio(
    video_path: str | Path,
    output_path: str | Path | None = None,
    sample_rate: int = 16000,
    mono: bool = True,
    progress_cb: ProgressCallback = None,
) -> Path:
    """Extract audio from a video file using *ffmpeg*.

    Args:
        video_path: Path to the input video file.
        output_path: Where to write the WAV file.  Defaults to
            ``<video_stem>_audio.wav`` next to the video.
        sample_rate: Output sample rate (Hz).  Whisper expects 16 000.
        mono: If *True*, down-mix to mono.
        progress_cb: Optional progress callback.

    Returns:
        The :class:`Path` to the extracted WAV file.

    Raises:
        FileNotFoundError: If the video file or *ffmpeg* cannot be found.
        RuntimeError: If the extraction command fails.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not check_ffmpeg():
        raise FileNotFoundError(
            "ffmpeg is not installed or not on PATH. "
            "Please install ffmpeg: https://ffmpeg.org/download.html"
        )

    if output_path is None:
        output_path = video_path.with_name(f"{video_path.stem}_audio.wav")
    output_path = Path(output_path)

    if progress_cb:
        progress_cb(0, 100, "Extracting audio from video…")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
    ]
    if mono:
        cmd.extend(["-ac", "1"])
    cmd.append(str(output_path))

    logger.info("Running: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ffmpeg timed out during audio extraction") from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}):\n{result.stderr}"
        )

    if progress_cb:
        progress_cb(100, 100, "Audio extraction complete")

    logger.info("Audio extracted to %s", output_path)
    return output_path
