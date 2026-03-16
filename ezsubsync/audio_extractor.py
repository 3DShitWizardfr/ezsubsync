"""Audio extraction from video files using ffmpeg."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[int, int, str], None]]

_TIME_PATTERN = re.compile(r"out_time_us=(-?\d+)")


def check_ffmpeg() -> bool:
    """Return *True* if ``ffmpeg`` is available on the system PATH."""
    return shutil.which("ffmpeg") is not None


def _get_video_duration_ms(video_path: Path) -> Optional[int]:
    """Return the duration of a video file in milliseconds via *ffprobe*."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(float(result.stdout.strip()) * 1000)
    except (subprocess.TimeoutExpired, ValueError):
        pass
    return None


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

    duration_ms = _get_video_duration_ms(video_path)

    if progress_cb:
        if duration_ms:
            progress_cb(0, 100, f"Extracting audio from video ({duration_ms // 1000}s)…")
        else:
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
    cmd.extend(["-progress", "pipe:1"])
    cmd.append(str(output_path))

    logger.info("Running: %s", " ".join(cmd))

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to start ffmpeg: {exc}") from exc

    # Drain stderr in a background thread to prevent pipe deadlock
    stderr_lines: List[str] = []
    stderr_thread = threading.Thread(
        target=lambda: stderr_lines.extend(process.stderr),  # type: ignore[union-attr]
        daemon=True,
    )
    stderr_thread.start()

    try:
        for line in process.stdout:  # type: ignore[union-attr]
            m = _TIME_PATTERN.match(line.strip())
            if m and progress_cb and duration_ms:
                time_us = int(m.group(1))
                if time_us > 0:
                    current_ms = time_us // 1000
                    pct = min(99, int(100 * current_ms / duration_ms))
                    progress_cb(
                        pct, 100,
                        f"Extracting audio: {pct}% — "
                        f"{current_ms // 1000}s / {duration_ms // 1000}s",
                    )

        process.wait(timeout=600)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise RuntimeError("ffmpeg timed out during audio extraction")
    finally:
        stderr_thread.join(timeout=5)

    if process.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {process.returncode}):\n{''.join(stderr_lines)}"
        )

    if progress_cb:
        progress_cb(100, 100, "Audio extraction complete")

    logger.info("Audio extracted to %s", output_path)
    return output_path
