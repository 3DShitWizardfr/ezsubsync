"""SRT subtitle file parser and writer."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

TIMESTAMP_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)
TIMECODE_LINE_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})"
)


@dataclass
class Subtitle:
    """A single subtitle entry."""

    index: int
    start_ms: int
    end_ms: int
    text: str

    @property
    def start_str(self) -> str:
        return ms_to_timestamp(self.start_ms)

    @property
    def end_str(self) -> str:
        return ms_to_timestamp(self.end_ms)

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass
class SubtitleFile:
    """A collection of subtitle entries."""

    subtitles: List[Subtitle] = field(default_factory=list)
    encoding: str = "utf-8"

    def __len__(self) -> int:
        return len(self.subtitles)

    def __getitem__(self, idx: int) -> Subtitle:
        return self.subtitles[idx]

    def __iter__(self):
        return iter(self.subtitles)


def timestamp_to_ms(ts: str) -> int:
    """Convert an SRT timestamp string to milliseconds.

    Args:
        ts: Timestamp in the format ``HH:MM:SS,mmm``.

    Returns:
        Total milliseconds.

    Raises:
        ValueError: If the timestamp format is invalid.
    """
    match = TIMESTAMP_RE.match(ts.strip())
    if not match:
        raise ValueError(f"Invalid timestamp format: {ts!r}")
    hours, minutes, seconds, millis = (int(g) for g in match.groups())
    return hours * 3_600_000 + minutes * 60_000 + seconds * 1_000 + millis


def ms_to_timestamp(ms: int) -> str:
    """Convert milliseconds to an SRT timestamp string.

    Args:
        ms: Total milliseconds (non-negative).

    Returns:
        Timestamp formatted as ``HH:MM:SS,mmm``.
    """
    if ms < 0:
        ms = 0
    hours = ms // 3_600_000
    ms %= 3_600_000
    minutes = ms // 60_000
    ms %= 60_000
    seconds = ms // 1_000
    millis = ms % 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def parse_srt(filepath: str | Path, encoding: Optional[str] = None) -> SubtitleFile:
    """Parse an SRT subtitle file.

    Args:
        filepath: Path to the ``.srt`` file.
        encoding: Character encoding. Auto-detected if *None*.

    Returns:
        A :class:`SubtitleFile` containing the parsed subtitles.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Subtitle file not found: {filepath}")

    if encoding is None:
        encoding = _detect_encoding(filepath)

    content = filepath.read_text(encoding=encoding, errors="replace")
    return parse_srt_string(content, encoding=encoding)


def parse_srt_string(content: str, encoding: str = "utf-8") -> SubtitleFile:
    """Parse SRT content from a string.

    Args:
        content: Raw SRT text.
        encoding: Encoding label to store on the result.

    Returns:
        A :class:`SubtitleFile` containing the parsed subtitles.
    """
    subtitles: List[Subtitle] = []
    # Normalise line endings and split into blocks
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\n+", content.strip())

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue

        # Find the timecode line
        timecode_idx: Optional[int] = None
        for i, line in enumerate(lines):
            if TIMECODE_LINE_RE.search(line):
                timecode_idx = i
                break

        if timecode_idx is None:
            continue

        # Index is the line before the timecode (if present and numeric)
        index = len(subtitles) + 1
        if timecode_idx > 0:
            idx_candidate = lines[timecode_idx - 1].strip()
            if idx_candidate.isdigit():
                index = int(idx_candidate)

        tc_match = TIMECODE_LINE_RE.search(lines[timecode_idx])
        if not tc_match:
            continue

        start_ms = timestamp_to_ms(tc_match.group(1))
        end_ms = timestamp_to_ms(tc_match.group(2))
        text = "\n".join(lines[timecode_idx + 1:]).strip()

        subtitles.append(Subtitle(
            index=index,
            start_ms=start_ms,
            end_ms=end_ms,
            text=text,
        ))

    return SubtitleFile(subtitles=subtitles, encoding=encoding)


def write_srt(sub_file: SubtitleFile, filepath: str | Path, encoding: Optional[str] = None) -> None:
    """Write subtitles to an SRT file.

    Args:
        sub_file: The subtitle data to write.
        filepath: Destination path.
        encoding: Character encoding (defaults to the encoding stored on *sub_file*).
    """
    filepath = Path(filepath)
    enc = encoding or sub_file.encoding or "utf-8"
    filepath.write_text(format_srt(sub_file), encoding=enc)


def format_srt(sub_file: SubtitleFile) -> str:
    """Format subtitles as an SRT string.

    Args:
        sub_file: The subtitle data.

    Returns:
        A valid SRT-formatted string.
    """
    parts: List[str] = []
    for i, sub in enumerate(sub_file.subtitles, start=1):
        parts.append(
            f"{i}\n{sub.start_str} --> {sub.end_str}\n{sub.text}\n"
        )
    return "\n".join(parts)


def _detect_encoding(filepath: Path) -> str:
    """Best-effort encoding detection for a file."""
    raw = filepath.read_bytes()[:4096]
    # Check BOM
    if raw[:3] == b"\xef\xbb\xbf":
        return "utf-8-sig"
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return "utf-16"
    # Try UTF-8
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass
    return "latin-1"
