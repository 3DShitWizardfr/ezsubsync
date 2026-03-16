"""Core synchronisation engine for subtitle alignment."""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .srt_parser import Subtitle, SubtitleFile, ms_to_timestamp

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[int, int, str], None]]


@dataclass
class SyncResult:
    """Result of a synchronisation operation."""

    synced: SubtitleFile
    method: str
    stats: Dict[str, object]


def _strip_tags(text: str) -> str:
    """Remove HTML/SRT formatting tags from text."""
    return re.sub(r"<[^>]+>", "", text).strip()


def _normalise(text: str) -> str:
    """Normalise text for comparison."""
    text = _strip_tags(text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def sync_by_sequence(
    reference: SubtitleFile,
    target: SubtitleFile,
    progress_cb: ProgressCallback = None,
) -> SyncResult:
    """Align target subtitles to reference by sequential order.

    Matches subtitles 1-to-1 by their position in the file.
    Extra subtitles in the target are time-shifted proportionally.

    Args:
        reference: Correctly timed subtitle file (language A).
        target: Subtitle file that needs re-timing (language B).
        progress_cb: Optional callback ``(current, total, message)``.

    Returns:
        A :class:`SyncResult` with the synchronised subtitles.
    """
    ref_subs = reference.subtitles
    tgt_subs = target.subtitles
    total = len(tgt_subs)

    if total == 0:
        return SyncResult(
            synced=SubtitleFile(subtitles=[], encoding=target.encoding),
            method="sequence",
            stats={"matched": 0, "total": 0},
        )

    synced: List[Subtitle] = []
    matched = 0

    for i, tgt in enumerate(tgt_subs):
        if progress_cb:
            progress_cb(i + 1, total, f"Aligning subtitle {i + 1}/{total} — {ms_to_timestamp(tgt.start_ms)}")

        if i < len(ref_subs):
            ref = ref_subs[i]
            synced.append(Subtitle(
                index=i + 1,
                start_ms=ref.start_ms,
                end_ms=ref.start_ms + tgt.duration_ms,
                text=tgt.text,
            ))
            matched += 1
        else:
            # Extrapolate from the last matched pair
            if ref_subs:
                last_ref = ref_subs[-1]
                offset = tgt.start_ms - tgt_subs[len(ref_subs) - 1].start_ms
                synced.append(Subtitle(
                    index=i + 1,
                    start_ms=last_ref.end_ms + offset,
                    end_ms=last_ref.end_ms + offset + tgt.duration_ms,
                    text=tgt.text,
                ))
            else:
                synced.append(Subtitle(
                    index=i + 1,
                    start_ms=tgt.start_ms,
                    end_ms=tgt.end_ms,
                    text=tgt.text,
                ))

    return SyncResult(
        synced=SubtitleFile(subtitles=synced, encoding=target.encoding),
        method="sequence",
        stats={"matched": matched, "total": total},
    )


def sync_by_similarity(
    reference: SubtitleFile,
    target: SubtitleFile,
    threshold: float = 0.4,
    progress_cb: ProgressCallback = None,
) -> SyncResult:
    """Align target subtitles to reference using content similarity.

    Uses ``difflib.SequenceMatcher`` to find the best match for each
    target subtitle among the reference subtitles.

    Args:
        reference: Correctly timed subtitle file (language A).
        target: Subtitle file that needs re-timing (language B).
        threshold: Minimum similarity ratio to accept a match.
        progress_cb: Optional callback ``(current, total, message)``.

    Returns:
        A :class:`SyncResult` with the synchronised subtitles.
    """
    ref_subs = reference.subtitles
    tgt_subs = target.subtitles
    total = len(tgt_subs)

    if total == 0 or len(ref_subs) == 0:
        return SyncResult(
            synced=SubtitleFile(subtitles=list(tgt_subs), encoding=target.encoding),
            method="similarity",
            stats={"matched": 0, "total": total, "threshold": threshold},
        )

    ref_texts = [_normalise(s.text) for s in ref_subs]
    synced: List[Subtitle] = []
    matched = 0

    for i, tgt in enumerate(tgt_subs):
        if progress_cb:
            progress_cb(i + 1, total, f"Matching subtitle {i + 1}/{total} — \"{tgt.text[:40]}\"")

        tgt_norm = _normalise(tgt.text)
        best_ratio = 0.0
        best_idx = -1

        for j, ref_norm in enumerate(ref_texts):
            ratio = difflib.SequenceMatcher(None, tgt_norm, ref_norm).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = j

        if best_ratio >= threshold and best_idx >= 0:
            ref = ref_subs[best_idx]
            synced.append(Subtitle(
                index=i + 1,
                start_ms=ref.start_ms,
                end_ms=ref.start_ms + tgt.duration_ms,
                text=tgt.text,
            ))
            matched += 1
        else:
            synced.append(Subtitle(
                index=i + 1,
                start_ms=tgt.start_ms,
                end_ms=tgt.end_ms,
                text=tgt.text,
            ))

    return SyncResult(
        synced=SubtitleFile(subtitles=synced, encoding=target.encoding),
        method="similarity",
        stats={"matched": matched, "total": total, "threshold": threshold},
    )


def sync_by_linear_shift(
    reference: SubtitleFile,
    target: SubtitleFile,
    progress_cb: ProgressCallback = None,
) -> SyncResult:
    """Align target to reference using a linear time mapping.

    Computes an affine transform (scale + offset) from matching the
    first and last subtitles, then applies it to all target entries.

    Args:
        reference: Correctly timed subtitle file.
        target: Subtitle file that needs re-timing.
        progress_cb: Optional callback ``(current, total, message)``.

    Returns:
        A :class:`SyncResult` with the synchronised subtitles.
    """
    ref_subs = reference.subtitles
    tgt_subs = target.subtitles
    total = len(tgt_subs)

    if total == 0 or len(ref_subs) == 0:
        return SyncResult(
            synced=SubtitleFile(subtitles=list(tgt_subs), encoding=target.encoding),
            method="linear_shift",
            stats={"scale": 1.0, "offset_ms": 0},
        )

    # Use first and last subtitle pairs for the linear mapping
    ref_start = ref_subs[0].start_ms
    ref_end = ref_subs[-1].start_ms
    tgt_start = tgt_subs[0].start_ms
    tgt_end = tgt_subs[-1].start_ms

    tgt_span = tgt_end - tgt_start
    ref_span = ref_end - ref_start

    if tgt_span == 0:
        scale = 1.0
    else:
        scale = ref_span / tgt_span

    offset = ref_start - tgt_start * scale

    synced: List[Subtitle] = []
    for i, tgt in enumerate(tgt_subs):
        new_start = int(tgt.start_ms * scale + offset)
        new_end = int(tgt.end_ms * scale + offset)

        if progress_cb:
            progress_cb(i + 1, total, f"Shifting subtitle {i + 1}/{total} — {ms_to_timestamp(tgt.start_ms)} → {ms_to_timestamp(max(0, new_start))}")

        synced.append(Subtitle(
            index=i + 1,
            start_ms=max(0, new_start),
            end_ms=max(0, new_end),
            text=tgt.text,
        ))

    return SyncResult(
        synced=SubtitleFile(subtitles=synced, encoding=target.encoding),
        method="linear_shift",
        stats={"scale": round(scale, 6), "offset_ms": round(offset, 2)},
    )


def sync_to_transcript(
    transcript_segments: List[Dict[str, object]],
    target: SubtitleFile,
    progress_cb: ProgressCallback = None,
) -> SyncResult:
    """Align target subtitles to a Whisper transcript.

    Each transcript segment is expected to have ``"start"`` and ``"end"``
    keys with float values in seconds, and ``"text"`` with the spoken text.

    Args:
        transcript_segments: Whisper transcript segments.
        target: Subtitle file that needs re-timing.
        progress_cb: Optional callback ``(current, total, message)``.

    Returns:
        A :class:`SyncResult` with the synchronised subtitles.
    """
    tgt_subs = target.subtitles
    total = len(tgt_subs)

    if total == 0 or len(transcript_segments) == 0:
        return SyncResult(
            synced=SubtitleFile(subtitles=list(tgt_subs), encoding=target.encoding),
            method="transcript",
            stats={"matched": 0, "total": total},
        )

    seg_texts = [_normalise(str(seg.get("text", ""))) for seg in transcript_segments]
    synced: List[Subtitle] = []
    matched = 0

    for i, tgt in enumerate(tgt_subs):
        if progress_cb:
            progress_cb(i + 1, total, f"Matching to transcript {i + 1}/{total} — \"{tgt.text[:40]}\"")

        tgt_norm = _normalise(tgt.text)
        best_ratio = 0.0
        best_idx = -1

        for j, seg_norm in enumerate(seg_texts):
            ratio = difflib.SequenceMatcher(None, tgt_norm, seg_norm).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = j

        if best_ratio >= 0.3 and best_idx >= 0:
            seg = transcript_segments[best_idx]
            start_s = float(seg.get("start", 0))
            synced.append(Subtitle(
                index=i + 1,
                start_ms=int(start_s * 1000),
                end_ms=int(start_s * 1000) + tgt.duration_ms,
                text=tgt.text,
            ))
            matched += 1
        else:
            synced.append(Subtitle(
                index=i + 1,
                start_ms=tgt.start_ms,
                end_ms=tgt.end_ms,
                text=tgt.text,
            ))

    return SyncResult(
        synced=SubtitleFile(subtitles=synced, encoding=target.encoding),
        method="transcript",
        stats={"matched": matched, "total": total},
    )


def cross_validate(
    synced: SubtitleFile,
    transcript_segments: List[Dict[str, object]],
    ocr_detections: Optional[List[Dict[str, object]]] = None,
    tolerance_ms: int = 2000,
    progress_cb: ProgressCallback = None,
) -> Tuple[int, int, List[str]]:
    """Cross-validate synced subtitles against transcript and OCR data.

    Args:
        synced: Synchronised subtitle file.
        transcript_segments: Whisper transcript segments.
        ocr_detections: Optional list of OCR detections with ``"time_ms"``
            and ``"text"`` keys.
        tolerance_ms: Allowed timing difference in milliseconds.
        progress_cb: Optional callback ``(current, total, message)``.

    Returns:
        A tuple of ``(confirmed, total, warnings)``.
    """
    subs = synced.subtitles
    total = len(subs)
    confirmed = 0
    warnings: List[str] = []

    for i, sub in enumerate(subs):
        if progress_cb:
            progress_cb(i + 1, total, f"Validating subtitle {i + 1}/{total} at {sub.start_str}")

        confirmed_by_transcript = False
        for seg in transcript_segments:
            seg_start_ms = int(float(seg.get("start", 0)) * 1000)
            if abs(sub.start_ms - seg_start_ms) <= tolerance_ms:
                confirmed_by_transcript = True
                break

        confirmed_by_ocr = False
        if ocr_detections:
            for det in ocr_detections:
                det_time = int(det.get("time_ms", 0))
                if abs(sub.start_ms - det_time) <= tolerance_ms:
                    confirmed_by_ocr = True
                    break

        if confirmed_by_transcript or confirmed_by_ocr:
            confirmed += 1
        else:
            warnings.append(
                f"Subtitle {i + 1} at {sub.start_str} could not be verified"
            )

    return confirmed, total, warnings
