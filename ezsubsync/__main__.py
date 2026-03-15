"""CLI entry point for ezsubsync."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .srt_parser import parse_srt, write_srt
from .sync_engine import (
    cross_validate,
    sync_by_linear_shift,
    sync_by_sequence,
    sync_by_similarity,
    sync_to_transcript,
)


def _progress(current: int, total: int, message: str) -> None:
    print(f"  [{current}/{total}] {message}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ezsubsync",
        description="Subtitle syncing made easy — synchronise subtitle files across languages.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    # Input files
    parser.add_argument(
        "-r", "--reference",
        help="Reference subtitle file (language A, correctly timed).",
    )
    parser.add_argument(
        "-t", "--target",
        help="Target subtitle file (language B, needs synchronisation).",
    )
    parser.add_argument(
        "-v", "--video",
        help="Video file for audio extraction and OCR.",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path. Defaults to <target_stem>_synced.srt.",
    )

    # Sync options
    parser.add_argument(
        "-m", "--method",
        choices=["sequence", "similarity", "linear_shift"],
        default="sequence",
        help="Synchronisation method (default: sequence).",
    )
    parser.add_argument(
        "--whisper", action="store_true",
        help="Use Whisper speech recognition on the video audio.",
    )
    parser.add_argument(
        "--whisper-model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size (default: base).",
    )
    parser.add_argument(
        "--ocr", action="store_true",
        help="Use OCR to detect on-screen text in the video.",
    )
    parser.add_argument(
        "--cross-validate", action="store_true",
        help="Cross-validate synced timings against audio/OCR.",
    )

    # GUI
    parser.add_argument(
        "--gui", action="store_true",
        help="Launch the graphical user interface.",
    )

    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Launch GUI when explicitly requested or when no arguments are given
    if args.gui or (not args.target and not args.reference and not args.video):
        from .gui import run_gui
        run_gui()
        return 0

    # Validate inputs for CLI mode
    if not args.target:
        parser.error("--target is required when running in CLI mode.")

    if not args.reference and not args.video:
        parser.error("Provide at least a --reference subtitle or a --video file.")

    # Resolve output path
    target_path = Path(args.target)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = target_path.with_name(f"{target_path.stem}_synced.srt")

    # Parse target
    print(f"Parsing target: {args.target}")
    target = parse_srt(args.target)
    print(f"  → {len(target)} subtitles")

    transcript_segments = None
    ocr_detections = None

    # Video processing
    if args.video:
        if args.whisper:
            from .audio_extractor import extract_audio
            from .whisper_transcriber import transcribe

            print("Extracting audio…")
            audio_path = extract_audio(args.video, progress_cb=_progress)

            print(f"Transcribing with Whisper ({args.whisper_model})…")
            transcript_segments = transcribe(
                audio_path, model_name=args.whisper_model, progress_cb=_progress,
            )
            print(f"  → {len(transcript_segments)} segments")

        if args.ocr:
            from .ocr_detector import detect_onscreen_text

            print("Running OCR text detection…")
            ocr_detections = detect_onscreen_text(args.video, progress_cb=_progress)
            print(f"  → detected text in {len(ocr_detections)} frames")

    # Synchronise
    result = None

    if args.reference:
        print(f"Parsing reference: {args.reference}")
        reference = parse_srt(args.reference)
        print(f"  → {len(reference)} subtitles")

        method = args.method
        print(f"Synchronising with method: {method}")

        if method == "sequence":
            result = sync_by_sequence(reference, target, _progress)
        elif method == "similarity":
            result = sync_by_similarity(reference, target, progress_cb=_progress)
        elif method == "linear_shift":
            result = sync_by_linear_shift(reference, target, _progress)

    elif transcript_segments:
        print("Synchronising to Whisper transcript…")
        result = sync_to_transcript(transcript_segments, target, _progress)

    if result is None:
        print("ERROR: Could not determine a synchronisation strategy.", file=sys.stderr)
        return 1

    # Cross-validation
    if args.cross_validate and transcript_segments:
        print("Cross-validating…")
        confirmed, total, warnings = cross_validate(
            result.synced, transcript_segments, ocr_detections,
            progress_cb=_progress,
        )
        print(f"  → Verified {confirmed}/{total} subtitles")
        for w in warnings[:10]:
            print(f"  ⚠ {w}")

    # Write output
    write_srt(result.synced, output_path)
    print(f"\nSynchronised subtitles written to: {output_path}")
    print(f"Method: {result.method} | Stats: {result.stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
