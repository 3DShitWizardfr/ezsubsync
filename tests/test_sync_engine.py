"""Tests for sync_engine module."""

import textwrap

import pytest

from ezsubsync.srt_parser import Subtitle, SubtitleFile, parse_srt_string
from ezsubsync.sync_engine import (
    cross_validate,
    sync_by_linear_shift,
    sync_by_sequence,
    sync_by_similarity,
    sync_to_transcript,
)

# ---- Fixtures ----

REFERENCE_SRT = textwrap.dedent("""\
    1
    00:00:01,000 --> 00:00:04,000
    Hello

    2
    00:00:05,000 --> 00:00:08,000
    World

    3
    00:00:10,000 --> 00:00:13,000
    Goodbye
""")

TARGET_SRT = textwrap.dedent("""\
    1
    00:00:10,000 --> 00:00:13,000
    Bonjour

    2
    00:00:15,000 --> 00:00:18,000
    Monde

    3
    00:00:20,000 --> 00:00:23,000
    Au revoir
""")


@pytest.fixture
def reference():
    return parse_srt_string(REFERENCE_SRT)


@pytest.fixture
def target():
    return parse_srt_string(TARGET_SRT)


# ---- sync_by_sequence ----

class TestSyncBySequence:
    def test_basic_alignment(self, reference, target):
        result = sync_by_sequence(reference, target)
        assert result.method == "sequence"
        synced = result.synced
        assert len(synced) == 3
        # Timings should come from reference
        assert synced[0].start_ms == 1_000
        assert synced[0].end_ms == 4_000
        # Text should come from target
        assert synced[0].text == "Bonjour"

    def test_preserves_target_text(self, reference, target):
        result = sync_by_sequence(reference, target)
        texts = [s.text for s in result.synced]
        assert texts == ["Bonjour", "Monde", "Au revoir"]

    def test_empty_target(self, reference):
        empty = SubtitleFile(subtitles=[])
        result = sync_by_sequence(reference, empty)
        assert len(result.synced) == 0

    def test_more_target_than_ref(self, reference):
        extra = parse_srt_string(textwrap.dedent("""\
            1
            00:00:10,000 --> 00:00:13,000
            A

            2
            00:00:15,000 --> 00:00:18,000
            B

            3
            00:00:20,000 --> 00:00:23,000
            C

            4
            00:00:25,000 --> 00:00:28,000
            D
        """))
        result = sync_by_sequence(reference, extra)
        assert len(result.synced) == 4
        assert result.stats["matched"] == 3

    def test_progress_callback(self, reference, target):
        calls = []
        def cb(c, t, m): calls.append((c, t, m))
        sync_by_sequence(reference, target, progress_cb=cb)
        assert len(calls) == 3

    def test_uses_reference_timing(self):
        """Both start and end times should come from the reference subtitle."""
        ref = parse_srt_string(textwrap.dedent("""\
            1
            00:00:01,000 --> 00:00:03,000
            Hi

            2
            00:00:05,000 --> 00:00:07,000
            Bye
        """))
        tgt = parse_srt_string(textwrap.dedent("""\
            1
            00:00:10,000 --> 00:00:14,500
            Bonjour

            2
            00:00:20,000 --> 00:00:25,200
            Au revoir
        """))
        result = sync_by_sequence(ref, tgt)
        # Start times come from reference
        assert result.synced[0].start_ms == 1_000
        assert result.synced[1].start_ms == 5_000
        # End times come from reference
        assert result.synced[0].end_ms == 3_000
        assert result.synced[1].end_ms == 7_000
        # Durations match reference (2000ms each)
        assert result.synced[0].duration_ms == 2_000
        assert result.synced[1].duration_ms == 2_000

    def test_progress_messages_contain_timestamp(self):
        calls = []
        def cb(c, t, m): calls.append((c, t, m))
        ref = parse_srt_string("1\n00:00:01,000 --> 00:00:04,000\nHi\n")
        tgt = parse_srt_string("1\n00:00:10,000 --> 00:00:13,000\nHola\n")
        sync_by_sequence(ref, tgt, progress_cb=cb)
        assert len(calls) == 1
        assert "00:00:10,000" in calls[0][2]


# ---- sync_by_similarity ----

class TestSyncBySimilarity:
    def test_identical_text_match(self):
        ref = parse_srt_string(textwrap.dedent("""\
            1
            00:00:01,000 --> 00:00:04,000
            Hello world
        """))
        tgt = parse_srt_string(textwrap.dedent("""\
            1
            00:00:10,000 --> 00:00:13,000
            Hello world
        """))
        result = sync_by_similarity(ref, tgt, threshold=0.5)
        assert result.synced[0].start_ms == 1_000
        assert result.stats["matched"] == 1

    def test_no_match_below_threshold(self):
        ref = parse_srt_string(textwrap.dedent("""\
            1
            00:00:01,000 --> 00:00:04,000
            AAAA
        """))
        tgt = parse_srt_string(textwrap.dedent("""\
            1
            00:00:10,000 --> 00:00:13,000
            ZZZZ
        """))
        result = sync_by_similarity(ref, tgt, threshold=0.9)
        # No match: keeps original timing
        assert result.synced[0].start_ms == 10_000
        assert result.stats["matched"] == 0

    def test_empty_inputs(self):
        empty = SubtitleFile(subtitles=[])
        ref = parse_srt_string("1\n00:00:01,000 --> 00:00:02,000\nHi\n")
        result = sync_by_similarity(ref, empty)
        assert len(result.synced) == 0

    def test_uses_reference_timing(self):
        """Both start and end times should come from the reference."""
        ref = parse_srt_string(textwrap.dedent("""\
            1
            00:00:01,000 --> 00:00:03,000
            Hello world
        """))
        tgt = parse_srt_string(textwrap.dedent("""\
            1
            00:00:10,000 --> 00:00:15,500
            Hello world
        """))
        result = sync_by_similarity(ref, tgt, threshold=0.5)
        assert result.synced[0].start_ms == 1_000
        assert result.synced[0].end_ms == 3_000
        assert result.synced[0].duration_ms == 2_000


# ---- sync_by_linear_shift ----

class TestSyncByLinearShift:
    def test_simple_shift(self, reference, target):
        result = sync_by_linear_shift(reference, target)
        assert result.method == "linear_shift"
        # First subtitle should map to ~reference start
        assert abs(result.synced[0].start_ms - reference[0].start_ms) < 100

    def test_single_subtitle(self):
        ref = parse_srt_string("1\n00:00:05,000 --> 00:00:08,000\nA\n")
        tgt = parse_srt_string("1\n00:00:00,000 --> 00:00:03,000\nB\n")
        result = sync_by_linear_shift(ref, tgt)
        assert result.synced[0].start_ms == 5_000

    def test_empty_target(self, reference):
        empty = SubtitleFile(subtitles=[])
        result = sync_by_linear_shift(reference, empty)
        assert len(result.synced) == 0


# ---- sync_to_transcript ----

class TestSyncToTranscript:
    def test_basic_transcript_sync(self):
        segments = [
            {"start": 1.0, "end": 4.0, "text": "Hello world"},
            {"start": 5.0, "end": 8.0, "text": "How are you"},
        ]
        tgt = parse_srt_string(textwrap.dedent("""\
            1
            00:00:10,000 --> 00:00:13,000
            Hello world

            2
            00:00:15,000 --> 00:00:18,000
            How are you
        """))
        result = sync_to_transcript(segments, tgt)
        assert result.synced[0].start_ms == 1_000
        assert result.synced[1].start_ms == 5_000
        assert result.stats["matched"] == 2

    def test_empty_transcript(self, target):
        result = sync_to_transcript([], target)
        assert len(result.synced) == 3
        assert result.stats["matched"] == 0

    def test_uses_transcript_timing(self):
        """Both start and end times should come from the transcript segment."""
        segments = [
            {"start": 1.0, "end": 4.0, "text": "Hello world"},
        ]
        tgt = parse_srt_string(textwrap.dedent("""\
            1
            00:00:10,000 --> 00:00:16,200
            Hello world
        """))
        result = sync_to_transcript(segments, tgt)
        assert result.synced[0].start_ms == 1_000
        # End time should come from transcript segment (4.0s = 4000ms)
        assert result.synced[0].end_ms == 4_000
        assert result.synced[0].duration_ms == 3_000


# ---- cross_validate ----

class TestCrossValidate:
    def test_all_confirmed(self):
        synced = parse_srt_string(textwrap.dedent("""\
            1
            00:00:01,000 --> 00:00:04,000
            Hi
        """))
        segments = [{"start": 1.0, "end": 4.0, "text": "Hi"}]
        confirmed, total, warnings = cross_validate(synced, segments)
        assert confirmed == 1
        assert total == 1
        assert warnings == []

    def test_warning_on_mismatch(self):
        synced = parse_srt_string(textwrap.dedent("""\
            1
            00:00:30,000 --> 00:00:33,000
            Late text
        """))
        segments = [{"start": 1.0, "end": 4.0, "text": "Early text"}]
        confirmed, total, warnings = cross_validate(synced, segments, tolerance_ms=1000)
        assert confirmed == 0
        assert len(warnings) == 1

    def test_ocr_confirmation(self):
        synced = parse_srt_string(textwrap.dedent("""\
            1
            00:00:05,000 --> 00:00:08,000
            Title
        """))
        segments = [{"start": 50.0, "end": 53.0, "text": "Far away"}]
        ocr = [{"time_ms": 5_500, "text": "Title text"}]
        confirmed, total, warnings = cross_validate(
            synced, segments, ocr_detections=ocr, tolerance_ms=1000,
        )
        assert confirmed == 1
