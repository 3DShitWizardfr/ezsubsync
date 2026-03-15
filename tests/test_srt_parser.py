"""Tests for srt_parser module."""

import textwrap
from pathlib import Path

import pytest

from ezsubsync.srt_parser import (
    SubtitleFile,
    format_srt,
    ms_to_timestamp,
    parse_srt,
    parse_srt_string,
    timestamp_to_ms,
    write_srt,
)


# ---------------------------------------------------------------- timestamp

class TestTimestampToMs:
    def test_zero(self):
        assert timestamp_to_ms("00:00:00,000") == 0

    def test_simple(self):
        assert timestamp_to_ms("00:01:30,500") == 90_500

    def test_hours(self):
        assert timestamp_to_ms("02:00:00,000") == 7_200_000

    def test_full(self):
        assert timestamp_to_ms("01:23:45,678") == (
            1 * 3_600_000 + 23 * 60_000 + 45 * 1_000 + 678
        )

    def test_dot_separator(self):
        assert timestamp_to_ms("00:00:01.500") == 1_500

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            timestamp_to_ms("invalid")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            timestamp_to_ms("")


class TestMsToTimestamp:
    def test_zero(self):
        assert ms_to_timestamp(0) == "00:00:00,000"

    def test_simple(self):
        assert ms_to_timestamp(90_500) == "00:01:30,500"

    def test_hours(self):
        assert ms_to_timestamp(7_200_000) == "02:00:00,000"

    def test_negative_clamped(self):
        assert ms_to_timestamp(-100) == "00:00:00,000"

    def test_roundtrip(self):
        ts = "01:23:45,678"
        assert ms_to_timestamp(timestamp_to_ms(ts)) == ts


# ---------------------------------------------------------------- parsing

SAMPLE_SRT = textwrap.dedent("""\
    1
    00:00:01,000 --> 00:00:04,000
    Hello, world!

    2
    00:00:05,000 --> 00:00:08,000
    This is a test.

    3
    00:00:09,500 --> 00:00:12,000
    Third subtitle line.
""")


class TestParseSrtString:
    def test_parse_count(self):
        sf = parse_srt_string(SAMPLE_SRT)
        assert len(sf) == 3

    def test_parse_first(self):
        sf = parse_srt_string(SAMPLE_SRT)
        assert sf[0].text == "Hello, world!"
        assert sf[0].start_ms == 1_000
        assert sf[0].end_ms == 4_000

    def test_parse_indices(self):
        sf = parse_srt_string(SAMPLE_SRT)
        assert [s.index for s in sf] == [1, 2, 3]

    def test_parse_empty(self):
        sf = parse_srt_string("")
        assert len(sf) == 0

    def test_parse_no_text_block(self):
        sf = parse_srt_string("just some random text\nwith no timecodes")
        assert len(sf) == 0

    def test_multiline_text(self):
        content = textwrap.dedent("""\
            1
            00:00:01,000 --> 00:00:04,000
            Line one
            Line two
        """)
        sf = parse_srt_string(content)
        assert len(sf) == 1
        assert sf[0].text == "Line one\nLine two"


class TestParseSrtFile:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_srt("/nonexistent/path.srt")

    def test_read_write_roundtrip(self, tmp_path: Path):
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(SAMPLE_SRT, encoding="utf-8")

        sf = parse_srt(srt_file)
        assert len(sf) == 3

        out_file = tmp_path / "output.srt"
        write_srt(sf, out_file)

        sf2 = parse_srt(out_file)
        assert len(sf2) == 3
        assert sf2[0].text == sf[0].text
        assert sf2[0].start_ms == sf[0].start_ms


class TestFormatSrt:
    def test_format_roundtrip(self):
        sf = parse_srt_string(SAMPLE_SRT)
        formatted = format_srt(sf)
        sf2 = parse_srt_string(formatted)
        assert len(sf2) == len(sf)
        for a, b in zip(sf, sf2):
            assert a.text == b.text
            assert a.start_ms == b.start_ms
            assert a.end_ms == b.end_ms

    def test_format_empty(self):
        sf = SubtitleFile(subtitles=[])
        assert format_srt(sf) == ""
