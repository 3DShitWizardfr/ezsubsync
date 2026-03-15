"""Tests for the CLI entry point."""

from ezsubsync.__main__ import build_parser


class TestBuildParser:
    def test_required_target(self):
        parser = build_parser()
        # --target is required
        import pytest
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["-t", "subs.srt", "-r", "ref.srt"])
        assert args.target == "subs.srt"
        assert args.reference == "ref.srt"
        assert args.method == "sequence"
        assert args.whisper_model == "base"
        assert args.gui is False

    def test_gui_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-t", "x.srt", "--gui"])
        assert args.gui is True

    def test_all_options(self):
        parser = build_parser()
        args = parser.parse_args([
            "-t", "target.srt",
            "-r", "ref.srt",
            "-v", "movie.mp4",
            "-o", "out.srt",
            "-m", "similarity",
            "--whisper",
            "--whisper-model", "small",
            "--ocr",
            "--cross-validate",
            "--verbose",
        ])
        assert args.target == "target.srt"
        assert args.reference == "ref.srt"
        assert args.video == "movie.mp4"
        assert args.output == "out.srt"
        assert args.method == "similarity"
        assert args.whisper is True
        assert args.whisper_model == "small"
        assert args.ocr is True
        assert args.cross_validate is True
        assert args.verbose is True
