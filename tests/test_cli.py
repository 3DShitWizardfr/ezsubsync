"""Tests for the CLI entry point."""

import sys
from unittest.mock import MagicMock, patch

from ezsubsync.__main__ import build_parser, main


class TestBuildParser:
    def test_no_args_parses_ok(self):
        parser = build_parser()
        # No args should parse without error (GUI will be launched)
        args = parser.parse_args([])
        assert args.target is None
        assert args.gui is False

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
        args = parser.parse_args(["--gui"])
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


class TestMainGuiLaunch:
    def test_no_args_launches_gui(self):
        """Running with no arguments should launch the GUI."""
        mock_gui = MagicMock()
        with patch.dict(sys.modules, {"ezsubsync.gui": mock_gui}):
            assert main([]) == 0
            mock_gui.run_gui.assert_called_once()

    def test_gui_flag_launches_gui(self):
        """Running with --gui should launch the GUI."""
        mock_gui = MagicMock()
        with patch.dict(sys.modules, {"ezsubsync.gui": mock_gui}):
            assert main(["--gui"]) == 0
            mock_gui.run_gui.assert_called_once()

    def test_target_without_ref_or_video_errors(self):
        """CLI mode with --target but no --reference/--video should fail."""
        import pytest
        with pytest.raises(SystemExit):
            main(["-t", "subs.srt"])
