"""Tests for audio_extractor module."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from ezsubsync.audio_extractor import check_ffmpeg, extract_audio


class TestCheckFfmpeg:
    def test_found(self):
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            assert check_ffmpeg() is True

    def test_not_found(self):
        with patch("shutil.which", return_value=None):
            assert check_ffmpeg() is False


class TestExtractAudio:
    def test_missing_video(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            extract_audio(tmp_path / "nonexistent.mp4")

    def test_no_ffmpeg(self, tmp_path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00")
        with patch("ezsubsync.audio_extractor.check_ffmpeg", return_value=False):
            with pytest.raises(FileNotFoundError, match="ffmpeg"):
                extract_audio(video)

    def test_success(self, tmp_path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00")
        expected_out = tmp_path / "test_audio.wav"

        mock_proc = MagicMock()
        mock_proc.stdout = iter([])
        mock_proc.stderr = iter([])
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0

        with patch("ezsubsync.audio_extractor.check_ffmpeg", return_value=True), \
             patch("ezsubsync.audio_extractor._get_video_duration_ms", return_value=None), \
             patch("subprocess.Popen", return_value=mock_proc):
            result = extract_audio(video)
            assert result == expected_out

    def test_progress_callback_reports_percentage(self, tmp_path):
        """Progress callback should receive percentage updates during extraction."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00")

        mock_proc = MagicMock()
        # Simulate ffmpeg progress output
        mock_proc.stdout = iter([
            "out_time_us=5000000\n",
            "progress=continue\n",
            "out_time_us=10000000\n",
            "progress=end\n",
        ])
        mock_proc.stderr = iter([])
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0

        calls = []
        def cb(c, t, m): calls.append((c, t, m))

        with patch("ezsubsync.audio_extractor.check_ffmpeg", return_value=True), \
             patch("ezsubsync.audio_extractor._get_video_duration_ms", return_value=20_000), \
             patch("subprocess.Popen", return_value=mock_proc):
            extract_audio(video, progress_cb=cb)

        # Should have: start message, 25% (5s/20s), 50% (10s/20s), complete
        messages = [m for _, _, m in calls]
        assert any("25%" in m for m in messages)
        assert any("50%" in m for m in messages)
        assert any("complete" in m.lower() for m in messages)
