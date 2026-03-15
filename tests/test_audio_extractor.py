"""Tests for audio_extractor module."""

from unittest.mock import patch

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

        with patch("ezsubsync.audio_extractor.check_ffmpeg", return_value=True), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            result = extract_audio(video)
            assert result == expected_out
            mock_run.assert_called_once()
