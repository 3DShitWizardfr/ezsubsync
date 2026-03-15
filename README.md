# ezsubsync

Subtitle syncing made easy — synchronise subtitle files across languages.

## Features

- **Cross-language subtitle sync** — align a target subtitle file (language B)
  to the timings of a reference subtitle file (language A).
- **Multiple sync methods** — sequence order, content similarity, or linear
  time shift.
- **Video-based processing** (optional) — extract audio with ffmpeg, run
  local Whisper speech recognition for precise timings, and use OCR to
  detect on-screen text (titles, locations, character names).
- **Cross-validation** — verify synced timings against the audio transcript
  and OCR detections.
- **Graphical interface** — simple tkinter GUI for file selection, option
  configuration, and real-time progress.
- **Command-line interface** — scriptable CLI for batch workflows.

## Installation

```bash
pip install -r requirements.txt
```

Optional system dependencies:

| Dependency | Required for |
|------------|-------------|
| [ffmpeg](https://ffmpeg.org/download.html) | Audio extraction from video |
| [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) | On-screen text detection |

## Usage

### GUI

Launch the GUI by simply running ezsubsync with no arguments:

```bash
python -m ezsubsync
```

You can also explicitly request the GUI with the `--gui` flag:

```bash
python -m ezsubsync --gui
```

### CLI examples

Sync using a reference subtitle file (sequence order):

```bash
python -m ezsubsync -r reference_en.srt -t target_fr.srt -o synced_fr.srt
```

Sync using content similarity:

```bash
python -m ezsubsync -r ref.srt -t target.srt -m similarity
```

Sync using linear time shift:

```bash
python -m ezsubsync -r ref.srt -t target.srt -m linear_shift
```

Sync with video, Whisper transcription, and OCR:

```bash
python -m ezsubsync -t target.srt -v movie.mp4 --whisper --ocr --cross-validate
```

Use a reference file with video-based cross-validation:

```bash
python -m ezsubsync -r ref.srt -t target.srt -v movie.mp4 --whisper --ocr --cross-validate
```

### Full CLI options

```
python -m ezsubsync --help
```

## Running tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## License

This project is licensed under the GNU General Public License v3.0 — see the
[LICENSE](LICENSE) file for details.
