"""Graphical user interface for ezsubsync using tkinter."""

from __future__ import annotations

import logging
import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from . import __version__
from .srt_parser import SubtitleFile, parse_srt, write_srt
from .sync_engine import (
    SyncResult,
    cross_validate,
    sync_by_linear_shift,
    sync_by_sequence,
    sync_by_similarity,
    sync_to_transcript,
)

logger = logging.getLogger(__name__)

SYNC_METHODS = {
    "Sequence order": "sequence",
    "Content similarity": "similarity",
    "Linear time shift": "linear_shift",
}

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]


class EzSubSyncApp:
    """Main application window."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"ezsubsync v{__version__}")
        self.root.minsize(620, 520)

        # State variables
        self.reference_path = tk.StringVar()
        self.target_path = tk.StringVar()
        self.video_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.sync_method = tk.StringVar(value="Sequence order")
        self.whisper_model = tk.StringVar(value="base")
        self.use_whisper = tk.BooleanVar(value=False)
        self.use_ocr = tk.BooleanVar(value=False)
        self.cross_validate_var = tk.BooleanVar(value=False)

        self._build_ui()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 4}

        # --- File selection frame ---
        file_frame = ttk.LabelFrame(self.root, text="Input Files", padding=8)
        file_frame.pack(fill="x", **pad)

        self._file_row(file_frame, "Reference subtitle (lang A):", self.reference_path, 0, [("SRT files", "*.srt")])
        self._file_row(file_frame, "Target subtitle (lang B):", self.target_path, 1, [("SRT files", "*.srt")])
        self._file_row(file_frame, "Video file (optional):", self.video_path, 2, [("Video files", "*.mp4 *.mkv *.avi *.mov *.webm")])
        self._file_row(file_frame, "Output file:", self.output_path, 3, [("SRT files", "*.srt")], save=True)

        # --- Options frame ---
        opt_frame = ttk.LabelFrame(self.root, text="Sync Options", padding=8)
        opt_frame.pack(fill="x", **pad)

        ttk.Label(opt_frame, text="Sync method:").grid(row=0, column=0, sticky="w")
        method_combo = ttk.Combobox(
            opt_frame, textvariable=self.sync_method,
            values=list(SYNC_METHODS.keys()), state="readonly", width=25,
        )
        method_combo.grid(row=0, column=1, sticky="w", padx=4)

        ttk.Label(opt_frame, text="Whisper model:").grid(row=1, column=0, sticky="w")
        model_combo = ttk.Combobox(
            opt_frame, textvariable=self.whisper_model,
            values=WHISPER_MODELS, state="readonly", width=25,
        )
        model_combo.grid(row=1, column=1, sticky="w", padx=4)

        ttk.Checkbutton(opt_frame, text="Use Whisper speech recognition", variable=self.use_whisper).grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(opt_frame, text="Use OCR text detection", variable=self.use_ocr).grid(row=3, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(opt_frame, text="Cross-validate results", variable=self.cross_validate_var).grid(row=4, column=0, columnspan=2, sticky="w")

        # --- Progress frame ---
        prog_frame = ttk.LabelFrame(self.root, text="Progress", padding=8)
        prog_frame.pack(fill="both", expand=True, **pad)

        self.progress_bar = ttk.Progressbar(prog_frame, mode="determinate", maximum=100)
        self.progress_bar.pack(fill="x", pady=(0, 4))

        self.status_label = ttk.Label(prog_frame, text="Ready")
        self.status_label.pack(anchor="w")

        self.log_text = tk.Text(prog_frame, height=8, state="disabled", wrap="word")
        self.log_text.pack(fill="both", expand=True, pady=(4, 0))

        # --- Action buttons ---
        btn_frame = ttk.Frame(self.root, padding=8)
        btn_frame.pack(fill="x")

        self.sync_button = ttk.Button(btn_frame, text="Sync Subtitles", command=self._on_sync)
        self.sync_button.pack(side="right", padx=4)

        ttk.Button(btn_frame, text="Quit", command=self.root.destroy).pack(side="right")

    def _file_row(self, parent, label: str, var: tk.StringVar, row: int, filetypes, save: bool = False) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=var, width=45).grid(row=row, column=1, sticky="ew", padx=4)
        parent.columnconfigure(1, weight=1)

        if save:
            cmd = lambda: var.set(filedialog.asksaveasfilename(filetypes=filetypes, defaultextension=".srt"))
        else:
            cmd = lambda: var.set(filedialog.askopenfilename(filetypes=filetypes))
        ttk.Button(parent, text="Browse…", command=cmd).grid(row=row, column=2)

    # ------------------------------------------------------------------ logging

    def _log(self, msg: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_progress(self, current: int, total: int, message: str) -> None:
        if total > 0:
            pct = int(100 * current / total)
            self.progress_bar["value"] = pct
        self.status_label.configure(text=message)
        self._log(message)
        self.root.update_idletasks()

    def _progress_callback(self, current: int, total: int, message: str) -> None:
        self.root.after(0, self._set_progress, current, total, message)

    # ------------------------------------------------------------------ sync

    def _on_sync(self) -> None:
        target = self.target_path.get().strip()
        if not target:
            messagebox.showerror("Error", "Please select a target subtitle file.")
            return

        ref = self.reference_path.get().strip()
        video = self.video_path.get().strip()

        if not ref and not video:
            messagebox.showerror(
                "Error",
                "Please provide either a reference subtitle file or a video file.",
            )
            return

        output = self.output_path.get().strip()
        if not output:
            p = Path(target)
            output = str(p.with_name(f"{p.stem}_synced.srt"))
            self.output_path.set(output)

        self.sync_button.configure(state="disabled")
        self.progress_bar["value"] = 0
        self._log("--- Starting synchronisation ---")

        thread = threading.Thread(target=self._run_sync, daemon=True)
        thread.start()

    def _run_sync(self) -> None:
        try:
            self._do_sync()
        except Exception as exc:
            self.root.after(0, messagebox.showerror, "Error", str(exc))
            self.root.after(0, self._log, f"ERROR: {exc}")
            logger.exception("Sync failed")
        finally:
            self.root.after(0, self.sync_button.configure, {"state": "normal"})

    def _do_sync(self) -> None:
        target_path = self.target_path.get().strip()
        ref_path = self.reference_path.get().strip()
        video_path = self.video_path.get().strip()
        output_path = self.output_path.get().strip()

        # Parse target
        self._progress_callback(5, 100, "Parsing target subtitles…")
        target = parse_srt(target_path)
        self._progress_callback(10, 100, f"Parsed {len(target)} target subtitles")

        transcript_segments = None
        ocr_detections = None

        # --- Video-based processing ---
        if video_path:
            if self.use_whisper.get():
                self._progress_callback(15, 100, "Extracting audio…")
                from .audio_extractor import extract_audio
                audio_path = extract_audio(
                    video_path, progress_cb=self._progress_callback,
                )
                self._progress_callback(30, 100, "Running Whisper transcription…")
                from .whisper_transcriber import transcribe
                transcript_segments = transcribe(
                    audio_path,
                    model_name=self.whisper_model.get(),
                    progress_cb=self._progress_callback,
                )
                self._progress_callback(60, 100, f"Transcribed {len(transcript_segments)} segments")

            if self.use_ocr.get():
                self._progress_callback(65, 100, "Running OCR detection…")
                from .ocr_detector import detect_onscreen_text
                ocr_detections = detect_onscreen_text(
                    video_path, progress_cb=self._progress_callback,
                )
                self._progress_callback(75, 100, f"OCR found text in {len(ocr_detections)} frames")

        # --- Synchronisation ---
        result: Optional[SyncResult] = None

        if ref_path:
            self._progress_callback(80, 100, "Parsing reference subtitles…")
            reference = parse_srt(ref_path)
            self._progress_callback(82, 100, f"Parsed {len(reference)} reference subtitles")

            method_key = SYNC_METHODS.get(self.sync_method.get(), "sequence")
            self._progress_callback(85, 100, f"Syncing using method: {method_key}")

            if method_key == "sequence":
                result = sync_by_sequence(reference, target, self._progress_callback)
            elif method_key == "similarity":
                result = sync_by_similarity(reference, target, progress_cb=self._progress_callback)
            elif method_key == "linear_shift":
                result = sync_by_linear_shift(reference, target, self._progress_callback)

        elif transcript_segments:
            self._progress_callback(80, 100, "Syncing to Whisper transcript…")
            result = sync_to_transcript(
                transcript_segments, target, self._progress_callback,
            )

        if result is None:
            raise RuntimeError("Could not determine a synchronisation strategy.")

        # --- Cross-validation ---
        if self.cross_validate_var.get() and transcript_segments:
            self._progress_callback(90, 100, "Cross-validating…")
            confirmed, total, warnings = cross_validate(
                result.synced, transcript_segments, ocr_detections,
                progress_cb=self._progress_callback,
            )
            self._progress_callback(95, 100, f"Verified {confirmed}/{total} subtitles")
            for w in warnings[:10]:
                self._progress_callback(95, 100, f"  ⚠ {w}")

        # --- Write output ---
        self._progress_callback(98, 100, f"Writing output to {output_path}")
        write_srt(result.synced, output_path)

        stats = result.stats
        self._progress_callback(
            100, 100,
            f"Done! Method: {result.method}, "
            f"matched {stats.get('matched', '?')}/{stats.get('total', '?')}",
        )
        self.root.after(
            0, messagebox.showinfo, "Success",
            f"Synchronised subtitles saved to:\n{output_path}",
        )


def run_gui() -> None:
    """Launch the ezsubsync GUI."""
    root = tk.Tk()
    EzSubSyncApp(root)
    root.mainloop()
