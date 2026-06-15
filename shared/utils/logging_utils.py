"""
Component: Experiment Logger
Location: shared/utils/logging_utils.py

Handles:
- Structured JSONL logging (machine-readable)
- Formatted console output (human-readable)
- Log file (human-readable, persistent)

Ref: rules/CONVENTIONS.md Section 3
"""
import json
import logging
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional


# ══════════════════════════════════════════════════════════════
# Console color codes (ANSI)
# ══════════════════════════════════════════════════════════════
_C_RESET  = "\033[0m"
_C_BOLD   = "\033[1m"
_C_GREEN  = "\033[92m"
_C_YELLOW = "\033[93m"
_C_CYAN   = "\033[96m"
_C_RED    = "\033[91m"
_C_STAR   = "\033[93m"  # Gold for "best" events


def _now_str() -> str:
    """Return current time as string in UTC+7."""
    tz = timezone(timedelta(hours=7))
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")


# ══════════════════════════════════════════════════════════════
# ExperimentLogger
# ══════════════════════════════════════════════════════════════

class ExperimentLogger:
    """Unified logger for experiment runs.

    Writes to:
      1. Console (colored, human-readable)
      2. run_dir/logs/train.log (plain text, human-readable)
      3. run_dir/logs/train.jsonl (JSONL, machine-readable)

    Usage:
        logger = ExperimentLogger(run_dir=Path("experiments/runs/cdga__..."))
        logger.info("Training started")
        logger.log_epoch(epoch=1, phase="train", loss=0.42, lr=6e-4, time_s=142.3)
        logger.log_epoch(epoch=1, phase="val", miou=78.23, bf1_3=45.21, ...)
        logger.log_hook_stats(epoch=1, stats={"orig_boundary": 0.023, ...})
        logger.close()
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self._log_path = run_dir / "logs" / "train.log"
        self._jsonl_path = run_dir / "logs" / "train.jsonl"

        # Setup Python logger for file output
        self._file_logger = logging.getLogger(f"exp.{run_dir.name}")
        self._file_logger.setLevel(logging.DEBUG)
        self._file_logger.propagate = False

        fh = logging.FileHandler(self._log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        self._file_logger.addHandler(fh)

        # Open JSONL file
        self._jsonl_file = open(self._jsonl_path, "a", encoding="utf-8", buffering=1)

        self.info(f"Logger initialized. Run: {run_dir.name}")

    # ──────────────────────────────────────────────────────────
    # Core logging methods
    # ──────────────────────────────────────────────────────────

    def info(self, message: str) -> None:
        """Log a plain info message."""
        line = f"[{_now_str()}] {message}"
        print(line, file=sys.stdout, flush=True)
        self._file_logger.info(line)
        self._write_jsonl({"type": "info", "message": message, "ts": _now_str()})

    def warning(self, message: str) -> None:
        line = f"[{_now_str()}] {_C_YELLOW}[WARNING]{_C_RESET} {message}"
        print(line, file=sys.stdout, flush=True)
        self._file_logger.warning(line)
        self._write_jsonl({"type": "warning", "message": message, "ts": _now_str()})

    def error(self, message: str) -> None:
        line = f"[{_now_str()}] {_C_RED}[ERROR]{_C_RESET} {message}"
        print(line, file=sys.stderr, flush=True)
        self._file_logger.error(line)
        self._write_jsonl({"type": "error", "message": message, "ts": _now_str()})

    # ──────────────────────────────────────────────────────────
    # Epoch logging
    # ──────────────────────────────────────────────────────────

    def log_epoch(
        self,
        epoch: int,
        phase: str,
        total_epochs: Optional[int] = None,
        **metrics: Any,
    ) -> None:
        """Log epoch-level metrics.

        Args:
            epoch: Current epoch number (1-indexed).
            phase: One of 'train', 'val'.
            total_epochs: Total number of epochs (for progress display).
            **metrics: Arbitrary metric key=value pairs.
                Train phase expects: loss, lr, time_s
                Val phase expects: miou, mf1, oa, bf1_3, bf1_5, time_s
        """
        epoch_str = f"Epoch {epoch:>3d}"
        if total_epochs:
            epoch_str += f"/{total_epochs}"

        if phase == "train":
            loss = metrics.get("loss", float("nan"))
            lr = metrics.get("lr", float("nan"))
            time_s = metrics.get("time_s", 0.0)
            console_line = (
                f"[{_now_str()}] [{epoch_str}] "
                f"{_C_CYAN}TRAIN{_C_RESET} | "
                f"loss: {_C_BOLD}{loss:.4f}{_C_RESET} | "
                f"lr: {lr:.2e} | "
                f"time: {time_s:.1f}s"
            )

        elif phase == "val":
            miou   = metrics.get("miou", float("nan"))
            mf1    = metrics.get("mf1", float("nan"))
            oa     = metrics.get("oa", float("nan"))
            bf1_3  = metrics.get("bf1_3", float("nan"))
            time_s = metrics.get("time_s", 0.0)
            console_line = (
                f"[{_now_str()}] [{epoch_str}] "
                f"{_C_GREEN}VAL  {_C_RESET} | "
                f"mIoU: {_C_BOLD}{miou:5.2f}{_C_RESET} | "
                f"mF1: {mf1:5.2f} | "
                f"BF1@3: {bf1_3:5.2f} | "
                f"OA: {oa:5.2f} | "
                f"time: {time_s:.1f}s"
            )

        else:
            console_line = f"[{_now_str()}] [{epoch_str}] [{phase.upper()}] {metrics}"

        print(console_line, flush=True)
        # Strip ANSI codes for file log
        clean_line = _strip_ansi(console_line)
        self._file_logger.info(clean_line)

        # JSONL — always include epoch + phase
        record = {"epoch": epoch, "phase": phase, **metrics, "ts": _now_str()}
        self._write_jsonl(record)

    def log_hook_stats(self, epoch: int, stats: dict[str, float]) -> None:
        """Log gradient hook statistics (CDGA-specific).

        Expected stats keys (from CDGAHook.grad_stats):
            orig_boundary, mod_boundary, orig_interior, mod_interior
        """
        if not stats:
            return

        orig_b = stats.get("orig_boundary", float("nan"))
        mod_b  = stats.get("mod_boundary", float("nan"))
        snr_ratio = mod_b / (orig_b + 1e-12)

        console_line = (
            f"[{_now_str()}] [Epoch {epoch:>3d}] "
            f"{_C_YELLOW}HOOK {_C_RESET} | "
            f"∇_B(orig): {orig_b:.4f} | "
            f"∇_B(mod): {mod_b:.4f} | "
            f"SNR×: {snr_ratio:.2f}"
        )
        print(console_line, flush=True)
        self._file_logger.info(_strip_ansi(console_line))
        self._write_jsonl({
            "epoch": epoch, "phase": "hook",
            "snr_ratio": snr_ratio, **stats, "ts": _now_str()
        })

    def log_best_event(self, epoch: int, metric_name: str, value: float, prev: float) -> None:
        """Log when a new best metric is achieved."""
        console_line = (
            f"[{_now_str()}] [Epoch {epoch:>3d}] "
            f"{_C_STAR}★ New best {metric_name}: {value:.2f} "
            f"(prev: {prev:.2f}){_C_RESET}"
        )
        print(console_line, flush=True)
        self._file_logger.info(_strip_ansi(console_line))
        self._write_jsonl({
            "epoch": epoch, "phase": "best_event",
            "metric": metric_name, "value": value, "prev": prev,
            "ts": _now_str()
        })

    def log_final_metrics(self, **metrics: Any) -> None:
        """Log final summary metrics after training completes."""
        self.info("─" * 60)
        self.info("TRAINING COMPLETE — Final Summary")
        for k, v in metrics.items():
            if isinstance(v, float):
                self.info(f"  {k}: {v:.4f}")
            else:
                self.info(f"  {k}: {v}")
        self.info("─" * 60)
        self._write_jsonl({"phase": "final", **metrics, "ts": _now_str()})

    # ──────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────

    def _write_jsonl(self, record: dict) -> None:
        """Write one record to the JSONL file."""
        self._jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._jsonl_file.flush()

    def close(self) -> None:
        """Close file handles."""
        self._jsonl_file.close()
        for handler in self._file_logger.handlers:
            handler.close()


# ══════════════════════════════════════════════════════════════
# Utility: Strip ANSI codes (for writing to plain text file)
# ══════════════════════════════════════════════════════════════

import re
_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)
