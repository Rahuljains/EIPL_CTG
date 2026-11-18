"""Shared plumbing: run names, output folders, logging, ETA, seeding, run banners.

Author: Rahul Jain
Date:   17 Nov 2026
Usage:  from ctg.utils import setup_run, saved, print_eta
"""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from . import config as C

LOGGER_NAME = "ctg"


# ---------------------------------------------------------------- run plumbing
def make_run_name() -> str:
    """Return a timestamp run name, e.g. 20261117_1532."""
    return datetime.now().strftime("%Y%m%d_%H%M")


def setup_output_dirs(output_dir: Path = C.OUTPUT_DIR) -> Dict[str, Path]:
    """Create output/{figures,sheets,...} and return them as a name -> Path dict."""
    dirs = {}
    for name in C.OUTPUT_SUBDIRS:
        path = output_dir / name
        path.mkdir(parents=True, exist_ok=True)
        dirs[name] = path
    return dirs


def set_seed(seed: int = C.SEED) -> None:
    """Seed python, numpy and (if importable) torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(script_name: str, run_name: str,
               log_dir: Optional[Path] = None) -> logging.Logger:
    """Return a logger writing to the console and to logs/run_<script>_<run>.log."""
    log_dir = log_dir or (C.OUTPUT_DIR / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{LOGGER_NAME}.{script_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_dir / f"run_{script_name}_{run_name}.log",
                                       encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


class RunContext:
    """Holds the run name, logger, output dirs and the list of files written."""

    def __init__(self, script_name: str, run_name: str, logger: logging.Logger,
                 dirs: Dict[str, Path]) -> None:
        self.script_name = script_name
        self.run_name = run_name
        self.log = logger
        self.dirs = dirs
        self.started = time.time()
        self.files: List[Path] = []

    def saved(self, path: Path, note: str = "") -> Path:
        """Record a written file and log it."""
        path = Path(path)
        self.files.append(path)
        suffix = f"  ({note})" if note else ""
        self.log.info("Saved -> %s%s", path, suffix)
        return path

    def end_banner(self) -> None:
        """Print the end banner: end time, duration and every file written."""
        dur = time.time() - self.started
        self.log.info("=" * 78)
        self.log.info("Finished %s at %s | duration %s",
                      self.script_name, datetime.now().strftime("%H:%M:%S"),
                      format_duration(dur))
        self.log.info("Files written (%d):", len(self.files))
        for path in self.files:
            self.log.info("   %s", path)
        self.log.info("=" * 78)


def setup_run(script_name: str, config_values: Optional[Dict[str, Any]] = None,
              run_name: Optional[str] = None) -> RunContext:
    """Create dirs, seed, logger and print the start banner. Return a RunContext."""
    run_name = run_name or make_run_name()
    dirs = setup_output_dirs()
    set_seed(C.SEED)
    logger = get_logger(script_name, run_name)
    ctx = RunContext(script_name, run_name, logger, dirs)

    logger.info("=" * 78)
    logger.info("%s | RUN_NAME = %s", script_name, run_name)
    logger.info("Started at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Python %s on %s | device: %s",
                platform.python_version(), platform.system(), describe_device())
    logger.info("Root: %s | seed %d", C.ROOT, C.SEED)
    for key, value in (config_values or {}).items():
        logger.info("  %-24s = %s", key, value)
    logger.info("=" * 78)
    return ctx


def describe_device() -> str:
    """Return a one-line description of the compute device."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            return f"cuda ({name}, {total:.1f} GB)"
        return "cpu (torch present, no cuda)"
    except ImportError:
        return "cpu (torch not installed)"


# ---------------------------------------------------------------- progress/ETA
def format_duration(seconds: float) -> str:
    """Format a duration as e.g. '1h 04m 12s'."""
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {mins:02d}m {secs:02d}s"
    if mins:
        return f"{mins}m {secs:02d}s"
    return f"{secs}s"


def print_eta(logger: logging.Logger, done: int, total: int, started: float) -> None:
    """Log the estimated finish as a clock time, e.g. 'Estimated finish: 18:42'."""
    if done <= 0:
        return
    elapsed = time.time() - started
    remaining = elapsed / done * (total - done)
    finish = datetime.now() + timedelta(seconds=remaining)
    logger.info("Estimated finish: %s (in ~%s)",
                finish.strftime("%H:%M"), format_duration(remaining))


# ---------------------------------------------------------------- file helpers
def latest_file(folder: Path, pattern: str,
                logger: Optional[logging.Logger] = None,
                exclude: Iterable[str] = ()) -> Path:
    """Return the newest file matching pattern in folder; fail clearly if none.

    `exclude` drops any match whose name contains one of the given substrings, so
    `ctu_clinical_*` does not pick up `ctu_clinical_clean_*`.
    """
    exclude = tuple(exclude)
    matches = [p for p in folder.glob(pattern)
               if not any(token in p.name for token in exclude)]
    matches.sort(key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} in {folder}. "
                                "Run the upstream script first.")
    chosen = matches[-1]
    if logger is not None:
        logger.info("Using input: %s", chosen)
    return chosen


def resolve_input(folder: Path, stem: str, run: Optional[str], suffix: str,
                  logger: Optional[logging.Logger] = None,
                  exclude: Iterable[str] = ()) -> Path:
    """Resolve <stem>_<run><suffix>, or the latest <stem>_*<suffix> when run is None."""
    if run:
        path = folder / f"{stem}_{run}{suffix}"
        if not path.exists():
            raise FileNotFoundError(f"Expected input not found: {path}")
        if logger is not None:
            logger.info("Using input: %s", path)
        return path
    return latest_file(folder, f"{stem}_*{suffix}", logger, exclude=exclude)


def run_name_from_path(path: Path, stem: str) -> str:
    """Extract the run name out of <stem>_<run><suffix>."""
    return path.stem[len(stem) + 1:]


def library_versions(packages: Iterable[str] = ()) -> Dict[str, str]:
    """Return {package: version} for the given packages plus python itself."""
    versions = {"python": platform.python_version()}
    for name in packages:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[name] = "not installed"
    return versions


def save_config_snapshot(ctx: RunContext, extra: Optional[Dict[str, Any]] = None,
                         packages: Iterable[str] = ("numpy", "pandas", "scipy",
                                                    "sklearn", "matplotlib")) -> Path:
    """Write logs/config_<RUN>.json with the config module, versions and extras."""
    snapshot = {
        "run_name": ctx.run_name,
        "script": ctx.script_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": C.SEED,
        "versions": library_versions(packages),
        "config": {k: _jsonable(v) for k, v in vars(C).items()
                   if k.isupper() and not k.startswith("_")},
    }
    if extra:
        snapshot["extra"] = {k: _jsonable(v) for k, v in extra.items()}
    path = ctx.dirs["logs"] / f"config_{ctx.run_name}.json"
    path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return ctx.saved(path)


def _jsonable(value: Any) -> Any:
    """Convert Paths, tuples and numpy scalars so json.dumps accepts them."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


# ---------------------------------------------------------------- decisions log
def log_decision(topic: str, text: str, run_name: str = "",
                 reports_dir: Optional[Path] = None) -> Path:
    """Append one judgment call to output/reports/decisions_log.md."""
    reports_dir = reports_dir or (C.OUTPUT_DIR / "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "decisions_log.md"
    if not path.exists():
        path.write_text("# Decisions log\n\n"
                        "Every judgment call made while building the CTG Risk Monitor.\n"
                        "Aggregates only; no participant-level values.\n\n",
                        encoding="utf-8")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    tag = f" (run {run_name})" if run_name else ""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"- **{stamp}{tag} | {topic}** - {text}\n")
    return path
