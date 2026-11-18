"""Loading the CTU-UHB intrapartum cardiotocography database.

Finds the .hea headers wherever the zip extracted them, parses the clinical and
outcome comment lines robustly, and reads the 4 Hz FHR/UC waveforms via wfdb.

`dbID` is dropped the moment a header is parsed and is never returned, written
or logged (organisation privacy rule).

Author: Rahul Jain
Date:   17 Nov 2026
Usage:  from ctg.io_ctu import find_ctu_headers, parse_header_comments, load_record
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import wfdb

from . import config as C


@dataclass
class CtuRecord:
    """One CTU-UHB labour: raw 4 Hz signals plus its header fields."""

    record_id: int
    fhr_4hz: np.ndarray          # bpm, NaN where the header coded 0 (missing)
    uc_4hz: np.ndarray           # UC units, NaN where exactly 0
    fields: Dict[str, float]     # parsed header comments, dbID already removed
    sig_names: List[str]
    fs: float
    n_samples: int

    @property
    def duration_min(self) -> float:
        """Recording length in minutes."""
        return self.n_samples / (self.fs * 60.0)


def find_ctu_headers(ctu_dir: Path = C.CTU_DIR) -> List[Path]:
    """Return every .hea under ctu_dir, sorted by numeric record id.

    The PhysioNet zip may add a nested folder, so the search is recursive and the
    exact layout is never assumed.
    """
    if not ctu_dir.exists():
        raise FileNotFoundError(f"CTU-UHB folder not found: {ctu_dir}")
    headers = [p for p in ctu_dir.rglob("*.hea") if p.stem.isdigit()]
    if not headers:
        raise FileNotFoundError(f"No numeric *.hea files under {ctu_dir}")
    return sorted(headers, key=lambda p: int(p.stem))


def parse_header_comments(header_path: Path) -> Dict[str, float]:
    """Parse the '#key   value' comment lines of one .hea into {key: float}.

    Key is everything between '#' and the last whitespace-separated token; the
    value is that last token as a float, or NaN when it cannot be parsed.
    Section lines starting with '#--' are skipped, and dbID is dropped.
    """
    fields: Dict[str, float] = {}
    for raw in header_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.rstrip()
        if not line.startswith("#") or line.startswith("#--"):
            continue
        body = line[1:].strip()
        if not body:
            continue
        tokens = body.split()
        if len(tokens) < 2:
            continue                       # key with no value: nothing to record
        key = " ".join(tokens[:-1]).strip()
        if key in C.HEADER_KEY_DROP:       # privacy: dbID never leaves this function
            continue
        try:
            value = float(tokens[-1])
        except ValueError:
            value = float("nan")
        fields[key] = value
    return fields


def load_record(header_path: Path, drop_zero_uc: bool = True) -> CtuRecord:
    """Read one CTU-UHB record: 4 Hz FHR/UC with 0 mapped to NaN, plus its fields.

    FHR is stored with 0 meaning missing; UC zeros are also treated as missing
    (plan M1 step 3). Channel order is read from sig_name, never assumed.
    """
    record = wfdb.rdrecord(str(header_path.with_suffix("")))
    sig_names = [str(name).strip().upper() for name in record.sig_name]
    if "FHR" not in sig_names:
        raise ValueError(f"{header_path.name}: no FHR channel in {record.sig_name}")

    signal = np.asarray(record.p_signal, dtype=np.float64)
    fhr = signal[:, sig_names.index("FHR")].copy()
    if "UC" in sig_names:
        uc = signal[:, sig_names.index("UC")].copy()
    else:
        uc = np.full_like(fhr, np.nan)

    fhr[fhr == 0] = np.nan                 # dataset convention: 0 = missing
    if drop_zero_uc:
        uc[uc == 0] = np.nan

    return CtuRecord(
        record_id=int(header_path.stem),
        fhr_4hz=fhr,
        uc_4hz=uc,
        fields=parse_header_comments(header_path),
        sig_names=sig_names,
        fs=float(record.fs),
        n_samples=int(record.sig_len),
    )


def second_stage_index_1hz(fields: Dict[str, float], n_samples_1hz: int) -> int:
    """Convert 'Pos. II.st.' (4 Hz sample index) to a 1 Hz index, or -1 if absent.

    Values that are missing, non-positive, or beyond the recording mean there is
    no second stage inside the recording.
    """
    raw: Optional[float] = fields.get("Pos. II.st.")
    if raw is None or not np.isfinite(raw) or raw <= 0:
        return -1
    index = int(raw // C.FS_RAW)
    if index >= n_samples_1hz:
        return -1
    return index
