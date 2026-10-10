"""Reproducibility record written next to every results folder."""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy
import pandas
import scipy
import sklearn


def write_runlog(out_dir: Path, **params) -> None:
    src = Path(__file__).resolve().parents[2] / "data" / "raw" / "SOURCE.json"
    log = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {m.__name__: m.__version__ for m in (numpy, scipy, pandas, sklearn, matplotlib)},
        "dataset": json.loads(src.read_text()) if src.exists() else None,
        "params": params,
    }
    (out_dir / "runlog.json").write_text(json.dumps(log, indent=2, default=str))
