from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MLRUNS = ROOT / "mlruns"
MLRUNS.mkdir(exist_ok=True)

os.chdir(ROOT)
# Use a proper file:// URI, not a raw Windows path. MLflow parses the
# --backend-store-uri as a URI, and a bare "D:\..." path is misread as
# scheme="d" (the drive letter + colon looks like a URI scheme), which
# breaks the model registry store lookup. Path.as_uri() produces a
# correct "file:///D:/..." URI on Windows (and "file:///..." on POSIX).
MLRUNS_URI = MLRUNS.as_uri()
subprocess.run(
    [sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", MLRUNS_URI],
    check=True,
)