
"""
Continuous COP feature extraction for Improvement 1.

Raw hierarchy:
Organized/<subject>/<Firm|Foam>/<Open|Closed>/BDS*.txt

Unlike the original discrete pipeline, this module keeps angle and
distance as continuous observations.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

@dataclass
class ContinuousSequence:
    path: str
    subject_id: int | None
    condition: str
    angle: np.ndarray
    distance: np.ndarray
    max_angle: float
    max_distance: float

def read_bds_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    cols = {}
    for col in df.columns:
        n = str(col).strip().lower()
        if n.startswith("copx"):
            cols["copx"] = col
        elif n.startswith("copy"):
            cols["copy"] = col
    if len(cols) != 2:
        raise ValueError(f"Could not find COPx/COPy in {path}; columns={list(df.columns)}")
    x = pd.to_numeric(df[cols["copx"]], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[cols["copy"]], errors="coerce").to_numpy(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 20:
        raise ValueError(f"Too few valid COP samples in {path}: {len(x)}")
    return x, y

def extract_continuous_distance_angle(path, kappa=1, freq_div=5):
    if kappa < 1 or freq_div < 1:
        raise ValueError("kappa and freq_div must be >= 1")
    x, y = read_bds_file(path)
    x, y = x[::freq_div], y[::freq_div]
    if len(x) <= kappa:
        raise ValueError(f"Not enough samples after downsampling: {len(x)}")
    now = np.column_stack((x[kappa:], y[kappa:]))
    prev = np.column_stack((x[:-kappa], y[:-kappa]))
    delta = now - prev
    distance = np.linalg.norm(delta, axis=1)

    dot = np.sum(now * prev, axis=1)
    denom = np.linalg.norm(now, axis=1) * np.linalg.norm(prev, axis=1)
    cosine = np.ones_like(dot)
    valid = denom > 1e-12
    cosine[valid] = dot[valid] / denom[valid]
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    angle[~valid] = 0.0

    return ContinuousSequence(
        str(path), infer_subject_id(path), infer_condition(path),
        angle, distance,
        float(np.max(angle)), float(np.max(distance))
    )

def standardize_sequence(seq: ContinuousSequence) -> np.ndarray:
    X = np.column_stack((seq.angle, seq.distance)).astype(float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-12] = 1.0
    return (X - mean) / std

def infer_subject_id(path):
    for p in reversed(Path(path).parts):
        if p.isdigit():
            return int(p)
    return None

def infer_condition(path):
    parts = [p.lower() for p in Path(path).parts]
    surface = "Firm" if "firm" in parts else ("Foam" if "foam" in parts else None)
    vision = "Open" if "open" in parts else ("Closed" if "closed" in parts else None)
    return f"{surface}_{vision}" if surface and vision else "Unknown"

def find_bds_files(root):
    files = sorted(Path(root).rglob("BDS*.txt"))
    if not files:
        raise FileNotFoundError(f"No BDS*.txt files found below: {root}")
    return files
