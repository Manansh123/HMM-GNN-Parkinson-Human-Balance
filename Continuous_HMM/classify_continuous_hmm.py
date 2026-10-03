
"""
Classification for the Continuous-HMM representation.

This script joins the generated 52-feature subject vectors with the existing
Parkinson ARFF labels and compares SVM, Random Forest and MLP using the same
10-fold subject-level CV idea. It is intentionally separate from extraction.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

def read_arff(path):
    rows, labels = [], []
    in_data = False
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower() == "@data":
                in_data = True
                continue
            if not in_data or line.startswith("%"):
                continue
            parts = [x.strip() for x in line.split(",")]
            labels.append(parts[0])
            rows.append([float(x) for x in parts[1:]])
    return np.asarray(rows), np.asarray(labels)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default="Continuous_HMM/output/continuous_subject_vectors.csv")
    ap.add_argument("--arff", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42,52,62,72,82])
    args = ap.parse_args()

    v = pd.read_csv(args.vectors)
    v = v.set_index("SubjectID")
    Xall, yraw = read_arff(args.arff)

    # Existing ARFF is class-first and the 52 features correspond to subject IDs
    # in the same subject order. We therefore use the first column count to select
    # subjects 1..N only when explicit IDs are unavailable.
    if len(Xall) != 89:
        print(f"Warning: expected 89 Parkinson rows, found {len(Xall)}")

    ids = list(v.index)
    # If ARFF rows are the known 89 Parkinson instances, use the existing
    # subject ordering only after confirming IDs externally. This script prints
    # the required mapping warning instead of silently inventing a mapping.
    raise SystemExit(
        "STOP: provide an explicit SubjectID -> Parkinson-label mapping before "
        "classification. Extraction is complete, but label identity must not be guessed."
    )

if __name__ == "__main__":
    main()
