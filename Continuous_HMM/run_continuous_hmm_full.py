
"""
FULL CONTINUOUS-HMM EXTRACTION — Improvement 1.

This implements the subject/condition aggregation required for the research
experiment. It does NOT classify yet.

For every BDS trial:
    COP -> continuous angle/distance -> Gaussian HMM -> 13-D embedding

For every subject-condition:
    mean(available trial embeddings) -> 13-D condition vector

For every subject:
    Firm_Open + Firm_Closed + Foam_Open + Foam_Closed -> 52-D vector

Missing conditions are filled with zeros, matching the original C program's
null-vector behavior. A separate completeness CSV records missing trials.

Run from repo root:
python Continuous_HMM/run_continuous_hmm_full.py --data ".\\Santos_and_Duarte\\The_HBED_Dataset\\Organized"

For a fast validation:
python Continuous_HMM/run_continuous_hmm_full.py --data ".\\Santos_and_Duarte\\The_HBED_Dataset\\Organized" --max-subjects 2
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import csv
import numpy as np
import pandas as pd

from cop_features import find_bds_files, extract_continuous_distance_angle, standardize_sequence
from continuous_hmm import ContinuousGaussianHMM

CONDITIONS = ["Firm_Open", "Firm_Closed", "Foam_Open", "Foam_Closed"]

def grouped_files(root):
    groups = defaultdict(list)
    for p in find_bds_files(root):
        seq = extract_continuous_distance_angle(p, kappa=1, freq_div=5)
        sid = seq.subject_id
        if sid is None or seq.condition not in CONDITIONS:
            continue
        groups[(sid, seq.condition)].append(p)
    return groups

def trial_embedding(path, states, seed, n_iter):
    seq = extract_continuous_distance_angle(path, kappa=1, freq_div=5)
    X = standardize_sequence(seq)
    model = ContinuousGaussianHMM(states, n_iter=n_iter, random_state=seed)
    model.fit([X])
    return model.embedding(seq.max_angle, seq.max_distance), len(X)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="Continuous_HMM/output")
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--states", type=int, default=11)
    ap.add_argument("--n-iter", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    groups = grouped_files(root)
    subjects = sorted({sid for sid, _ in groups})
    if args.max_subjects:
        subjects = subjects[:args.max_subjects]

    print(f"Subjects discovered: {len(subjects)}")
    print(f"BDS files discovered: {sum(len(v) for (s,c),v in groups.items() if s in subjects)}")
    print(f"Conditions: {', '.join(CONDITIONS)}")
    print(f"Continuous HMM states: {args.states}")

    rows = []
    completeness = []

    for idx, sid in enumerate(subjects, 1):
        row = {"SubjectID": sid}
        complete = True

        for condition in CONDITIONS:
            files = groups.get((sid, condition), [])
            embeddings = []
            trial_lengths = []

            for p in files:
                try:
                    emb, n = trial_embedding(p, args.states, args.seed, args.n_iter)
                    embeddings.append(emb)
                    trial_lengths.append(n)
                except Exception as e:
                    print(f"[WARN] Subject {sid} {condition} {p.name}: {e}")

            if embeddings:
                condition_vec = np.mean(np.vstack(embeddings), axis=0)
            else:
                condition_vec = np.zeros(args.states + 2, dtype=float)
                complete = False

            for j, value in enumerate(condition_vec):
                row[f"{condition}_{j+1:02d}"] = float(value)

            completeness.append({
                "SubjectID": sid,
                "Condition": condition,
                "TrialsFound": len(files),
                "TrialsUsed": len(embeddings),
                "MeanTrialLength": float(np.mean(trial_lengths)) if trial_lengths else 0.0,
                "Complete": int(bool(embeddings)),
            })

        rows.append(row)
        print(f"[{idx:03d}/{len(subjects):03d}] Subject {sid} done")

    df = pd.DataFrame(rows)
    df.to_csv(out / "continuous_subject_vectors.csv", index=False)
    pd.DataFrame(completeness).to_csv(out / "continuous_condition_completeness.csv", index=False)

    expected_cols = 1 + 4 * (args.states + 2)
    print("\n=== CONTINUOUS HMM EXTRACTION COMPLETE ===")
    print("Subjects:", len(df))
    print("Shape:", df.shape)
    print("Expected columns:", expected_cols)
    print("Missing values:", int(df.isna().sum().sum()))
    print("Output:", out / "continuous_subject_vectors.csv")
    print("Completeness:", out / "continuous_condition_completeness.csv")

if __name__ == "__main__":
    main()
