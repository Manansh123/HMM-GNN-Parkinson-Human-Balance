
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
import re
from collections import defaultdict
from pathlib import Path
from xml.parsers.expat import model

import numpy as np
import pandas as pd

from continuous_hmm import ContinuousGaussianHMM


CONDITIONS = {
    "Firm_Open": Path("Firm") / "Open",
    "Firm_Closed": Path("Firm") / "Closed",
    "Foam_Open": Path("Foam") / "Open",
    "Foam_Closed": Path("Foam") / "Closed",
}


def get_subject_id(subject_dir: Path):
    try:
        return int(subject_dir.name)
    except ValueError:
        return None


def find_oliveira_files(root: Path):
    groups = defaultdict(list)

    for subject_dir in sorted(root.iterdir()):
        if not subject_dir.is_dir():
            continue

        sid = get_subject_id(subject_dir)

        if sid is None:
            continue

        for condition, relative_path in CONDITIONS.items():
            condition_dir = subject_dir / relative_path

            if not condition_dir.exists():
                continue

            for path in sorted(condition_dir.glob("*.txt")):
                groups[(sid, condition)].append(path)

    return groups


def read_cop_file(path: Path):
    """
    Oliveira format:

    Time
    GRFml
    GRFap
    GRFv
    Mml
    Map
    Mv
    Mfree
    COPml
    COPap
    """

    data = np.loadtxt(
        path,
        delimiter="\t",
        skiprows=1,
    )

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 10:
        raise ValueError(
            f"Expected at least 10 columns, got {data.shape[1]}"
        )

    cop_x = data[:, 8]   # COPml
    cop_y = data[:, 9]   # COPap

    return cop_x, cop_y


def extract_angle_distance(
    path: Path,
    kappa: int = 1,
    freq_div: int = 5,
):
    cop_x, cop_y = read_cop_file(path)

    # Same basic downsampling idea as original C implementation.
    cop_x = cop_x[::freq_div]
    cop_y = cop_y[::freq_div]

    if len(cop_x) <= kappa:
        raise ValueError("Not enough COP samples after downsampling.")

    p0 = np.column_stack(
        [cop_x[:-kappa], cop_y[:-kappa]]
    )

    p1 = np.column_stack(
        [cop_x[kappa:], cop_y[kappa:]]
    )

    v0 = p0
    v1 = p1

    norm0 = np.linalg.norm(v0, axis=1)
    norm1 = np.linalg.norm(v1, axis=1)

    dot = np.sum(v0 * v1, axis=1)

    denominator = norm0 * norm1

    valid = denominator > 1e-12

    cos_theta = np.ones_like(dot)

    cos_theta[valid] = (
        dot[valid] / denominator[valid]
    )

    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle = np.degrees(np.arccos(cos_theta))

    distance = np.sqrt(
        (p1[:, 0] - p0[:, 0]) ** 2
        + (p1[:, 1] - p0[:, 1]) ** 2
    )

    valid_values = (
        np.isfinite(angle)
        & np.isfinite(distance)
    )

    angle = angle[valid_values]
    distance = distance[valid_values]

    if len(angle) == 0:
        raise ValueError("No valid angle/distance observations.")

    X = np.column_stack([angle, distance])

    return X, float(np.max(angle)), float(np.max(distance))


def trial_embedding(
    path: Path,
    states: int,
    seed: int,
    n_iter: int,
):
    X, max_angle, max_distance = extract_angle_distance(
        path,
        kappa=1,
        freq_div=5,
    )

    # Continuous normalization only.
    # No discretization is performed.
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    std = np.maximum(std, 1e-8)

    X_scaled = (X - mean) / std

    model = ContinuousGaussianHMM(
        states,
        n_iter=n_iter,
        random_state=seed,
    )

    model.fit([X_scaled])

    embedding = model.embedding(
        max_angle,
        max_distance,
    )

    return embedding, len(X)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
        help="Path to Oliveira/The_Parkinson_Dataset/Organized",
    )

    parser.add_argument(
        "--out",
        default="Continuous_HMM/output_oliveira",
    )

    parser.add_argument(
        "--max-subjects",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--states",
        type=int,
        default=11,
    )

    parser.add_argument(
        "--n-iter",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    root = Path(args.data)
    out = Path(args.out)

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    groups = find_oliveira_files(root)

    subjects = sorted(
        {
            sid
            for sid, _ in groups.keys()
            if 1 <= sid <= 32
        }
    )

    if args.max_subjects:
        subjects = subjects[:args.max_subjects]

    total_files = sum(
        len(files)
        for (sid, _), files in groups.items()
        if sid in subjects
    )

    print()
    print("==============================================")
    print(" OLIVEIRA CONTINUOUS HMM")
    print("==============================================")
    print(f"Subjects discovered : {len(subjects)}")
    print(f"TXT trials          : {total_files}")
    print(f"States              : {args.states}")
    print(f"Iterations          : {args.n_iter}")
    print("Time shift          : 1")
    print("Downsampling        : 5")
    print("==============================================")
    print()

    rows = []
    completeness = []

    for idx, sid in enumerate(subjects, 1):

        row = {
            "SubjectID": sid
        }

        print(
            f"[{idx:02d}/{len(subjects):02d}] "
            f"Subject {sid}"
        )

        for condition in CONDITIONS:

            files = groups.get(
                (sid, condition),
                []
            )

            embeddings = []
            lengths = []

            print(
                f"  {condition}: "
                f"{len(files)} trial(s)"
            )

            for path in files:

                try:

                    emb, length = trial_embedding(
                        path,
                        args.states,
                        args.seed,
                        args.n_iter,
                    )

                    embeddings.append(emb)
                    lengths.append(length)

                except Exception as exc:

                    print(
                        f"    [WARN] {path.name}: "
                        f"{exc}"
                    )

            if embeddings:

                condition_vector = np.mean(
                    np.vstack(embeddings),
                    axis=0,
                )

            else:

                condition_vector = np.zeros(
                    args.states + 2,
                    dtype=float,
                )

            for j, value in enumerate(
                condition_vector
            ):

                row[
                    f"{condition}_{j + 1:02d}"
                ] = float(value)

            completeness.append(
                {
                    "SubjectID": sid,
                    "Condition": condition,
                    "TrialsFound": len(files),
                    "TrialsUsed": len(embeddings),
                    "MeanTrialLength": (
                        float(np.mean(lengths))
                        if lengths
                        else 0.0
                    ),
                    "Complete": int(
                        len(embeddings) > 0
                    ),
                }
            )

        rows.append(row)

    df = pd.DataFrame(rows)

    completeness_df = pd.DataFrame(
        completeness
    )

    vector_file = (
        out /
        "oliveira_continuous_subject_vectors.csv"
    )

    completeness_file = (
        out /
        "oliveira_continuous_completeness.csv"
    )

    df.to_csv(
        vector_file,
        index=False,
    )

    completeness_df.to_csv(
        completeness_file,
        index=False,
    )

    expected_columns = (
        1 + 4 * (args.states + 2)
    )

    print()
    print("==============================================")
    print(" EXTRACTION COMPLETE")
    print("==============================================")
    print(f"Shape              : {df.shape}")
    print(f"Expected columns   : {expected_columns}")
    print(
        f"Missing values     : "
        f"{int(df.isna().sum().sum())}"
    )
    print(f"Vector output      : {vector_file}")
    print(f"Completeness       : {completeness_file}")
    print("==============================================")
    print()


if __name__ == "__main__":
    main()