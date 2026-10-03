from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from cop_features import (
    find_bds_files,
    extract_continuous_distance_angle,
    standardize_sequence,
)

from continuous_hmm import ContinuousGaussianHMM


# ============================================================
# CONFIGURATION
# ============================================================

CONDITIONS = [
    "Firm_Open",
    "Firm_Closed",
    "Foam_Open",
    "Foam_Closed",
]

STATE_VALUES = [7, 9, 11, 13]
KAPPA_VALUES = [1, 5, 10]
FREQ_DIV_VALUES = [3, 5, 7]

SEEDS = [42, 123, 2024, 7, 99]

N_ITER = 100

OLIVEIRA_ROOT = Path(
    "Oliveira/The_Parkinson_Dataset/Organized"
)

SANTOS_ROOT = Path(
    "Santos_and_Duarte/The_HBED_Dataset/Organized"
)

NO_ILLNESS_FILE = Path(
    "Oliveira + Santos_and_Duarte/"
    "Experiments/Arffs/Create_Arffs/Shift_1/"
    "No_illness.txt"
)

OUTPUT_DIR = Path(
    "Continuous_HMM/phase2_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OLIVEIRA FILE DISCOVERY
# ============================================================

def find_oliveira_files():

    groups = defaultdict(list)

    for subject_dir in sorted(OLIVEIRA_ROOT.iterdir()):

        if not subject_dir.is_dir():
            continue

        try:
            sid = int(subject_dir.name)
        except ValueError:
            continue

        if not 1 <= sid <= 32:
            continue

        for condition, relative_path in {
            "Firm_Open": Path("Firm") / "Open",
            "Firm_Closed": Path("Firm") / "Closed",
            "Foam_Open": Path("Foam") / "Open",
            "Foam_Closed": Path("Foam") / "Closed",
        }.items():

            condition_dir = subject_dir / relative_path

            if not condition_dir.exists():
                continue

            for path in sorted(condition_dir.glob("*.txt")):
                groups[(sid, condition)].append(path)

    return groups

# ============================================================
# SANTOS FILE DISCOVERY
# ============================================================

def find_santos_files(
    kappa,
    freq_div
):

    groups = defaultdict(list)

    for path in find_bds_files(
        SANTOS_ROOT
    ):

        try:

            seq = extract_continuous_distance_angle(
                path,
                kappa=kappa,
                freq_div=freq_div,
            )

        except Exception:

            continue

        sid = seq.subject_id

        if (
            sid is None
            or seq.condition not in CONDITIONS
        ):
            continue

        groups[
            (sid, seq.condition)
        ].append(path)

    return groups


# ============================================================
# OLIVEIRA EXTRACTION
# ============================================================

def oliveira_trial_embedding(
    path,
    states,
    kappa,
    freq_div,
    seed,
):

    seq = extract_continuous_distance_angle(
        path,
        kappa=kappa,
        freq_div=freq_div,
    )

    X = standardize_sequence(seq)

    model = ContinuousGaussianHMM(
        states,
        n_iter=N_ITER,
        random_state=seed,
    )

    model.fit([X])

    return model.embedding(
        seq.max_angle,
        seq.max_distance,
    )


# ============================================================
# SANTOS EXTRACTION
# ============================================================

def santos_trial_embedding(
    path,
    states,
    kappa,
    freq_div,
    seed,
):

    seq = extract_continuous_distance_angle(
        path,
        kappa=kappa,
        freq_div=freq_div,
    )

    X = standardize_sequence(seq)

    model = ContinuousGaussianHMM(
        states,
        n_iter=N_ITER,
        random_state=seed,
    )

    model.fit([X])

    return model.embedding(
        seq.max_angle,
        seq.max_distance,
    )


# ============================================================
# CONDITION AGGREGATION
# ============================================================

def aggregate_condition(
    files,
    dataset,
    states,
    kappa,
    freq_div,
    seed,
):

    embeddings = []

    for path in files:

        try:

            if dataset == "oliveira":

                emb = oliveira_trial_embedding(
                    path,
                    states,
                    kappa,
                    freq_div,
                    seed,
                )

            else:

                emb = santos_trial_embedding(
                    path,
                    states,
                    kappa,
                    freq_div,
                    seed,
                )

            embeddings.append(
                emb
            )

        except Exception:
            continue

    if not embeddings:

        return np.zeros(
            states + 2,
            dtype=float,
        )

    return np.mean(
        np.vstack(embeddings),
        axis=0,
    )


# ============================================================
# LOAD NEGATIVE IDS
# ============================================================

def load_negative_ids():

    ids = []

    with open(
        NO_ILLNESS_FILE,
        "r",
    ) as f:

        for line in f:

            line = line.strip()

            if line:
                ids.append(
                    int(line)
                )

    return ids


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset(
    oliveira_groups,
    santos_groups,
    states,
    kappa,
    freq_div,
    seed,
):

    rows = []

    # --------------------------------------------------------
    # POSITIVE — Oliveira
    # --------------------------------------------------------

    for sid in sorted(
        {
            x[0]
            for x in oliveira_groups.keys()
        }
    ):

        features = []

        for condition in CONDITIONS:

            files = oliveira_groups.get(
                (sid, condition),
                [],
            )

            emb = aggregate_condition(
                files,
                "oliveira",
                states,
                kappa,
                freq_div,
                seed,
            )

            features.extend(
                emb.tolist()
            )

        rows.append(
            [sid]
            + features
            + ["Positive"]
        )

    # --------------------------------------------------------
    # NEGATIVE — Santos
    # --------------------------------------------------------

    negative_ids = load_negative_ids()

    for sid in negative_ids:

        features = []

        for condition in CONDITIONS:

            files = santos_groups.get(
                (sid, condition),
                [],
            )

            emb = aggregate_condition(
                files,
                "santos",
                states,
                kappa,
                freq_div,
                seed,
            )

            features.extend(
                emb.tolist()
            )

        rows.append(
            [sid]
            + features
            + ["Negative"]
        )

    n_features = 4 * (
        states + 2
    )

    columns = (
        ["SubjectID"]
        +
        [
            f"Feature_{i+1:03d}"
            for i in range(
                n_features
            )
        ]
        +
        ["Class"]
    )

    df = pd.DataFrame(
        rows,
        columns=columns,
    )

    return df


# ============================================================
# SVM EVALUATION
# ============================================================

def evaluate_svm(
    df,
    seed,
):

    X = df.drop(
        columns=[
            "SubjectID",
            "Class",
        ]
    ).to_numpy(
        dtype=float
    )

    y = df["Class"].map(
        {
            "Negative": 0,
            "Positive": 1,
        }
    ).to_numpy()

    cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=seed,
    )

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                ),
            ),
        ]
    )

    acc = []
    f1 = []
    kappa = []

    for train_idx, test_idx in cv.split(
        X,
        y,
    ):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        model.fit(
            X_train,
            y_train,
        )

        pred = model.predict(
            X_test
        )

        acc.append(
            accuracy_score(
                y_test,
                pred,
            )
        )

        f1.append(
            f1_score(
                y_test,
                pred,
                average="macro",
            )
        )

        kappa.append(
            cohen_kappa_score(
                y_test,
                pred,
            )
        )

    return (
        np.mean(acc),
        np.mean(f1),
        np.mean(kappa),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n========================================"
    )
    print(
        "PHASE 2 — ADAPTIVE PARAMETER SEARCH"
    )
    print(
        "EXACT CONTINUOUS-HMM PIPELINE"
    )
    print(
        "========================================\n"
    )

    print(
        "Loading Oliveira files..."
    )

    oliveira_groups = (
        find_oliveira_files()
    )

    print(
        "Oliveira subjects:",
        len(
            {
                sid
                for sid, _ in oliveira_groups
            }
        ),
    )

    negative_ids = load_negative_ids()

    print(
        "Negative Santos subjects:",
        len(negative_ids),
    )

    configurations = list(
        itertools.product(
            STATE_VALUES,
            KAPPA_VALUES,
            FREQ_DIV_VALUES,
        )
    )

    print(
        "Configurations:",
        len(configurations),
    )

    print(
        "Total CV evaluations:",
        len(configurations)
        * len(SEEDS),
    )

    results = []

    for number, (
        states,
        kappa,
        freq_div,
    ) in enumerate(
        configurations,
        start=1,
    ):

        print(
            f"\n[{number}/{len(configurations)}] "
            f"states={states}, "
            f"kappa={kappa}, "
            f"freq_div={freq_div}"
        )

        seed_acc = []
        seed_f1 = []
        seed_kappa = []

        # Build Santos groups ONCE for this
        # extraction configuration.

        print(
            "  Extracting Santos..."
        )

        santos_groups = (
            find_santos_files(
                kappa,
                freq_div,
            )
        )

        for seed in SEEDS:

            print(
                f"  Seed {seed}..."
            )

            df = build_dataset(
                oliveira_groups,
                santos_groups,
                states,
                kappa,
                freq_div,
                seed,
            )

            # Safety checks
            assert len(df) == 89

            assert (
                df["Class"]
                .value_counts()
                ["Positive"]
                == 32
            )

            assert (
                df["Class"]
                .value_counts()
                ["Negative"]
                == 57
            )

            X = df.drop(
                columns=[
                    "SubjectID",
                    "Class",
                ]
            )

            assert np.isfinite(
                X.to_numpy(
                    dtype=float
                )
            ).all()

            acc, f1, kap = evaluate_svm(
                df,
                seed,
            )

            seed_acc.append(acc)
            seed_f1.append(f1)
            seed_kappa.append(kap)

            print(
                f"    Acc={acc*100:.4f}% "
                f"F1={f1*100:.4f}% "
                f"Kappa={kap:.4f}"
            )

        result = {
            "states": states,
            "kappa": kappa,
            "freq_div": freq_div,
            "accuracy_mean": np.mean(
                seed_acc
            ),
            "accuracy_std": np.std(
                seed_acc
            ),
            "f1_mean": np.mean(
                seed_f1
            ),
            "f1_std": np.std(
                seed_f1
            ),
            "kappa_mean": np.mean(
                seed_kappa
            ),
            "kappa_std": np.std(
                seed_kappa
            ),
        }

        results.append(
            result
        )

        pd.DataFrame(
            results
        ).to_csv(
            OUTPUT_DIR
            / "phase2_progress.csv",
            index=False,
        )

        print(
            f"  >>> Accuracy: "
            f"{result['accuracy_mean']*100:.4f}%"
        )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "accuracy_mean",
        ascending=False,
    )

    results_df.to_csv(
        OUTPUT_DIR
        / "phase2_results_sorted.csv",
        index=False,
    )

    best = results_df.iloc[0]

    print(
        "\n========================================"
    )
    print(
        "PHASE 2 COMPLETE"
    )
    print(
        "========================================"
    )

    print(
        "\nBest configuration:"
    )

    print(
        "States   :",
        int(best["states"]),
    )

    print(
        "Kappa    :",
        int(best["kappa"]),
    )

    print(
        "Freq div :",
        int(best["freq_div"]),
    )

    print(
        f"\nAccuracy : "
        f"{best['accuracy_mean']*100:.4f}% "
        f"+/- "
        f"{best['accuracy_std']*100:.4f}%"
    )

    print(
        f"Macro F1 : "
        f"{best['f1_mean']*100:.4f}% "
        f"+/- "
        f"{best['f1_std']*100:.4f}%"
    )

    print(
        f"Kappa    : "
        f"{best['kappa_mean']:.4f} "
        f"+/- "
        f"{best['kappa_std']:.4f}"
    )

    print(
        "\nTop 10 configurations:"
    )

    print(
        results_df.head(10).to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_DIR
        / "phase2_results_sorted.csv"
    )


if __name__ == "__main__":
    main()