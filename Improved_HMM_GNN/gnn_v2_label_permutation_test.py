import numpy as np
import torch

from gnn_improved_v2_all_shifts_multiseed import (
    read_arff,
    create_graph,
    run_experiment,
    ARFF_DIR,
    SHIFT_FILES
)


# ============================================================
# CONFIG
# ============================================================

NUM_PERMUTATIONS = 5

# Different random permutations of the labels
PERMUTATION_SEEDS = [
    101,
    202,
    303,
    404,
    505
]

# Different CV/model seeds
CV_SEEDS = [
    42,
    52,
    62,
    72,
    82
]

SHIFT_NAME = "Shift 1"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("HMM + GNN V2 — LABEL PERMUTATION SANITY CHECK")
    print("=" * 80)

    filename = SHIFT_FILES[SHIFT_NAME]

    arff_path = ARFF_DIR / filename

    print("\nDataset:")
    print(arff_path)

    # --------------------------------------------------------
    # Load original dataset
    # --------------------------------------------------------

    X, y, classes = read_arff(
        arff_path
    )

    print("\nOriginal Dataset:")
    print(
        f"Samples : {len(X)}"
    )

    print(
        f"Features: {X.shape[1]}"
    )

    print(
        f"Positive: {np.sum(y == 1)}"
    )

    print(
        f"Negative: {np.sum(y == 0)}"
    )

    print(
        f"Classes : {classes}"
    )

    # --------------------------------------------------------
    # Create graph
    # --------------------------------------------------------

    A = create_graph()

    all_results = []

    # ========================================================
    # LABEL PERMUTATIONS
    # ========================================================

    for permutation_number, permutation_seed in enumerate(
        PERMUTATION_SEEDS,
        start=1
    ):

        print("\n\n")
        print("=" * 80)
        print(
            f"PERMUTATION {permutation_number}"
        )
        print(
            f"Permutation Seed: {permutation_seed}"
        )
        print("=" * 80)

        # ----------------------------------------------------
        # Randomly shuffle labels
        #
        # Features X remain unchanged.
        # Only labels are randomized.
        # ----------------------------------------------------

        rng = np.random.default_rng(
            permutation_seed
        )

        y_permuted = rng.permutation(
            y
        )

        print(
            f"Shuffled Positive: "
            f"{np.sum(y_permuted == 1)}"
        )

        print(
            f"Shuffled Negative: "
            f"{np.sum(y_permuted == 0)}"
        )

        permutation_results = []

        # ----------------------------------------------------
        # Run 5 CV seeds
        # ----------------------------------------------------

        for cv_seed in CV_SEEDS:

            print("\n")
            print(
                f"  CV Seed: {cv_seed}"
            )
            print(
                "  " + "-" * 60
            )

            result = run_experiment(
                X,
                y_permuted,
                cv_seed,
                A
            )

            permutation_results.append(
                result
            )

            print(
                f"      Accuracy : "
                f"{result['accuracy']:.4f}%"
            )

            print(
                f"      Kappa    : "
                f"{result['kappa']:.4f}"
            )

            print(
                f"      Macro F1 : "
                f"{result['macro_f1']:.4f}%"
            )

        # ----------------------------------------------------
        # Permutation statistics
        # ----------------------------------------------------

        accuracies = np.array(
            [
                r["accuracy"]
                for r in permutation_results
            ]
        )

        kappas = np.array(
            [
                r["kappa"]
                for r in permutation_results
            ]
        )

        macro_f1s = np.array(
            [
                r["macro_f1"]
                for r in permutation_results
            ]
        )

        result_summary = {
            "accuracy_mean":
                np.mean(accuracies),

            "accuracy_std":
                np.std(accuracies),

            "kappa_mean":
                np.mean(kappas),

            "macro_f1_mean":
                np.mean(macro_f1s)
        }

        all_results.append(
            result_summary
        )

        print("\nPermutation Result:")

        print(
            f"Accuracy Mean : "
            f"{result_summary['accuracy_mean']:.4f}%"
        )

        print(
            f"Accuracy Std  : "
            f"{result_summary['accuracy_std']:.4f}%"
        )

        print(
            f"Kappa Mean    : "
            f"{result_summary['kappa_mean']:.4f}"
        )

        print(
            f"Macro F1 Mean : "
            f"{result_summary['macro_f1_mean']:.4f}%"
        )

    # ========================================================
    # FINAL PERMUTATION RESULTS
    # ========================================================

    accuracy_means = np.array(
        [
            r["accuracy_mean"]
            for r in all_results
        ]
    )

    kappa_means = np.array(
        [
            r["kappa_mean"]
            for r in all_results
        ]
    )

    macro_f1_means = np.array(
        [
            r["macro_f1_mean"]
            for r in all_results
        ]
    )

    print("\n\n")
    print("=" * 90)
    print(
        "FINAL LABEL-PERMUTATION RESULTS"
    )
    print("=" * 90)

    print(
        f"{'Permutation':<15}"
        f"{'Accuracy':>18}"
        f"{'Kappa':>15}"
        f"{'Macro F1':>18}"
    )

    print("-" * 90)

    for i in range(
        NUM_PERMUTATIONS
    ):

        print(
            f"{i + 1:<15}"
            f"{accuracy_means[i]:>17.4f}%"
            f"{kappa_means[i]:>15.4f}"
            f"{macro_f1_means[i]:>17.4f}%"
        )

    print("-" * 90)

    print(
        f"{'Overall Mean':<15}"
        f"{np.mean(accuracy_means):>17.4f}%"
        f"{np.mean(kappa_means):>15.4f}"
        f"{np.mean(macro_f1_means):>17.4f}%"
    )

    print(
        f"{'Overall Std':<15}"
        f"{np.std(accuracy_means):>17.4f}%"
        f"{np.std(kappa_means):>15.4f}"
        f"{np.std(macro_f1_means):>17.4f}%"
    )

    print("=" * 90)

    # ========================================================
    # INTERPRETATION
    # ========================================================

    overall_accuracy = np.mean(
        accuracy_means
    )

    print("\nInterpretation:")

    if overall_accuracy < 65:

        print(
            "PASS: Randomized labels produce "
            "near-chance performance."
        )

    elif overall_accuracy < 80:

        print(
            "CAUTION: Random-label performance "
            "is somewhat above chance."
        )

    else:

        print(
            "WARNING: Random-label performance "
            "is unusually high. Investigate "
            "the pipeline for possible leakage."
        )

    print("=" * 90)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()