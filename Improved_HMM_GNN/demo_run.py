"""
DEMO RUN - HMM + Improved GNN V2
--------------------------------
Purpose:
    Short live demonstration of the proposed HMM + GNN pipeline.

IMPORTANT:
    This is a DEMO ONLY. Its result must NOT be presented as the 
    official final 10-fold / multi-seed experiment result. This is 
    a shortened experiment using 3-fold cross-validation and 20 epochs,
    whereas the main project uses 10-fold cross-validation, 150 epochs, 
    and 5 seeds for the final evaluation.

Expected project structure:
HMM-Based-HBE-main/
├── Improved_HMM_GNN/
│   └── demo_run.py
└── Oliveira + Santos_and_Duarte/
    └── Experiments/
        └── Arffs/
            └── Arff_Files/
                └── without_zscore10/
                    └── Parkinson10_Shift_1.arff

Run:
    python demo_run.py

Optional:
    python demo_run.py --epochs 20 --folds 3
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. SETTINGS
# ============================================================

SEED = 42
N_CONDITIONS = 4
FEATURES_PER_CONDITION = 13
N_NODES = N_CONDITIONS * FEATURES_PER_CONDITION
HIDDEN = 32
DROPOUT = 0.10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. ARFF LOADER
# ============================================================

def load_arff(path):
    """
    Loads the Parkinson ARFF used in the project.

    Expected format:
        @ATTRIBUTE class {Positive,Negative}
        @ATTRIBUTE x1 NUMERIC
        ...
        @DATA
        Positive,....
        Negative,....
    """

    rows = []
    data_started = False

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or line.startswith("%"):
                continue

            if line.lower() == "@data":
                data_started = True
                continue

            if data_started:
                parts = [x.strip() for x in line.split(",")]

                if len(parts) < 2:
                    continue

                label = parts[0]
                values = [float(x) for x in parts[1:]]

                rows.append((values, label))

    if not rows:
        raise ValueError("No data rows found in ARFF file.")

    X = np.array([r[0] for r in rows], dtype=np.float32)

    label_map = {
        "Negative": 0,
        "Positive": 1,
        "negative": 0,
        "positive": 1,
    }

    try:
        y = np.array([label_map[r[1]] for r in rows], dtype=np.int64)
    except KeyError as e:
        raise ValueError(f"Unknown class label in ARFF: {e}")

    if X.shape[1] != 52:
        raise ValueError(
            f"Expected 52 HMM features, but found {X.shape[1]}."
        )

    return X, y


# ============================================================
# 3. STRATIFIED K-FOLD SPLIT
# ============================================================

def stratified_folds(y, n_folds, seed):
    """
    Small dependency-free stratified K-fold implementation.
    """

    rng = np.random.default_rng(seed)

    classes = np.unique(y)
    class_indices = {}

    for c in classes:
        idx = np.where(y == c)[0].copy()
        rng.shuffle(idx)
        class_indices[c] = idx

    fold_bins = [[] for _ in range(n_folds)]

    for c in classes:
        idx = class_indices[c]

        for i, value in enumerate(idx):
            fold_bins[i % n_folds].append(int(value))

    folds = []

    for i in range(n_folds):
        val_idx = np.array(sorted(fold_bins[i]), dtype=np.int64)
        train_idx = np.array(
            sorted(
                set(range(len(y))) - set(val_idx.tolist())
            ),
            dtype=np.int64,
        )

        folds.append((train_idx, val_idx))

    return folds


# ============================================================
# 4. GRAPH CONSTRUCTION
# ============================================================

def build_graph():
    """
    52-node graph:

    4 conditions × 13 HMM features

    Within each condition:
        x1 -- x2 -- ... -- x13

    Across conditions:
        same feature position is connected

    Self-loops are added.

    Returns normalized:
        D^(-1/2) A D^(-1/2)
    """

    A = np.zeros((N_NODES, N_NODES), dtype=np.float32)

    # Self-loops
    for i in range(N_NODES):
        A[i, i] = 1.0

    # Sequential edges inside each condition
    for condition in range(N_CONDITIONS):
        start = condition * FEATURES_PER_CONDITION

        for feature in range(FEATURES_PER_CONDITION - 1):
            u = start + feature
            v = start + feature + 1

            A[u, v] = 1.0
            A[v, u] = 1.0

    # Cross-condition edges:
    # same feature position across the 4 conditions
    for feature in range(FEATURES_PER_CONDITION):
        nodes = [
            condition * FEATURES_PER_CONDITION + feature
            for condition in range(N_CONDITIONS)
        ]

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u = nodes[i]
                v = nodes[j]

                A[u, v] = 1.0
                A[v, u] = 1.0

    degree = A.sum(axis=1)

    d_inv_sqrt = np.power(degree, -0.5)
    D_inv_sqrt = np.diag(d_inv_sqrt)

    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    return torch.tensor(A_norm, dtype=torch.float32)


# ============================================================
# 5. GCN LAYER
# ============================================================

class GCNLayer(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, A):
        # A X W
        x = torch.matmul(A, x)
        x = self.linear(x)

        return x


# ============================================================
# 6. IMPROVED GNN V2
# ============================================================

class ImprovedGNNV2(nn.Module):

    def __init__(self):
        super().__init__()

        # GCN 1: 1 -> 32
        self.gcn1 = GCNLayer(1, HIDDEN)
        self.norm1 = nn.LayerNorm(HIDDEN)

        # Residual projection: 32 -> 32
        self.residual_projection = nn.Linear(HIDDEN, HIDDEN)

        # GCN 2: 32 -> 32
        self.gcn2 = GCNLayer(HIDDEN, HIDDEN)
        self.norm2 = nn.LayerNorm(HIDDEN)

        # Classifier:
        # Mean pooling + Max pooling = 64
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN * 2, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 2),
        )

    def forward(self, x, A):

        # x: [batch, 52, 1]

        # -------------------------
        # First GCN block
        # -------------------------
        h = self.gcn1(x, A)
        h = self.norm1(h)
        h = F.relu(h)
        h = F.dropout(h, p=DROPOUT, training=self.training)

        residual = self.residual_projection(h)

        # -------------------------
        # Second GCN block
        # -------------------------
        h = self.gcn2(h, A)
        h = self.norm2(h)

        # Residual connection
        h = h + residual

        h = F.relu(h)
        h = F.dropout(h, p=DROPOUT, training=self.training)

        # -------------------------
        # Global Mean + Max pooling
        # -------------------------
        mean_pool = h.mean(dim=1)
        max_pool = h.max(dim=1).values

        graph_embedding = torch.cat(
            [mean_pool, max_pool],
            dim=1
        )

        # -------------------------
        # Classification
        # -------------------------
        out = self.classifier(graph_embedding)

        return out


# ============================================================
# 7. METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = float(np.mean(y_true == y_pred))

    cm = np.zeros((2, 2), dtype=int)

    for actual, predicted in zip(y_true, y_pred):
        cm[actual, predicted] += 1

    total = cm.sum()

    if total == 0:
        kappa = 0.0
    else:
        po = np.trace(cm) / total

        row_marginals = cm.sum(axis=1)
        col_marginals = cm.sum(axis=0)

        pe = np.sum(row_marginals * col_marginals) / (total * total)

        if abs(1.0 - pe) < 1e-12:
            kappa = 0.0
        else:
            kappa = (po - pe) / (1.0 - pe)

    f1_scores = []

    for c in range(2):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        f1_scores.append(f1)

    macro_f1 = float(np.mean(f1_scores))

    return accuracy, kappa, macro_f1, cm


# ============================================================
# 8. TRAIN ONE FOLD
# ============================================================

def train_one_fold(
    X,
    y,
    train_idx,
    val_idx,
    A,
    epochs,
    fold_number,
):

    # Raw HMM values are used, matching the V2 experiment.
    X_train = torch.tensor(
        X[train_idx],
        dtype=torch.float32,
        device=DEVICE,
    ).view(-1, N_NODES, 1)

    y_train = torch.tensor(
        y[train_idx],
        dtype=torch.long,
        device=DEVICE,
    )

    X_val = torch.tensor(
        X[val_idx],
        dtype=torch.float32,
        device=DEVICE,
    ).view(-1, N_NODES, 1)

    y_val = torch.tensor(
        y[val_idx],
        dtype=torch.long,
        device=DEVICE,
    )

    model = ImprovedGNNV2().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4,
    )

    criterion = nn.CrossEntropyLoss()

    print(
        f"\n  Fold {fold_number}: "
        f"train={len(train_idx)}, validation={len(val_idx)}"
    )

    for epoch in range(1, epochs + 1):

        model.train()

        optimizer.zero_grad()

        logits = model(X_train, A)

        loss = criterion(logits, y_train)

        loss.backward()

        optimizer.step()

        if epoch == 1 or epoch == epochs:
            print(
                f"    Epoch {epoch:>3}/{epochs} "
                f"| Loss: {loss.item():.4f}"
            )

    # Validation
    model.eval()

    with torch.no_grad():
        logits = model(X_val, A)
        predictions = torch.argmax(logits, dim=1)

    return (
        y_val.cpu().numpy(),
        predictions.cpu().numpy(),
    )


# ============================================================
# 9. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Short live demo of Improved HMM + GNN V2"
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to Parkinson10_Shift_1.arff",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Demo training epochs per fold (default: 20)",
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=3,
        help="Number of CV folds for demo (default: 3)",
    )

    args = parser.parse_args()

    set_seed(SEED)

    # --------------------------------------------------------
    # Locate ARFF automatically
    # --------------------------------------------------------

    script_dir = Path(__file__).resolve().parent

    default_data = (
        script_dir.parent
        / "Oliveira + Santos_and_Duarte"
        / "Experiments"
        / "Arffs"
        / "Arff_Files"
        / "without_zscore10"
        / "Parkinson10_Shift_1.arff"
    )

    data_path = (
        Path(args.data).resolve()
        if args.data
        else default_data
    )

    print("=" * 70)
    print("       HMM + IMPROVED GNN V2 — LIVE DEMO")
    print("=" * 70)

    print("\nDEMO ONLY — NOT THE OFFICIAL FINAL RESULT")
    print("This run is only for demonstrating that the pipeline works.")

    print(f"\nDevice : {DEVICE}")
    print(f"ARFF   : {data_path}")
    print(f"Epochs : {args.epochs}")
    print(f"Folds  : {args.folds}")

    if not data_path.exists():
        print("\nERROR: ARFF file not found.")
        print("\nExpected location:")
        print(default_data)
        print("\nUse --data to provide the correct path.")
        return

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\n[1/5] Loading HMM feature dataset...")

    X, y = load_arff(data_path)

    print(f"      Samples : {len(X)}")
    print(f"      Features: {X.shape[1]}")
    print(
        f"      Classes : Negative={np.sum(y == 0)}, "
        f"Positive={np.sum(y == 1)}"
    )

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    print("\n[2/5] Building 52-node HMM graph...")

    A = build_graph().to(DEVICE)

    print("      Nodes   : 52")
    print("      Structure: 4 conditions × 13 HMM features")
    print("      Edges   : sequential + cross-condition + self-loops")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\n[3/5] Creating Improved GNN V2...")

    model = ImprovedGNNV2()

    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("      GCN 1       : 1 -> 32")
    print("      LayerNorm   : Yes")
    print("      Residual    : Yes")
    print("      GCN 2       : 32 -> 32")
    print("      Pooling     : Mean + Max")
    print("      Classifier  : 64 -> 32 -> 2")
    print(f"      Parameters  : {total_params:,}")

    # --------------------------------------------------------
    # Cross-validation
    # --------------------------------------------------------

    print("\n[4/5] Running short stratified CV demo...")

    folds = stratified_folds(
        y,
        n_folds=args.folds,
        seed=SEED,
    )

    all_true = []
    all_pred = []

    for fold_number, (train_idx, val_idx) in enumerate(
        folds,
        start=1,
    ):

        y_true, y_pred = train_one_fold(
            X,
            y,
            train_idx,
            val_idx,
            A,
            args.epochs,
            fold_number,
        )

        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())

        fold_acc, fold_kappa, fold_f1, fold_cm = calculate_metrics(
            y_true,
            y_pred,
        )

        print(
            f"    Fold result -> "
            f"Accuracy: {fold_acc * 100:.2f}% | "
            f"Kappa: {fold_kappa:.4f} | "
            f"Macro F1: {fold_f1 * 100:.2f}%"
        )

        print("    Confusion Matrix:")
        print(f"      {fold_cm[0, 0]:>3} {fold_cm[0, 1]:>3}")
        print(f"      {fold_cm[1, 0]:>3} {fold_cm[1, 1]:>3}")

    # --------------------------------------------------------
    # Final demo metrics
    # --------------------------------------------------------

    print("\n[5/5] Calculating overall DEMO metrics...")

    acc, kappa, macro_f1, cm = calculate_metrics(
        all_true,
        all_pred,
    )

    print("\n" + "=" * 70)
    print("                 DEMO RESULT")
    print("=" * 70)

    print(f"Accuracy : {acc * 100:.4f}%")
    print(f"Kappa    : {kappa:.4f}")
    print(f"Macro F1 : {macro_f1 * 100:.4f}%")

    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("               Negative Positive")
    print(
        f"Actual Negative    {cm[0,0]:>3}      {cm[0,1]:>3}"
    )
    print(
        f"Actual Positive    {cm[1,0]:>3}      {cm[1,1]:>3}"
    )

    print("\n" + "=" * 70)
    print("IMPORTANT:")
    print("This is a short DEMO RUN.")
    print("Do NOT quote this result as the official paper result.")
    print("Official results come from the full 10-fold multi-seed experiments.")
    print("=" * 70)


if __name__ == "__main__":
    main()
