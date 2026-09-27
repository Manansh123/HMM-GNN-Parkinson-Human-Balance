import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

BASE_PATH = Path(
    r"C:\Users\gupta\Desktop\HMM-Based-HBE-main"
    r"\Oliveira + Santos_and_Duarte"
    r"\Experiments\Arffs\Arff_Files"
    r"\without_zscore10"
)

SHIFTS = [
    "Shift_1",
    "Shift_5",
    "Shift_10",
    "Shift_1_5",
    "Shift_1_10",
    "Shift_5_10",
    "Shift_1_5_10"
]

NUM_NODES = 52
NUM_FOLDS = 10
EPOCHS = 150

LR = 0.001
WEIGHT_DECAY = 1e-4

HIDDEN_DIM = 32
DROPOUT = 0.10

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")


# ============================================================
# READ ARFF
# ============================================================

def read_arff(path):

    X = []
    y = []

    with open(path, "r") as f:

        data_started = False

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.lower() == "@data":
                data_started = True
                continue

            if not data_started:
                continue

            if line.startswith("%"):
                continue

            parts = [
                x.strip()
                for x in line.split(",")
            ]

            if len(parts) < NUM_NODES + 1:
                continue

            label = parts[0]

            features = [
                float(x)
                for x in parts[1:NUM_NODES + 1]
            ]

            X.append(features)
            y.append(label)

    X = np.array(
        X,
        dtype=np.float32
    )

    classes = sorted(set(y))

    class_to_id = {
        c: i
        for i, c in enumerate(classes)
    }

    y = np.array(
        [
            class_to_id[label]
            for label in y
        ],
        dtype=np.int64
    )

    return X, y, classes


# ============================================================
# GRAPH
# ============================================================

def create_graph():

    A = np.zeros(
        (NUM_NODES, NUM_NODES),
        dtype=np.float32
    )

    # Four conditions × 13 features
    for condition in range(4):

        start = condition * 13

        for i in range(12):

            a = start + i
            b = start + i + 1

            A[a, b] = 1
            A[b, a] = 1

    # Same feature across conditions
    for feature in range(13):

        nodes = [
            feature,
            13 + feature,
            26 + feature,
            39 + feature
        ]

        for i in range(4):

            for j in range(i + 1, 4):

                A[
                    nodes[i],
                    nodes[j]
                ] = 1

                A[
                    nodes[j],
                    nodes[i]
                ] = 1

    # Self loops
    A += np.eye(
        NUM_NODES,
        dtype=np.float32
    )

    # GCN normalization
    degree = A.sum(axis=1)

    inv_sqrt = np.zeros_like(
        degree
    )

    mask = degree > 0

    inv_sqrt[mask] = (
        degree[mask] ** -0.5
    )

    D = np.diag(inv_sqrt)

    A = D @ A @ D

    return torch.tensor(
        A,
        dtype=torch.float32
    )


# ============================================================
# GRAPH CONVOLUTION
# ============================================================

class GraphConvolution(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim
    ):

        super().__init__()

        self.linear = nn.Linear(
            in_dim,
            out_dim
        )

    def forward(self, x, A):

        return self.linear(
            torch.matmul(A, x)
        )


# ============================================================
# RESIDUAL GNN
# ============================================================

class ImprovedGNN(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # First GCN
        # ----------------------------------------------------

        self.gcn1 = GraphConvolution(
            1,
            HIDDEN_DIM
        )

        self.norm1 = nn.LayerNorm(
            HIDDEN_DIM
        )

        # ----------------------------------------------------
        # Second GCN
        # ----------------------------------------------------

        self.gcn2 = GraphConvolution(
            HIDDEN_DIM,
            HIDDEN_DIM
        )

        self.norm2 = nn.LayerNorm(
            HIDDEN_DIM
        )

        # ----------------------------------------------------
        # Residual projection
        # ----------------------------------------------------

        self.residual = nn.Linear(
            HIDDEN_DIM,
            HIDDEN_DIM
        )

        self.dropout = nn.Dropout(
            DROPOUT
        )

        # ----------------------------------------------------
        # Classifier
        #
        # Mean = 32
        # Max  = 32
        # Total = 64
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                HIDDEN_DIM * 2,
                32
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                32,
                2
            )
        )

    def forward(
        self,
        x,
        A
    ):

        # ----------------------------------------------------
        # Layer 1
        # ----------------------------------------------------

        x = self.gcn1(
            x,
            A
        )

        x = self.norm1(
            x
        )

        x = F.relu(
            x
        )

        x = self.dropout(
            x
        )

        # Save residual
        identity = self.residual(
            x
        )

        # ----------------------------------------------------
        # Layer 2
        # ----------------------------------------------------

        x = self.gcn2(
            x,
            A
        )

        x = self.norm2(
            x
        )

        # ----------------------------------------------------
        # Residual connection
        # ----------------------------------------------------

        x = x + identity

        x = F.relu(
            x
        )

        x = self.dropout(
            x
        )

        # ----------------------------------------------------
        # Global Mean Pooling
        # ----------------------------------------------------

        mean_pool = x.mean(
            dim=0
        )

        # ----------------------------------------------------
        # Global Max Pooling
        # ----------------------------------------------------

        max_pool = x.max(
            dim=0
        ).values

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        graph_embedding = torch.cat(
            [
                mean_pool,
                max_pool
            ],
            dim=0
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        return self.classifier(
            graph_embedding
        )


# ============================================================
# STRATIFIED FOLDS
# ============================================================

def stratified_folds(y):

    rng = np.random.default_rng(
        SEED
    )

    class_0 = np.where(
        y == 0
    )[0]

    class_1 = np.where(
        y == 1
    )[0]

    rng.shuffle(class_0)
    rng.shuffle(class_1)

    folds = [
        []
        for _ in range(NUM_FOLDS)
    ]

    for i, idx in enumerate(class_0):

        folds[
            i % NUM_FOLDS
        ].append(idx)

    for i, idx in enumerate(class_1):

        folds[
            i % NUM_FOLDS
        ].append(idx)

    return [
        np.array(sorted(fold))
        for fold in folds
    ]


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    TP = np.sum(
        (y_true == 1) &
        (y_pred == 1)
    )

    TN = np.sum(
        (y_true == 0) &
        (y_pred == 0)
    )

    FP = np.sum(
        (y_true == 0) &
        (y_pred == 1)
    )

    FN = np.sum(
        (y_true == 1) &
        (y_pred == 0)
    )

    total = len(y_true)

    accuracy = (
        TP + TN
    ) / total

    precision_pos = (
        TP / (TP + FP)
        if TP + FP > 0
        else 0
    )

    recall_pos = (
        TP / (TP + FN)
        if TP + FN > 0
        else 0
    )

    f1_pos = (
        2 * precision_pos * recall_pos /
        (precision_pos + recall_pos)
        if precision_pos + recall_pos > 0
        else 0
    )

    precision_neg = (
        TN / (TN + FN)
        if TN + FN > 0
        else 0
    )

    recall_neg = (
        TN / (TN + FP)
        if TN + FP > 0
        else 0
    )

    f1_neg = (
        2 * precision_neg * recall_neg /
        (precision_neg + recall_neg)
        if precision_neg + recall_neg > 0
        else 0
    )

    macro_f1 = (
        f1_pos + f1_neg
    ) / 2

    # Cohen Kappa
    actual_pos = TP + FN
    actual_neg = TN + FP

    predicted_pos = TP + FP
    predicted_neg = TN + FN

    po = accuracy

    pe = (
        actual_pos * predicted_pos
        +
        actual_neg * predicted_neg
    ) / (total * total)

    kappa = (
        (po - pe) / (1 - pe)
        if pe != 1
        else 1.0
    )

    return {
        "accuracy": accuracy,
        "kappa": kappa,
        "macro_f1": macro_f1,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN
    }


# ============================================================
# TRAIN ONE FOLD
# ============================================================

def train_one_fold(
    X,
    y,
    train_idx,
    test_idx,
    A
):

    model = ImprovedGNN().to(
        DEVICE
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()

    X_train = torch.tensor(
        X[train_idx],
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y[train_idx],
        dtype=torch.long
    )

    X_test = torch.tensor(
        X[test_idx],
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    for epoch in range(EPOCHS):

        for i in range(
            len(X_train)
        ):

            node_features = X_train[
                i
            ].view(
                NUM_NODES,
                1
            )

            optimizer.zero_grad()

            output = model(
                node_features,
                A
            )

            loss = criterion(
                output.unsqueeze(0),
                y_train[i].unsqueeze(0)
            )

            loss.backward()

            optimizer.step()

    # --------------------------------------------------------
    # Testing
    # --------------------------------------------------------

    model.eval()

    predictions = []

    with torch.no_grad():

        for i in range(
            len(X_test)
        ):

            node_features = X_test[
                i
            ].view(
                NUM_NODES,
                1
            )

            output = model(
                node_features,
                A
            )

            predictions.append(
                torch.argmax(
                    output
                ).item()
            )

    return np.array(
        predictions
    )


# ============================================================
# MAIN
# ============================================================

def run_shift(shift_name):

    arff_path = (
        BASE_PATH /
        f"Parkinson10_{shift_name}.arff"
    )

    print("\n")
    print("=" * 70)
    print(f"RUNNING IMPROVED V2: {shift_name}")
    print("=" * 70)

    X, y, classes = read_arff(
        arff_path
    )

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

    A = create_graph()

    folds = stratified_folds(y)

    all_true = []
    all_pred = []

    fold_accuracies = []

    for fold in range(NUM_FOLDS):

        test_idx = folds[fold]

        train_idx = np.concatenate(
            [
                folds[i]
                for i in range(NUM_FOLDS)
                if i != fold
            ]
        )

        predictions = train_one_fold(
            X,
            y,
            train_idx,
            test_idx,
            A
        )

        actual = y[test_idx]

        accuracy = np.mean(
            predictions == actual
        )

        fold_accuracies.append(
            accuracy
        )

        all_true.extend(actual)
        all_pred.extend(predictions)

        print(
            f"Fold {fold + 1:02d}: "
            f"Accuracy = "
            f"{accuracy * 100:.4f}%"
        )

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    metrics = calculate_metrics(
        all_true,
        all_pred
    )

    print("\nFinal Result:")

    print(
        f"Accuracy : "
        f"{metrics['accuracy'] * 100:.4f}%"
    )

    print(
        f"Kappa    : "
        f"{metrics['kappa']:.4f}"
    )

    print(
        f"Macro F1 : "
        f"{metrics['macro_f1'] * 100:.4f}%"
    )

    print(
        f"TN={metrics['TN']} "
        f"FP={metrics['FP']} "
        f"FN={metrics['FN']} "
        f"TP={metrics['TP']}"
    )

    return {
        "shift": shift_name,
        "accuracy": metrics["accuracy"] * 100,
        "kappa": metrics["kappa"],
        "macro_f1": metrics["macro_f1"] * 100
    }


def main():

    print("=" * 70)
    print("IMPROVED HMM + GNN V2 — ALL 7 SHIFTS")
    print("=" * 70)

    results = []

    for shift in SHIFTS:

        result = run_shift(
            shift
        )

        results.append(
            result
        )

    print("\n\n")
    print("=" * 75)
    print("FINAL IMPROVED V2 RESULTS")
    print("=" * 75)

    print(
        f"{'Shift':<15}"
        f"{'Accuracy':>12}"
        f"{'Kappa':>12}"
        f"{'Macro F1':>12}"
    )

    print("-" * 75)

    for result in results:

        print(
            f"{result['shift']:<15}"
            f"{result['accuracy']:>11.4f}%"
            f"{result['kappa']:>12.4f}"
            f"{result['macro_f1']:>11.4f}%"
        )

    print("=" * 75)


if __name__ == "__main__":
    main()