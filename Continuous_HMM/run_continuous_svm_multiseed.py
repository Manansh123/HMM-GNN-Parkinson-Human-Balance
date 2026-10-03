import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, cohen_kappa_score


DATA = "Continuous_HMM/output/continuous_parkinson_89.csv"

SEEDS = [42, 123, 2024, 7, 99]

df = pd.read_csv(DATA)

X = df.drop(columns=["SubjectID", "Class"]).to_numpy(dtype=float)
y = df["Class"].map({
    "Negative": 0,
    "Positive": 1
}).to_numpy()

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale"
    ))
])

all_accuracy = []
all_f1 = []
all_kappa = []

print("\n=== Continuous HMM + SVM ===")
print("Samples :", len(df))
print("Features:", X.shape[1])
print("Seeds   :", SEEDS)
print("CV      : 10-fold")
print()

for seed in SEEDS:

    cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=seed
    )

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "kappa": make_scorer(cohen_kappa_score)
    }

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    acc = results["test_accuracy"]
    f1 = results["test_f1_macro"]
    kap = results["test_kappa"]

    all_accuracy.extend(acc)
    all_f1.extend(f1)
    all_kappa.extend(kap)

    print(
        f"Seed {seed}: "
        f"Accuracy={acc.mean()*100:.4f}% | "
        f"Macro F1={f1.mean()*100:.4f}% | "
        f"Kappa={kap.mean():.4f}"
    )

print("\n========================================")
print("FINAL 5-SEED × 10-FOLD RESULTS")
print("========================================")

print(
    f"Accuracy : "
    f"{np.mean(all_accuracy)*100:.4f}% ± "
    f"{np.std(all_accuracy)*100:.4f}%"
)

print(
    f"Macro F1 : "
    f"{np.mean(all_f1)*100:.4f}% ± "
    f"{np.std(all_f1)*100:.4f}%"
)

print(
    f"Kappa    : "
    f"{np.mean(all_kappa):.4f} ± "
    f"{np.std(all_kappa):.4f}"
)

print("\nTotal folds:", len(all_accuracy))