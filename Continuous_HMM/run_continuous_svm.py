import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, f1_score, cohen_kappa_score


DATA = "Continuous_HMM/output/continuous_parkinson_89.csv"

df = pd.read_csv(DATA)

X = df.drop(columns=["SubjectID", "Class"])
y = df["Class"]

# Encode classes
y = y.map({"Negative": 0, "Positive": 1}).to_numpy()

X = X.to_numpy(dtype=float)

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale"))
])

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

print("\n=== Continuous HMM + SVM ===")
print("Samples:", len(df))
print("Features:", X.shape[1])

print("\nAccuracy:")
print("Mean:", results["test_accuracy"].mean())
print("Std :", results["test_accuracy"].std())

print("\nMacro F1:")
print("Mean:", results["test_f1_macro"].mean())
print("Std :", results["test_f1_macro"].std())

print("\nCohen Kappa:")
print("Mean:", results["test_kappa"].mean())
print("Std :", results["test_kappa"].std())