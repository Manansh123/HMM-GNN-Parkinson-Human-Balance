import re
import numpy as np
from pathlib import Path

sv_path = Path(r"Santos_and_Duarte\HMM_learning\SubjectVectors.txt")
arff_path = Path(r"Oliveira + Santos_and_Duarte\Experiments\Arffs\Arff_Files\without_zscore10\Parkinson10_Shift_1.arff")

subjects = []
vectors = []

for line in sv_path.read_text(errors="ignore").splitlines():
    m = re.match(r"\[(\d+)\],(.*)", line.strip())
    if m:
        subjects.append(int(m.group(1)))
        vectors.append(np.array([float(v) for v in m.group(2).split(",")]))

data = arff_path.read_text(errors="ignore").split("@DATA", 1)[1].strip().splitlines()

print("SubjectVectors:", len(subjects))
print("ARFF rows:", len(data))
print()

for i, line in enumerate(data, 1):
    parts = line.split(",")
    label = parts[0]
    arff_vector = np.array([float(v) for v in parts[1:]])

    errors = [np.mean((arff_vector - v) ** 2) for v in vectors]
    best = int(np.argmin(errors))

    print(
        f"ARFF {i:02d}: {label:<8} -> "
        f"Subject {subjects[best]:3d} | "
        f"MSE = {errors[best]:.12g}"
    )
