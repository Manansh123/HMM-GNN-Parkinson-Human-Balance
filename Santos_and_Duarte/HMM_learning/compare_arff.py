import pandas as pd
import numpy as np

base = r"C:\Users\gupta\Desktop\HMM-Based-HBE-main\Santos_and_Duarte"

files = {
    "Shift_1": base + r"\Experiments\Arffs\Create_Arffs\Shift_1\Vectors.txt",
    "Shift_5": base + r"\Experiments\Arffs\Create_Arffs\Shift_5\Vectors.txt",
    "Shift_10": base + r"\Experiments\Arffs\Create_Arffs\Shift_10\Vectors.txt"
}

arff = r"C:\Users\gupta\Desktop\HMM-Based-HBE-main\Oliveira + Santos_and_Duarte\Experiments\Arffs\Arff_Files\without_zscore10\Parkinson10_Shift_1_5_10.arff"

def read_vectors(path):
    rows = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.strip("[]").split(",")

            subject_id = int(parts[0].strip())
            values = [float(x.strip()) for x in parts[1:]]

            rows.append([subject_id] + values)

    return pd.DataFrame(rows)


print("Reading shift vectors...")

s1 = read_vectors(files["Shift_1"])
s5 = read_vectors(files["Shift_5"])
s10 = read_vectors(files["Shift_10"])

print("Shift 1 :", s1.shape)
print("Shift 5 :", s5.shape)
print("Shift 10:", s10.shape)

# Sort by subject ID
s1 = s1.sort_values(0).reset_index(drop=True)
s5 = s5.sort_values(0).reset_index(drop=True)
s10 = s10.sort_values(0).reset_index(drop=True)

# Check subject IDs
print("\nSubject IDs identical:")

print(
    np.array_equal(
        s1[0].values,
        s5[0].values
    )
    and
    np.array_equal(
        s1[0].values,
        s10[0].values
    )
)

# Combine 52 features from each shift
combined = np.concatenate(
    [
        s1.iloc[:, 1:].values,
        s5.iloc[:, 1:].values,
        s10.iloc[:, 1:].values
    ],
    axis=1
)

print("\nCombined feature shape:", combined.shape)

# Read ARFF data
arff_rows = []

reading_data = False

with open(arff, "r") as f:

    for line in f:

        line = line.strip()

        if line.upper() == "@DATA":
            reading_data = True
            continue

        if reading_data and line:
            parts = line.split(",")

            label = parts[0]
            values = [float(x) for x in parts[1:]]

            arff_rows.append([label] + values)

arff_values = np.array(
    [row[1:] for row in arff_rows],
    dtype=float
)

print("ARFF feature shape:", arff_values.shape)

# Compare
max_difference = np.max(
    np.abs(combined - arff_values)
)

mean_difference = np.mean(
    np.abs(combined - arff_values)
)

print("\nMaximum absolute difference:", max_difference)
print("Mean absolute difference:", mean_difference)

if np.allclose(combined, arff_values, rtol=1e-5, atol=1e-5):
    print("\nRESULT: ARFF matches Shift 1 + Shift 5 + Shift 10 vectors.")
else:
    print("\nRESULT: ARFF does NOT exactly match the combined vectors.")