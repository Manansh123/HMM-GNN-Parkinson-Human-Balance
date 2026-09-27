import pandas as pd

file = "SubjectVectors.csv"

df = pd.read_csv(file, header=None)

df.columns = ["subject_id"] + [
    f"x{i}" for i in range(1, len(df.columns))
]

print("Dataset shape:", df.shape)

print("\nSubject IDs:")
print(df["subject_id"].head())

features = df.drop(columns=["subject_id"])

print("\nFeature Statistics:")
print(features.describe().T)

print("\nFeatures with highest standard deviation:")
print(features.std().sort_values(ascending=False).head(15))

print("\nHighly correlated feature pairs:")

corr = features.corr().abs()

for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        if corr.iloc[i, j] > 0.95:
            print(
                corr.columns[i],
                corr.columns[j],
                round(corr.iloc[i, j], 4)
            )

print("\nCorrelation with Subject ID:")

id_corr = features.corrwith(df["subject_id"]).abs()
print(id_corr.sort_values(ascending=False).head(10))