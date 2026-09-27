import csv

results = [
    ["Shift", "Original SMO Accuracy", "Original J48 Accuracy",
     "Basic GNN Accuracy", "Improved GNN V2 Accuracy",
     "Improved V2 Kappa", "Improved V2 Macro F1"],

    ["1",     98.8764, 98.8764, 97.7528, 100.0000, 1.0000, 100.0000],
    ["5",    100.0000, 97.7528, 100.0000, 100.0000, 1.0000, 100.0000],
    ["10",   100.0000, 98.8764, 100.0000, 98.8764, 0.9758, 98.7883],
    ["1+5",   98.8764, 98.8764, 96.6292, 100.0000, 1.0000, 100.0000],
    ["1+10",  98.8764, 98.8764, 96.6292, 100.0000, 1.0000, 100.0000],
    ["5+10", 100.0000, 97.7528, 100.0000, 100.0000, 1.0000, 100.0000],
    ["1+5+10",98.8764, 98.8764, 96.6292, 100.0000, 1.0000, 100.0000],
]

output_file = "final_comparison.csv"

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(results)

print("\nFINAL COMPARISON")
print("-" * 100)

for row in results:
    print(row)

print("\nSaved:", output_file)