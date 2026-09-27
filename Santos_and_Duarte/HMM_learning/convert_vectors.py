
import csv

input_file = "SubjectVectors.txt"
output_file = "SubjectVectors.csv"

with open(input_file, "r") as infile, open(output_file, "w", newline="") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        row[0] = row[0].strip("[]")
        writer.writerow(row)

print("Conversion completed!")
print("Output file:", output_file)