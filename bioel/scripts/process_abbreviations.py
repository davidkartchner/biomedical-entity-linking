import ujson
import os

# Set up necessary variables/parameters
all_abbreviations = {}
min_confidence_cutoff = 0.95
omitted = 0
included = 0

# Read in data
output_file = "abbreviations.json"
output_dir = "/home2/cye73/data/solve_abbrev"

# Make sure the output directory exists
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, output_file)

input_file = os.path.join(output_dir, "raw_abbreviations.txt")

with open(input_file, "r") as f:
    chunks = f.read().strip().split("\n\n")

# Get abbreviations from each article
for chunk in chunks:
    lines = chunk.split("\n")
    pmid = lines[0].split("|")[0].strip()
    abbrev_dict = {}
    for line in lines[1:]:
        abbrev, long_form, confidence_score = line.strip().split("|")
        confidence_score = float(confidence_score)
        if confidence_score > min_confidence_cutoff:
            abbrev_dict[abbrev] = long_form
            included += 1
        else:
            # print(abbrev, long_form, confidence_score)
            omitted += 1

    all_abbreviations[pmid] = abbrev_dict

with open(output_path, "w") as f:
    f.write(ujson.dumps(all_abbreviations, indent=2))
