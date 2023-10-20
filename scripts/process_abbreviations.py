import ujson

# Set up necessary variables/parameters
all_abbreviations = {}
min_confidence_cutoff = .95
omitted = 0
included = 0

# Read in data
filepath = 'data/raw_abbreviations.txt'
with open(filepath, 'r') as f:
    chunks = f.read().strip().split('\n\n')

# Get abbreviations from each article
for chunk in chunks:
    lines = chunk.split('\n')
    pmid = lines[0].split('|')[0].strip()
    abbrev_dict = {}
    for line in lines[1:]:
        abbrev, long_form, confidence_score = line.strip().split('|')
        confidence_score = float(confidence_score)
        if confidence_score > min_confidence_cutoff:
            abbrev_dict[abbrev] = long_form
            included += 1
        else:
            # print(abbrev, long_form, confidence_score)
            omitted += 1

    all_abbreviations[pmid] = abbrev_dict

with open('data/abbreviations.json', 'w') as f:
    f.write(ujson.dumps(all_abbreviations, indent=2))
            