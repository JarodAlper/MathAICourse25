import random

#Prompt: Using the oeis-dataset file, choose a random sequence from the file and
#store separately (1) the first 10 terms of the sequence and (2) the A-number of the sequence
# Path to the OEIS dataset file
oeis_file = '/Users/jarod/gitwork/playground/oeis-dataset'

with open(oeis_file, 'r') as f:
    lines = f.readlines()

print("Total lines in OEIS dataset:", len(lines))
random_line = random.choice(lines).strip()
print("Random line selected:", random_line)
# Expected format: "A000001 ,0,1,1,1,2,1,2,1,5,2,2,1,5,1,2"
# Strategy:
# 1. Split once on the first space to separate A-number token from the rest.
# 2. Remove any leading comma and whitespace from the remainder.
# 3. Split on commas and strip each term; drop empties.

# Split on first space only (some lines may have multiple spaces)
if ' ' in random_line:
    a_number_part, terms_part = random_line.split(' ', 1)
else:
    a_number_part, terms_part = random_line, ''

a_number = a_number_part.strip()
# Remove leading comma(s) and spaces
terms_part = terms_part.lstrip(' ,')

raw_terms = [t.strip() for t in terms_part.split(',') if t.strip() != '']

# Convert to integers where possible
terms_int = []
for t in raw_terms:
    try:
        terms_int.append(int(t))
    except ValueError:
        # Skip non-integer tokens silently; alternatively log
        continue

first_10_terms = terms_int[:10]

print("A-number:", a_number)
print("First 10 terms:", first_10_terms)