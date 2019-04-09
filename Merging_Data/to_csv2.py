#!/usr/bin/env python3
"""
This is a test on loading the merged data file and creating the tuples
(text, question, answer) for input, and their respective output. The tuples,
however, are joined into a single string (with separator `#`).
Generates a csv (separated with `\t` with the tuples.
"""
import json
from typing import Tuple, List

JSON_INPUT = 'merged.json'
CSV_OUTPUT = 'data_joined.csv'

with open(JSON_INPUT) as f:
    data = json.load(f)

tuples: List[Tuple[str, str, str, int]] = []

for instance in data['instances']:
    inst_text = instance['text']

    for question in instance['questions']:
        quest_text = question['@text']
        for answer in question['answer']:
            ans_text = answer['@text']
            ans_correct = int(answer['@correct'] == 'True')

            tuples.append((inst_text, quest_text, ans_text, ans_correct))

# Remove duplicates whilst keeping order
tuples = list(dict.fromkeys(tuples))

# Print some tuples to the screen
for inst_text, quest_text, ans_text, ans_correct in tuples[:15]:
    print('{}#{}#{}\t{}'.format(inst_text[:15], quest_text,
                                ans_text, ans_correct))

# Output all the tuples to a csv file with `|` as a separator, as
# the strings can have commas. Why not tabs? No clue, didn't think of them.
with open(CSV_OUTPUT, 'w') as out_file:
    print('Instance|Question|Answer\tLabel', file=out_file)
    for inst_text, quest_text, ans_text, ans_correct in tuples:
        print('{}#{}#{}\t{}'.format(inst_text, quest_text,
                                    ans_text, ans_correct),
              file=out_file)
