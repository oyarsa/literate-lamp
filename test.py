"""
This is a test on loading the merged data file and
creating the tuples (text, question, answer) for input,
and their respective output.
"""
import json
from typing import Tuple, List

with open('Merged/merged.json') as f:
    data = json.load(f)

tuples: List[Tuple[str, str, str]] = []
labels: List[str] = []

for instance in data['instances'][:3]:
    inst_text = instance['text']

    for question in instance['questions']:
        quest_text = question['@text']
        for answer in question['answer']:
            ans_text = answer['@text']
            ans_correct = answer['@correct']

            tuples.append((inst_text, quest_text, ans_text))
            labels.append(ans_correct)

for i, (inst_text, quest_text, ans_text) in enumerate(tuples):
    ans_correct = labels[i]
    print('("{}", "{}", "{}"): {}'.format(inst_text[:15], quest_text,
                                          ans_text, ans_correct))
