#!/usr/bin/env python3
"""
Merges the .json files in the folder (dev, train and test from MCScript, dev
and train from COIN) into a single .json file, removing artifacts from the XML
conversion.

These artifacts are:
    - #text, #tail: random text added by the tool, irrelevant to the real text
    - @text: some questions have repeated question texts in the same field,
             this removes the repetition

Also, we re-do the indexing, to avoid duplicating the indexes.
"""
import json
from typing import List, Dict, Any

files = [
    'coindev-data.json',
    'cointrain-data.json',
    'mcdev-data.json',
    'mctrain-data.json',
    'mctest-data.json',
]

master: List[Dict[str, Any]] = []
instance_id = 0

for name in files:
    with open(name) as f:
        jfile = json.load(f)

    instances = jfile['data']['instance']

    for i, instance in enumerate(instances):
        instance['@id'] = instance_id
        instance_id += 1

        del instance['#tail']
        del instance['#text']

        del instance['text']['#tail']
        instance['text'] = instance['text']['#text']

        questions = instance['questions']
        questions.pop('#tail', None)
        questions.pop('#text', None)

        if 'question' not in instance['questions']:
            instance['questions'] = []
        elif type(instance['questions']['question']) is list:
            instance['questions'] = instance['questions']['question']
        else:
            instance['questions'] = [instance['questions']['question']]
        questions = instance['questions']

        for j, question in enumerate(questions):
            if type(question) is dict:
                del question['#tail']
                del question['#text']
                text = question['@text']
                if '|' in text:
                    text = text.split('|')[0]
                    question['@text'] = text

                for answer in question['answer']:
                    del answer['#tail']
            else:
                print(i, j, question)

        master.append(instance)

with open('merged.json', 'w') as f:
    json.dump({'instances': master}, f, indent=4)
