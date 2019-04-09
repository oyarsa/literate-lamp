#!/usr/bin/env python3
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

                for answer in question['answer']:
                    del answer['#tail']
            else:
                print(i, j, question)

        master.append(instance)

with open('merged.json', 'w') as f:
    json.dump({'instances': master}, f, indent=4)
