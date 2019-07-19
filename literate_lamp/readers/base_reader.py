"""
Implements the data extraction part. The generation of features is left to
the derived classes.
"""
from typing import Iterator
import json

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader


class BaseReader(DatasetReader):
    """Base class for the readers in this module.
    As they all read from the same type of file, the `_read` function is
    shared. However, they provide different features for different modules,
    so they create different `Instance`s
    """

    def _read(self, file_path: str) -> Iterator[Instance]:
        "Reads a file from `file_path` and returns a list of `Instances`."
        with open(file_path) as datafile:
            data = json.load(datafile)

        instances = data['data']['instance']
        for instance in instances:
            passage_id = instance['@id']
            passage = instance['text']['#text']

            if 'question' not in instance['questions']:
                instance['questions'] = []
            elif isinstance(instance['questions']['question'], list):
                instance['questions'] = instance['questions']['question']
            else:
                instance['questions'] = [instance['questions']['question']]
            questions = instance['questions']

            for question in questions:
                question_id = question['@id']
                question_text = question['@text']
                question_type = question['@type']

                answers = ["", ""]
                labels = ["", ""]

                for answer_dicts in question['answer']:
                    if answer_dicts['@id'] == '0':
                        index = 0
                    else:
                        index = 1

                    answers[index] = answer_dicts['@text']
                    labels[index] = answer_dicts['@correct']

                assert "" not in answers, "Answers have to be non-empty"
                assert "" not in labels, "Labels have to be non-empty"

                yield self.text_to_instance(passage_id, question_id,
                                            question_type, passage,
                                            question_text, answers[0],
                                            answers[1], labels[0])
