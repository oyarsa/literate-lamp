"Predictor for the McScript dataset"
from typing import TextIO, cast, Sequence

import torch
# Base class for the Model we'll implement. Inherits from `torch.nn.Model`,
# but compatible with what the rest of the AllenNLP library expects.
from allennlp.models import Model

# Abstract class used to implement a prediction class. This takes text input,
# converts to a format the model accepts and generates an output.
# This is done in a JSON -> model -> JSON pipeline.
from allennlp.predictors import Predictor

# Used to implement the `Predictor`'s JSON pipeline.
from allennlp.common import JsonDict

# A sample is represented as an `Instance`, which data as `TextField`s, and
# the target as a `LabelField`.
from allennlp.data import Instance

# This implements the logic for reading a data file and extracting a list of
# `Instance`s from it.
from allennlp.data.dataset_readers import DatasetReader

from allennlp.training.metrics import CategoricalAccuracy


@Predictor.register('mcscript-predictor')
class McScriptPredictor(Predictor):
    """
    Predictor for our Question Answering task. It takes a `Model` and a
    `DatasetReader`, from which it can read a dataset and compute predictions
    on it using the model.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict(self, passage_id: str, question_id: str, passage: str,
                question: str, answer0: str, answer1: str) -> JsonDict:
        """
        Takes the sample data and creates a prediction JSON from it.
        This will then be used to create an `Instance` that will be passed to
        the model.
        """
        return self.predict_json({
            "passage_id": passage_id,
            "question_id": question_id,
            "passage": passage,
            "question": question,
            "answer0": answer0,
            "answer1": answer1
        })

    # Extracts data from a JSON and creates an `Instance`.
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        passage_id = json_dict['passage_id']
        question_id = json_dict['question_id']
        passage = json_dict['passage']
        question = json_dict['question']
        answer0 = json_dict['answer0']
        answer1 = json_dict['answer1']

        return self._dataset_reader.text_to_instance(
            passage_id, question_id, passage, question, answer0, answer1)


def score_questions(model: Model,
                    output_file: TextIO,
                    testset: Sequence[Instance]
                    ) -> float:
    metric = CategoricalAccuracy()

    for instance in testset:
        prediction = model.forward_on_instance(instance)

        correct = instance['label']
        predicted = torch.argmax(prediction['prob'])

        metric(prediction['prob'], correct)

        passage_id = instance['passage_id']
        question_id = instance['question_id']

        print('{},{},{}'.format(passage_id, question_id, predicted),
              file=output_file)

    return cast(float, metric.get_metric())
