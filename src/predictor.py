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


@Predictor.register('mcscript-predictor')
class McScriptPredictor(Predictor):
    """
    Predictor for our Question Answering task. It takes a `Model` and a
    `DatasetReader`, from which it can read a dataset and compute predictions
    on it using the model.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    # Takes the sample data and creates a prediction JSON from it.
    # This will then be used to create an `Instance` that will be passed to
    # the model.
    def predict(self, passage: str, question: str, answer: str) -> JsonDict:
        return self.predict_json(
            {"passage": passage, "question": question, "answer": answer})

    # Extracts data from a JSON and creates an `Instance`.
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        passage = json_dict['passage']
        question = json_dict['question']
        answer = json_dict['answer']

        return self._dataset_reader.text_to_instance(passage, question, answer)


class AnswerPredictor:
    """
    Given a passage, a question and two candidate answers, return which one
    of the answers is the (predicted) correct answer.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader,
                 verbose: bool = False) -> None:
        self.model = model
        self.reader = dataset_reader
        self.predictor = McScriptPredictor(model, dataset_reader)
        self.verbose = verbose

    def predict(self, passage: str, question: str, answer1: str, answer2: str
                ) -> int:
        """
        Given the passage, question and two candidate answers, return the index
        (0-indexed) for the right answer (0 for answer1, 1 for answer2).
        """
        prediction1 = self.predictor.predict(passage, question, answer1)
        prediction2 = self.predictor.predict(passage, question, answer2)

        prob1 = prediction1['prob']
        prob2 = prediction2['prob']

        if self.verbose:
            vec = torch.tensor((prob1, prob2)).transpose(0, 1)
            confidence = torch.softmax(vec, dim=1)
            print('Confidence:', confidence)

        if prob1 > prob2:
            return 0
        else:
            return 1
