"Base model for the classifiers. Responsible for handling the metrics."
from typing import Dict

import torch
from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.vocabulary import Vocabulary


class BaseModel(Model):
    """
    Base class that sets up the vocabulary, as well as the accuracy and
    loss measures so the derived don't have to. These will never change
    unless the task changes.
    """

    def __init__(self, vocab: Vocabulary) -> None:
        # We have to pass the vocabulary to the constructor.
        super().__init__(vocab)
        # Categorical (as this is a classification task) accuracy
        self.accuracy = CategoricalAccuracy()
        # CrossEntropyLoss is a combinational of LogSoftmax and
        # Negative Log Likelihood. We won't directly use Softmax in training.
        self.loss = torch.nn.CrossEntropyLoss()

    # This function computes the metrics we want to see during training.
    # For now, we only have the accuracy metric, but we could have a number
    # of different metrics here.
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
