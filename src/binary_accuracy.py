from typing import Optional

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric


@Metric.register("binary_accuracy")
class BinaryAccuracy(Metric):
    """
    Binary accuracy. Assumes predictions are real numbers in [0, 1], and labels
    are integers in {0, 1}.
    """

    def __init__(self) -> None:
        self.correct_count = 0.0
        self.total_count = 0.0

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None) -> None:
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, 1).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size). It must
            be the same shape as the ``predictions`` tensor.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(
            predictions, gold_labels, mask)

        # Some sanity checks.
        if predictions.dim() != 2:
            raise ConfigurationError(
                "predictions should have only two dimensions "
                "(one being batch_size), but it currently has {}. "
                "If you have multiple classes, please use CategoricalAccuracy"
                .format(predictions.size(-1)))
        if gold_labels.dim() != 1:
            raise ConfigurationError(
                "gold_labels should have only one dimension "
                "(batch_size), but it currently has {}. "
                "If you have multiple classes, please use CategoricalAccuracy"
                .format(predictions.size(-1)))

        predictions = predictions.view(-1)

        correct = predictions.round().eq(gold_labels.float())
        correct.unsqueeze_(-1)

        if mask is not None:
            correct *= mask.view(-1, 1).float()
            self.total_count += mask.sum()
        else:
            self.total_count += gold_labels.numel()
        self.correct_count += correct.sum()

    def get_metric(self, reset: bool = False) -> float:
        """
        Returns
        -------
        The accumulated accuracy.
        """
        if self.total_count > 1e-12:
            accuracy = float(self.correct_count) / float(self.total_count)
        else:
            accuracy = 0.0

        if reset:
            self.reset()

        return accuracy

    @overrides
    def reset(self) -> None:
        self.correct_count = 0.0
        self.total_count = 0.0
