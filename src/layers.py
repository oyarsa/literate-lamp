"""
Implements some layers that AllenNLP doesn't have.
"""
from typing import Optional
import torch
from overrides import overrides
from allennlp.modules.attention import Attention
from allennlp.nn.activations import Activation


class SequenceAttention(Attention):
    """
    Implements sequence attention as defined in Yuanfudao

        alpha = softmax(f(Wu)f(Wv))    (1)

    Outputs the scores (alpha), which should be used to compute a weighted sum
    of v. That means that the other part:

        Att_seq(u, v) = sum_i(alpha_i * v_i)   (2)

    Is not computed here. Should be ran as:

        allennlp.util.weighted_sum(v, alpha)   (3)

    This was done so this class ir consistent with how `Attention`-derived
    classes work in AllenNLP, as they only output the scores.

    Although the equation (1) doesn't mention masks, we do accept the mask for
    the second vector (v), and compute the softmax using that mask (zero-ing
    the padded dimensions).
    """

    def __init__(self,
                 input_dim: int,
                 activation: Optional[Activation] = None,
                 normalise: bool = True) -> None:
        super().__init__(normalise)
        self._weights = torch.nn.Linear(in_features=input_dim,
                                        out_features=input_dim)
        self._activation = activation or Activation.by_name('relu')()

    @overrides
    def _forward_internal(self, u: torch.Tensor, v: torch.Tensor
                          ) -> torch.Tensor:
        u_prime = self._activation(self._weights(u))
        v_prime = self._activation(self._weights(v))
        alpha = u_prime.bmm(v_prime.transpose(1, 2))
        return alpha