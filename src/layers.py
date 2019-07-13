"""
Implements some layers that AllenNLP doesn't have.
"""
from typing import Optional, Tuple
from pathlib import Path

import torch
from overrides import overrides
from allennlp.common import Params
from allennlp.modules.attention import Attention
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.nn.activations import Activation
from allennlp.data.vocabulary import Vocabulary
# These create text embeddings from a `TextField` input. Since our text data
# is represented using `TextField`s, this makes sense.
# Again, `TextFieldEmbedder` is the abstract class, `BasicTextFieldEmbedder`
# is the implementation (as we're just using simple embeddings, no fancy
# ELMo or BERT so far).
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
# This is the actual neural layer for the embedding. This will be passed into
# the embedder above.
from allennlp.modules.token_embedders import Embedding, PretrainedBertEmbedder
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper, CnnEncoder
from allennlp.modules.seq2seq_encoders import (
    Seq2SeqEncoder, PytorchSeq2SeqWrapper, StackedSelfAttentionEncoder)


class LinearSelfAttention(Attention):
    """
    Implements linear self attention.

        alpha = softmax(Wx)    (1)

    Outputs the scores (alpha), which should be used to compute a weighted sum
    of v. That means that the other part:

        Att(u, v) = sum_i(alpha_i * v_i)   (2)

    Is not computed here. Should be ran as:

        allennlp.util.weighted_sum(v, alpha)   (3)

    This was done so this class is consistent with how `Attention`-derived
    classes work in AllenNLP, as they only output the scores.

    Although the equation (1) doesn't mention masks, we do accept the mask for
    the second vector (v), and compute the softmax using that mask (zero-ing
    the padded dimensions).
    """

    def __init__(self,
                 input_dim: int,
                 normalise: bool = True,
                 bias: bool = False) -> None:
        super().__init__(normalise)
        self._weights = torch.nn.Linear(in_features=input_dim,
                                        out_features=1, bias=bias)
        self.input_dim = input_dim

    @overrides
    def _forward_internal(self, vector: torch.Tensor, _: torch.Tensor
                          ) -> torch.Tensor:
        Wx = self._weights(vector).squeeze(-1)
        return Wx


class BilinearAttention(Attention):
    """
    Implements bilinear attention.

        alpha = softmax(x'Wy)    (1)

    Outputs the scores (alpha), which should be used to compute a weighted sum
    of v. That means that the other part:

        Att(u, v) = sum_i(alpha_i * v_i)   (2)

    Is not computed here. Should be ran as:

        allennlp.util.weighted_sum(v, alpha)   (3)

    This was done so this class is consistent with how `Attention`-derived
    classes work in AllenNLP, as they only output the scores.

    Although the equation (1) doesn't mention masks, we do accept the mask for
    the second vector (v), and compute the softmax using that mask (zero-ing
    the padded dimensions).
    """

    def __init__(self,
                 vector_dim: int,
                 matrix_dim: int,
                 normalise: bool = True) -> None:
        super().__init__(normalise)
        self._weights = torch.nn.Linear(in_features=vector_dim,
                                        out_features=matrix_dim)

    @overrides
    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor
                          ) -> torch.Tensor:
        Wy = self._weights(vector).unsqueeze(1)
        alpha = Wy.bmm(matrix.transpose(1, 2)).squeeze(1)
        return alpha


class SequenceAttention(Attention):
    """
    Implements sequence attention as defined in Yuanfudao

        alpha = softmax(f(Wu)f(Wv))    (1)

    Outputs the scores (alpha), which should be used to compute a weighted sum
    of v. That means that the other part:

        Att_seq(u, v) = sum_i(alpha_i * v_i)   (2)

    Is not computed here. Should be ran as:

        allennlp.util.weighted_sum(v, alpha)   (3)

    This was done so this class is consistent with how `Attention`-derived
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


def learned_embeddings(vocab: Vocabulary, dimension: int,
                       namespace: str = 'tokens') -> BasicTextFieldEmbedder:
    "Returns an Embedding layer to be learned, i.e., not pre-trained."
    embedding = Embedding(num_embeddings=vocab.get_vocab_size(namespace),
                          embedding_dim=dimension)
    embeddings = BasicTextFieldEmbedder({namespace: embedding})
    return embeddings


def bert_embeddings(pretrained_model: Path, training: bool = False,
                    top_layer_only: bool = True
                    ) -> BasicTextFieldEmbedder:
    "Pre-trained embeddings using BERT"
    bert = PretrainedBertEmbedder(
        requires_grad=training,
        pretrained_model=pretrained_model,
        top_layer_only=top_layer_only
    )
    word_embeddings = BasicTextFieldEmbedder(
        token_embedders={'tokens': bert},
        embedder_to_indexer_map={'tokens': ['tokens', 'tokens-offsets']},
        allow_unmatched_keys=True)
    return word_embeddings


def glove_embeddings(vocab: Vocabulary, file_path: Path, dimension: int,
                     training: bool = False, namespace: str = 'tokens'
                     ) -> BasicTextFieldEmbedder:
    "Pre-trained embeddings using GloVe"
    token_embedding = Embedding.from_params(vocab, Params({
        "embedding_dim": dimension,
        "vocab_namespace": 'tokens',
        "pretrained_file": str(file_path)
    }))
    word_embeddings = BasicTextFieldEmbedder({namespace: token_embedding})
    return word_embeddings


def lstm_seq2seq(input_dim: int, output_dim: int, num_layers: int = 1,
                 bidirectional: bool = False, dropout: float = 0.0
                 ) -> Seq2SeqEncoder:
    """
    Our encoder is going to be an LSTM. We have to wrap it for AllenNLP,
    though.
    """
    return PytorchSeq2SeqWrapper(torch.nn.LSTM(
        input_dim, output_dim, batch_first=True, num_layers=num_layers,
        bidirectional=bidirectional, dropout=dropout))


def gru_seq2seq(input_dim: int, output_dim: int, num_layers: int = 1,
                bidirectional: bool = False, dropout: float = 0.0
                ) -> Seq2SeqEncoder:
    """
    Our encoder is going to be an LSTM. We have to wrap it for AllenNLP,
    though.
    """
    return PytorchSeq2SeqWrapper(torch.nn.GRU(
        input_dim, output_dim, batch_first=True, num_layers=num_layers,
        bidirectional=bidirectional, dropout=dropout))


def transformer_seq2seq(input_dim: int,
                        hidden_dim: int,
                        feedforward_hidden_dim: int = 2048,
                        num_layers: int = 6,
                        projection_dim: int = 64,
                        num_attention_heads: int = 8,
                        dropout: float = 0.1) -> Seq2SeqEncoder:
    return StackedSelfAttentionEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        feedforward_hidden_dim=feedforward_hidden_dim,
        num_layers=num_layers,
        projection_dim=projection_dim,
        num_attention_heads=num_attention_heads)


def lstm_encoder(input_dim: int, output_dim: int, num_layers: int = 1,
                 bidirectional: bool = False, dropout: float = 0.0
                 ) -> Seq2VecEncoder:
    """
    Our encoder is going to be an LSTM. We have to wrap it for AllenNLP,
    though.
    """
    return PytorchSeq2VecWrapper(torch.nn.LSTM(
        input_dim, output_dim, batch_first=True, num_layers=num_layers,
        bidirectional=bidirectional, dropout=dropout))


def gru_encoder(input_dim: int, output_dim: int, num_layers: int = 1,
                bidirectional: bool = False, dropout: float = 0.0
                ) -> Seq2VecEncoder:
    """
    Our encoder is going to be an GRU. We have to wrap it for AllenNLP,
    though.
    """
    return PytorchSeq2VecWrapper(torch.nn.GRU(
        input_dim, output_dim, batch_first=True, num_layers=num_layers,
        bidirectional=bidirectional, dropout=dropout))


def cnn_encoder(input_dim: int, output_dim: int, num_filters: int,
                ngram_filter_sizes: Tuple[int, ...] = (2, 3)
                ) -> Seq2VecEncoder:
    return CnnEncoder(embedding_dim=input_dim, output_dim=output_dim,
                      num_filters=num_filters,
                      ngram_filter_sizes=ngram_filter_sizes)
