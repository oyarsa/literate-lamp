"""
Implements some layers that AllenNLP doesn't have.
"""
from typing import Optional, Tuple
from pathlib import Path

import torch
from overrides import overrides
from allennlp.common import Params
from allennlp.modules.attention import Attention
from allennlp.nn.util import masked_softmax, weighted_sum
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


@Seq2SeqEncoder.register("multi_head_attention")
class MultiHeadAttention(Seq2SeqEncoder):
    # pylint: disable=line-too-long
    """
    This class implements the key-value scaled dot product attention mechanism
    detailed in the paper `Attention is all you Need`.
    The attention mechanism is a weighted sum of a projection V of the inputs,
    with respect to the scaled, normalised dot product of Q and K, which are
    also both linear projections of the input. This procedure is repeated for
    each attention head, using different parameters.

    The output will have as many items in the sequence as the query Q, and
    each vector will have the same dimension as V. V and K must have the
    same dimension.

    Parameters
    ----------
    num_heads : ``int``, required.
        The number of attention heads to use.
    key_input_dim : ``int``, required.
        The size of the last dimension of the keys tensor.
    query_input_dim : ``int``, required.
        The size of the last dimension of the queries tensor.
    value_input_dim : ``int``, required.
        The size of the last dimension of the keys tensor.
    attention_dim ``int``, required.
        The total dimension of the query and key projections which comprise the
        dot product attention function. Must be divisible by ``num_heads``.
    values_dim : ``int``, required.
        The total dimension which the input is projected to for representing
        the values, which are combined using the attention. Must be divisible
        by ``num_heads``.
    output_projection_dim : ``int``, optional (default = None)
        The dimensionality of the final output projection. If this is not
        passed explicitly, the projection has size `input_size`.
    attention_dropout_prob : ``float``, optional (default = 0.1).
        The dropout probability applied to the normalised attention
        distributions.
    """

    def __init__(self,
                 num_heads: int,
                 query_input_dim: int,
                 key_input_dim: int,
                 value_input_dim: int,
                 attention_dim: int,
                 values_dim: int,
                 output_projection_dim: int = Optional[None],
                 attention_dropout_prob: float = 0.1) -> None:
        super(MultiHeadAttention, self).__init__()

        self._num_heads = num_heads
        self._query_input_dim = query_input_dim
        self._key_input_dim = key_input_dim
        self._value_input_dim = value_input_dim
        self._output_dim = output_projection_dim or value_input_dim
        self._attention_dim = attention_dim
        self._values_dim = values_dim

        if value_input_dim != key_input_dim:
            raise ValueError(f"Key input size ({key_input_dim}) and "
                             f"Value input size ({value_input_dim}) must be "
                             f"equal.")

        if attention_dim % num_heads != 0:
            raise ValueError(f"Key size ({attention_dim}) must be divisible by"
                             f" the number of attention heads ({num_heads}).")

        if values_dim % num_heads != 0:
            raise ValueError(f"Value size ({values_dim}) must be divisible by "
                             f"the number of attention heads ({num_heads}).")

        self._q_projection = torch.nn.Linear(query_input_dim, attention_dim)
        self._k_projection = torch.nn.Linear(key_input_dim, attention_dim)
        self._v_projection = torch.nn.Linear(value_input_dim, values_dim)

        self._scale = (key_input_dim // num_heads) ** 0.5
        self._output_projection = torch.nn.Linear(values_dim, self._output_dim)
        self._attention_dropout = torch.nn.Dropout(attention_dropout_prob)

    def get_input_dim(self) -> int:
        return self._query_input_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    @overrides
    def is_bidirectional(self) -> bool:
        return False

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                keys: torch.Tensor,
                queries: torch.Tensor,
                values: torch.Tensor,
                mask: torch.LongTensor = None) -> torch.FloatTensor:
        """
        Parameters
        ----------
        keys : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps_2, input_dim)
        queries : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps_1, input_dim)
        values : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps_2, input_dim)
        mask : ``torch.FloatTensor``, optional (default = None).
            A tensor of shape (batch_size, timesteps_2).
        Returns
        -------
        A tensor of shape (batch_size, timesteps_1, output_projection_dim),
        where output_projection_dim = input_dim by default.
        """
        num_heads = self._num_heads

        _, timesteps_1, _ = queries.shape
        batch_size, timesteps_2, _ = values.shape

        if mask is None:
            mask = values.new_ones(batch_size, timesteps_2)

        # Shape (batch_size, timesteps_1, attention_dim)
        queries = self._q_projection(queries)
        # Shape (batch_size, timesteps_2, attention_dim)
        keys = self._k_projection(keys)
        # Shape (batch_size, timesteps_2, values_dim)
        values = self._v_projection(values)

        # Shape (num_heads * batch_size, timesteps_2, values_dim / num_heads)
        values_per_head = values.view(batch_size, timesteps_2, num_heads,
                                      int(self._values_dim/num_heads))
        values_per_head = values_per_head.transpose(1, 2).contiguous()
        values_per_head = values_per_head.view(
            batch_size * num_heads, timesteps_2,
            int(self._values_dim/num_heads))

        # Shape (num_heads * batch_size, timesteps_1, attention_dim/num_heads)
        queries_per_head = queries.view(batch_size, timesteps_1, num_heads,
                                        int(self._attention_dim/num_heads))
        queries_per_head = queries_per_head.transpose(1, 2).contiguous()
        queries_per_head = queries_per_head.view(
            batch_size * num_heads, timesteps_1,
            int(self._attention_dim/num_heads))

        # Shape (num_heads * batch_size, timesteps_2, attention_dim/num_heads)
        keys_per_head = keys.view(batch_size, timesteps_2, num_heads,
                                  int(self._attention_dim/num_heads))
        keys_per_head = keys_per_head.transpose(1, 2).contiguous()
        keys_per_head = keys_per_head.view(
            batch_size * num_heads, timesteps_2,
            int(self._attention_dim/num_heads))

        # shape (num_heads * batch_size, timesteps_2, timesteps_1)
        scaled_similarities = torch.bmm(
            queries_per_head / self._scale, keys_per_head.transpose(1, 2))

        # shape (num_heads * batch_size, timesteps_2, timesteps_1)
        # Normalise the distributions, using the same mask for all heads.
        attention = masked_softmax(scaled_similarities,
                                   mask.repeat(1, num_heads).view(
                                       batch_size * num_heads, timesteps_2),
                                   memory_efficient=True)
        attention = self._attention_dropout(attention)

        # Take a weighted sum of the values with respect to the attention
        # distributions for each element in the num_heads * batch_size dim.
        # shape (num_heads * batch_size, timesteps_2, values_dim/num_heads)
        outputs = weighted_sum(values_per_head, attention)

        # Reshape back to original shape (batch_size, timesteps_1, values_dim)
        # shape (batch_size, num_heads, timesteps_1, values_dim/num_heads)
        outputs = outputs.view(batch_size, num_heads,
                               timesteps_1, int(self._values_dim / num_heads))
        # shape (batch_size, timesteps_1, num_heads, values_dim/num_heads)
        outputs = outputs.transpose(1, 2).contiguous()
        # shape (batch_size, timesteps_1, values_dim)
        outputs = outputs.view(batch_size, timesteps_1, self._values_dim)

        # Project back to original input size.
        # shape (batch_size, timesteps_1, input_size)
        outputs = self._output_projection(outputs)
        return outputs


class HeterogenousSequenceAttention(Seq2SeqEncoder):
    """
    Implements sequence attention as defined in Yuanfudao

        alpha = softmax(f(Wu)f(W'v))    (1)

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
                 u_input_dim: int,
                 v_input_dim: int,
                 projection_dim: int,
                 activation: Optional[Activation] = None) -> None:
        super(HeterogenousSequenceAttention, self).__init__()
        self._output_dim = projection_dim
        self._u_input_dim = u_input_dim
        self._v_input_dim = v_input_dim
        self._u_projection = torch.nn.Linear(in_features=u_input_dim,
                                             out_features=projection_dim)
        self._v_projection = torch.nn.Linear(in_features=v_input_dim,
                                             out_features=projection_dim)
        self._activation = activation or Activation.by_name('relu')()

    def get_input_dim(self) -> int:
        return self._u_input_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    @overrides
    def is_bidirectional(self) -> bool:
        return False

    @overrides
    def forward(self, u: torch.Tensor, v: torch.Tensor,
                v_mask: Optional[torch.Tensor]) -> torch.Tensor:
        u_prime = self._activation(self._u_projection(u))
        v_prime = self._activation(self._v_projection(v))

        scores = u_prime.bmm(v_prime.transpose(1, 2))

        alpha = masked_softmax(scores, v_mask, memory_efficient=True)
        result = alpha.bmm(v_prime)

        return result


@Seq2SeqEncoder.register("multi_head_attention_v2")
class MultiHeadAttentionV2(Seq2SeqEncoder):
    # pylint: disable=line-too-long
    """
    This class implements the key-value scaled dot product attention mechanism
    based on the one in the paper `Attention is all you Need`.

    We take two matrices, U and V. U is mapped into three matrices Q_u, K_u and
    V_u. U is mapped into Q_v and K_v. We compute similarities between Q_u and
    K_v, and also Q_v and K_u. We then perform matrix multiplication between
    these similarities. The final output is performed by multiplying this
    matrix with V_u.

    Parameters
    ----------
    num_heads : ``int``, required.
        The number of attention heads to use.
    u_input_dim : ``int``, required.
        The size of the last dimension of the U tensor.
    v_input_dim : ``int``, required.
        The size of the last dimension of the V tensor.
    attention_dim ``int``, required.
        The total dimension of the Q, K and V projections which comprise the
        dot product attention function. Must be divisible by ``num_heads``.
    output_projection_dim : ``int``, optional (default = None)
        The dimensionality of the final output projection. If this is not
        passed explicitly, the projection has size `input_size`.
    attention_dropout_prob : ``float``, optional (default = 0.1).
        The dropout probability applied to the normalised attention
        distributions.
    """

    def __init__(self,
                 num_heads: int,
                 u_input_dim: int,
                 v_input_dim: int,
                 attention_dim: int,
                 output_projection_dim: int = Optional[None],
                 attention_dropout_prob: float = 0.1) -> None:
        super(MultiHeadAttentionV2, self).__init__()

        self._num_heads = num_heads
        self._u_input_dim = u_input_dim
        self._v_input_dim = v_input_dim
        self._output_dim = output_projection_dim or u_input_dim
        self._attention_dim = attention_dim

        if attention_dim % num_heads != 0:
            raise ValueError(f"Attention size ({attention_dim}) must be "
                             f"divisible by the number of attention heads "
                             f"({num_heads}).")

        self._u_q_projection = torch.nn.Linear(u_input_dim, attention_dim)
        self._v_q_projection = torch.nn.Linear(v_input_dim, attention_dim)
        self._u_k_projection = torch.nn.Linear(u_input_dim, attention_dim)
        self._v_k_projection = torch.nn.Linear(v_input_dim, attention_dim)
        self._u_v_projection = torch.nn.Linear(u_input_dim, attention_dim)

        self._scale = (u_input_dim // num_heads) ** 0.5
        self._output_projection = torch.nn.Linear(attention_dim,
                                                  self._output_dim)
        self._attention_dropout = torch.nn.Dropout(attention_dropout_prob)

    def get_input_dim(self) -> int:
        return self._u_input_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    @overrides
    def is_bidirectional(self) -> bool:
        return False

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        "x : tensor of shape (batch_size, timesteps_1, attention_dim)"
        batch_size, timesteps, attention_dim = x.shape
        num_heads = self._num_heads
        head_size = int(attention_dim / num_heads)

        # Shape (num_heads * batch_size, timesteps, attention_dim / num_heads)
        x_per_head = x.view(batch_size, timesteps, num_heads, head_size)
        x_per_head = x_per_head.transpose(1, 2).contiguous()
        x_per_head = x_per_head.view(batch_size * num_heads, timesteps,
                                     head_size)
        return x_per_head

    def _multiply_and_mask(self,
                           q: torch.Tensor,
                           k: torch.Tensor,
                           k_mask: torch.Tensor) -> torch.Tensor:
        first_dim, timesteps_1, head_size = q.shape
        _, timesteps_2, _ = k.shape
        # shape (num_heads * batch_size, timesteps_1, timesteps_2)
        scaled_similarities = torch.bmm(q / self._scale, k.transpose(1, 2))

        # Normalise the distributions, using the same mask for all heads.
        # shape (num_heads * batch_size, timesteps_2)
        k_mask = k_mask.repeat(1, self._num_heads).view(first_dim, timesteps_2)
        k_mask = k_mask.unsqueeze(1).byte()
        masked = scaled_similarities.masked_fill((1 - k_mask), 1e-32)

        return masked

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                u: torch.Tensor,
                v: torch.Tensor,
                u_mask: torch.LongTensor = None,
                v_mask: torch.LongTensor = None) -> torch.FloatTensor:
        """
        Parameters
        ----------
        u : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps_1, input_dim)
        v : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps_2, input_dim)
        u_mask : ``torch.FloatTensor``, optional (default = None).
            A tensor of shape (batch_size, timesteps_1).
        v_mask : ``torch.FloatTensor``, optional (default = None).
            A tensor of shape (batch_size, timesteps_2).
        Returns
        -------
        A tensor of shape (batch_size, timesteps_1, output_projection_dim),
        where output_projection_dim = input_dim by default.
        """
        num_heads = self._num_heads
        head_size = int(self._attention_dim / num_heads)

        _, timesteps_1, _ = u.shape
        batch_size, timesteps_2, _ = v.shape

        if u_mask is None:
            u_mask = u.new_ones(batch_size, timesteps_1)
        if v_mask is None:
            v_mask = v.new_ones(batch_size, timesteps_2)

        # Shape (batch_size, timesteps_1, attention_dim)
        q_u = self._u_q_projection(u)
        k_u = self._u_k_projection(u)
        # Shape (batch_size, timesteps_2, attention_dim)
        q_v = self._v_q_projection(v)
        k_v = self._v_k_projection(v)
        # Shape (batch_size, timesteps_1, values_dim)
        v_u = self._u_v_projection(u)

        # v_u_per_head =v_u.view(batch_size, timesteps_1, num_heads, head_size)
        # v_u_per_head = v_u_per_head.transpose(1, 2).contiguous()
        # v_u_per_head = v_u_per_head.view(batch_size * num_heads, timesteps_1,
        #                                  head_size)

        # Shape (num_heads * batch_size, timesteps_1, values_dim / num_heads)
        q_u_per_head = self._reshape_heads(q_u)
        # Shape (num_heads * batch_size, timesteps_1, values_dim / num_heads)
        k_u_per_head = self._reshape_heads(k_u)
        # Shape (num_heads * batch_size, timesteps_1, values_dim / num_heads)
        v_u_per_head = self._reshape_heads(v_u)

        # Shape (num_heads * batch_size, timesteps_2, values_dim / num_heads)
        q_v_per_head = self._reshape_heads(q_v)
        # Shape (num_heads * batch_size, timesteps_2, values_dim / num_heads)
        k_v_per_head = self._reshape_heads(k_v)

        # Normalise the distributions, using the same mask for all heads.
        # shape (num_heads * batch_size, timesteps_1, timesteps_2)
        sim_1 = self._multiply_and_mask(q_u_per_head, k_v_per_head, v_mask)
        # shape (num_heads * batch_size, timesteps_2, timesteps_1)
        sim_2 = self._multiply_and_mask(q_v_per_head, k_u_per_head, u_mask)

        combined_similarities = sim_1.bmm(sim_2)
        attention = torch.softmax(combined_similarities, dim=-1)
        outputs = attention.bmm(v_u_per_head)

        # Reshape back to original shape (batch_size, timesteps_1, values_dim)
        # shape (batch_size, num_heads, timesteps_1, values_dim/num_heads)
        outputs = outputs.view(batch_size, num_heads, timesteps_1, head_size)
        # shape (batch_size, timesteps_1, num_heads, values_dim/num_heads)
        outputs = outputs.transpose(1, 2).contiguous()
        # shape (batch_size, timesteps_1, values_dim)
        outputs = outputs.view(batch_size, timesteps_1, self._attention_dim)

        # Project back to original input size.
        # shape (batch_size, timesteps_1, input_size)
        outputs = self._output_projection(outputs)
        return outputs
