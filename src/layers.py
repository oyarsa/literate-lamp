"""
Implements some layers that AllenNLP doesn't have.
"""
from typing import Optional, Tuple, Union
from pathlib import Path

import torch
from overrides import overrides
from allennlp.common import Params
from allennlp.modules.attention import Attention
from allennlp.nn.util import (masked_softmax, weighted_sum,
                              add_positional_features)
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
from allennlp.modules.seq2vec_encoders import (PytorchSeq2VecWrapper,
                                               CnnEncoder,
                                               Seq2VecEncoder)
from allennlp.modules.seq2seq_encoders import (Seq2SeqEncoder,
                                               StackedSelfAttentionEncoder,
                                               PytorchSeq2SeqWrapper)
from allennlp.modules.layer_norm import LayerNorm

from util import clone_module


class BilinearMatrixAttention(Attention):
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
                 matrix1_dim: int,
                 matrix2_dim: int,
                 normalise: bool = True) -> None:
        super().__init__(normalise)
        self._weights = torch.nn.Linear(in_features=matrix2_dim,
                                        out_features=matrix1_dim)

    @overrides
    def _forward_internal(self, matrix1: torch.Tensor, matrix2: torch.Tensor
                          ) -> torch.Tensor:
        """
        Args:
            matrix1 : Tensor of shape (batch_size, seq_len1, hdim1)
            matrix2 : Tensor of shape (batch_size, seq_len2, hdim2)
        Output:
            alpha : Tensor of shape (batch_size, seq_len1, seq_len2)
        """
        # Shape : (batch_size, seq_len_2, hdim1)
        Wy = self._weights(matrix2)
        # Shape : (batch_size, seq_len_1, seq_len_2)
        alpha = matrix1.bmm(Wy.transpose(-2, -1))
        return alpha


class LinearAttention(Seq2VecEncoder):
    def __init__(self, input_dim: int, bias: bool = False) -> None:
        super(LinearAttention, self).__init__()
        self.input_dim = input_dim
        self.linear = torch.nn.Linear(in_features=input_dim,
                                      out_features=1,
                                      bias=bias)

    def forward(self,
                inputs: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Arguments:
            inputs : batch_size * seq_len * dim
            mask   : batch_size * seq_len
                (1 for padding, 0 for true input)
        Output:
            attended : batch * dim
        """
        # Applying the linear gives as a dim * 1 vector per batch.
        # Remove the last dimension as it's not useful.
        scores = self.linear(inputs).squeeze(-1)

        if mask is not None:
            # Mask the padded input with a very low number before softmax.
            scores = scores.masked_fill((1 - mask), 1e-32)
        weights = torch.softmax(scores, dim=-1)

        # To get a 1 * seq_len vector per batch
        weights = weights.unsqueeze(1)
        # Weighted average of the sequence
        output = weights.bmm(inputs)
        # Now we have a 1 * dim vector per batch. Remove the useless dimension.
        output = output.squeeze(1)

        return output

    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.input_dim


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
                        model_dim: int,
                        feedforward_hidden_dim: int = 2048,
                        num_layers: int = 6,
                        projection_dim: int = 64,
                        num_attention_heads: int = 8,
                        ttype: str = 'custom',
                        dropout: float = 0.1) -> Seq2SeqEncoder:
    if ttype == 'custom':
        return TransformerEncoder(
            input_dim=input_dim,
            model_dim=model_dim,
            feedforward_hidden_dim=feedforward_hidden_dim,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            dropout_prob=dropout
        )
    elif ttype == 'allen':
        return StackedSelfAttentionEncoder(
            input_dim=input_dim,
            hidden_dim=model_dim,
            feedforward_hidden_dim=feedforward_hidden_dim,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            projection_dim=model_dim,
            dropout_prob=dropout
        )
    else:
        raise ValueError(f'Invalid transformer type {ttype}')


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

        assert value_input_dim == key_input_dim
        self._kv_projection = torch.nn.Linear(key_input_dim, 2*attention_dim)

        self._scale = (attention_dim // num_heads) ** 0.5
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
                mask: torch.Tensor = None) -> torch.Tensor:
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
        # Shape (batch_size, timesteps_2, 2*attention_dim)
        keys_values = self._kv_projection(keys)
        # Shape (batch_size, timesteps_2, values_dim)
        keys, values = torch.chunk(keys_values, 2, dim=-1)

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

        self._u_qkv_projection = torch.nn.Linear(u_input_dim, 3*attention_dim)
        self._v_qkv_projection = torch.nn.Linear(v_input_dim, 3*attention_dim)

        self._scale = (attention_dim // num_heads) ** 0.5
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

    def _reshape_outputs(self,
                         outputs: torch.FloatTensor
                         ) -> torch.FloatTensor:
        num_heads = self._num_heads
        attention_dim = self._attention_dim
        first_dim, timesteps, head_size = outputs.shape
        batch_size = int(first_dim / num_heads)

        # Reshape back to original shape (batch_size, timesteps, values_dim)

        # shape (batch_size, num_heads, timesteps, values_dim/num_heads)
        outputs = outputs.view(batch_size, num_heads, timesteps, head_size)
        # shape (batch_size, timesteps, num_heads, values_dim/num_heads)
        outputs = outputs.transpose(1, 2).contiguous()
        # shape (batch_size, timesteps, values_dim)
        outputs = outputs.view(batch_size, timesteps, attention_dim)
        return outputs

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
        first_dim, timesteps_2, head_size = k.shape
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
                v_mask: torch.LongTensor = None
                ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
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
        if u_mask is None:
            u_mask = u.new_ones(u.size(0), u.size(1))
        if v_mask is None:
            v_mask = v.new_ones(v.size(0), v.size(1))

        # Shape (batch_size, timesteps_1, 3 * attention_dim)
        qkv_u = self._u_qkv_projection(u)
        # Shape (batch_size, timesteps_1, attention_dim)
        q_u, k_u, v_u = torch.chunk(qkv_u, 3, dim=-1)

        # Shape (batch_size, timesteps_2, 3 * attention_dim)
        qkv_v = self._v_qkv_projection(v)
        # Shape (batch_size, timesteps_2, attention_dim)
        q_v, k_v, v_v = torch.chunk(qkv_v, 3, dim=-1)

        # Shape (num_heads * batch_size, timesteps_1, values_dim / num_heads)
        q_u_per_head = self._reshape_heads(q_u)
        k_u_per_head = self._reshape_heads(k_u)
        v_u_per_head = self._reshape_heads(v_u)

        # Shape (num_heads * batch_size, timesteps_2, values_dim / num_heads)
        q_v_per_head = self._reshape_heads(q_v)
        k_v_per_head = self._reshape_heads(k_v)
        v_v_per_head = self._reshape_heads(v_v)

        # Normalise the distributions, using the same mask for all heads.
        # shape (num_heads * batch_size, timesteps_1, timesteps_2)
        sim_1 = self._multiply_and_mask(q_u_per_head, k_v_per_head, v_mask)
        # shape (num_heads * batch_size, timesteps_2, timesteps_1)
        sim_2 = self._multiply_and_mask(q_v_per_head, k_u_per_head, u_mask)

        combined_similarities_u = sim_1.bmm(sim_2)
        attention_u = torch.softmax(combined_similarities_u, dim=-1)
        outputs_u = attention_u.bmm(v_u_per_head)

        combined_similarities_v = sim_2.bmm(sim_1)
        attention_v = torch.softmax(combined_similarities_v, dim=-1)
        outputs_v = attention_v.bmm(v_v_per_head)

        outputs_u = self._reshape_outputs(outputs_u)
        outputs_v = self._reshape_outputs(outputs_v)

        # Project back to original input size.
        # shape (batch_size, timesteps_1, input_size)
        outputs_u = self._output_projection(outputs_u)
        outputs_v = self._output_projection(outputs_v)
        return outputs_u, outputs_v


class TransformerEncoderBlock(torch.nn.Module):
    def __init__(self,
                 model_dim: int,
                 attention_dim: int,
                 num_heads: int,
                 feedforward_dim: int,
                 dropout: float = 0.1
                 ) -> None:
        super(TransformerEncoderBlock, self).__init__()

        self.attn = MultiHeadAttention(num_heads=num_heads,
                                       query_input_dim=model_dim,
                                       key_input_dim=model_dim,
                                       value_input_dim=model_dim,
                                       attention_dim=attention_dim,
                                       values_dim=attention_dim,
                                       output_projection_dim=model_dim,
                                       attention_dropout_prob=dropout)
        self.attn_dropout = torch.nn.Dropout(dropout)

        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(model_dim, feedforward_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(feedforward_dim, model_dim),
            torch.nn.Dropout(dropout)
        )

        self.norm1 = LayerNorm(model_dim)
        self.norm2 = LayerNorm(model_dim)

    def forward(self,
                src: torch.Tensor,
                src_mask: torch.Tensor,
                ) -> torch.Tensor:
        attn = self.attn(src, src, src, src_mask)
        attn = self.attn_dropout(attn)

        out1 = self.norm1(src + attn)
        ffn = self.ffn(out1)
        out2 = self.norm2(out1 + ffn)

        return out2


class RelationTransformerEncoderBlock(torch.nn.Module):
    def __init__(self,
                 model_dim: int,
                 attention_dim: int,
                 num_heads: int,
                 feedforward_dim: int,
                 dropout: float = 0.1
                 ) -> None:
        super(RelationTransformerEncoderBlock, self).__init__()

        self.attn = MultiHeadAttentionV2(num_heads=num_heads,
                                         u_input_dim=model_dim,
                                         v_input_dim=model_dim,
                                         attention_dim=attention_dim,
                                         output_projection_dim=model_dim,
                                         attention_dropout_prob=dropout)
        self.attn_dropout = torch.nn.Dropout(dropout)

        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(model_dim, feedforward_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(feedforward_dim, model_dim),
            torch.nn.Dropout(dropout)
        )

        self.norm1 = LayerNorm(model_dim)
        self.norm2 = LayerNorm(model_dim)

    def _second_stage(self,
                      x: torch.Tensor,
                      attn: torch.Tensor
                      ) -> torch.Tensor:
        attn = self.attn_dropout(attn)
        out1 = self.norm1(x + attn)
        ffn = self.ffn(out1)
        out2 = self.norm2(out1 + ffn)
        return out2

    def forward(self,
                src: torch.Tensor,
                aux: torch.Tensor,
                src_mask: torch.Tensor,
                aux_mask: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_src, attn_aux = self.attn(src, aux, src_mask, aux_mask)
        src = self._second_stage(src, attn_src)
        aux = self._second_stage(aux, attn_aux)
        return src, aux


@Seq2SeqEncoder.register("transformer-encoder")
class TransformerEncoder(Seq2SeqEncoder):
    # pylint: disable=line-too-long
    """
    Implements a stacked self-attention encoder similar to the Transformer
    architecture in `Attention is all you Need`.

    This encoder combines 3 layers in a 'block':

    1. A 2 layer FeedForward network.
    2. Multi-headed self attention, which uses 2 learnt linear projections
       to perform a dot-product similarity between every pair of elements
       scaled by the square root of the sequence length.
    3. Layer Normalisation.

    We use the torch.nn.TransformerEncoderLayer block.
    These are then stacked into ``num_layers`` layers.

    Parameters
    ----------
    input_dim : ``int``, required.
        The input dimension of the encoder.
    model_dim : ``int``, required.
        The hidden dimension used for the _input_ to self attention layers
        and the _output_ from the feedforward layers.
    feedforward_hidden_dim : ``int``, required.
        The middle dimension of the FeedForward network. The input and output
        dimensions are fixed to ensure sizes match up for the self attention
        layers.
    num_layers : ``int``, required.
        The number of stacked TransformerEncoder blocks.
    num_attention_heads : ``int``, required.
        The number of attention heads to use per layer.
    dropout_prob : ``float``, optional, (default = 0.1)
        The dropout probability between the layers in the block.
    """

    def __init__(self,
                 input_dim: int,
                 model_dim: int,
                 feedforward_hidden_dim: int,
                 num_layers: int,
                 num_attention_heads: int,
                 dropout_prob: float = 0.1
                 ) -> None:
        super(TransformerEncoder, self).__init__()

        self.input_dim = input_dim
        self.output_dim = model_dim

        self.projection = torch.nn.Linear(input_dim, model_dim)

        encoder_block = TransformerEncoderBlock(
            model_dim=model_dim,
            num_heads=num_attention_heads,
            feedforward_dim=feedforward_hidden_dim,
            attention_dim=model_dim,
            dropout=dropout_prob
        )
        self.blocks = clone_module(encoder_block, num_layers)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim

    @overrides
    def is_bidirectional(self) -> bool:
        return False

    @overrides
    def forward(self,
                inputs: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        output = add_positional_features(inputs)
        output = self.projection(output)

        for block in self.blocks:
            output = block(output, mask)

        return output


@Seq2SeqEncoder.register("rel-transformer-encoder")
class RelationalTransformerEncoder(Seq2SeqEncoder):
    # pylint: disable=line-too-long
    """
    Implements a stacked self-attention encoder similar to the Transformer
    architecture in `Attention is all you Need`.

    This encoder combines 3 layers in a 'block':

    1. A 2 layer FeedForward network.
    2. Multi-headed self attention, which uses 2 learnt linear projections
       to perform a dot-product similarity between every pair of elements
       scaled by the square root of the sequence length.
    3. Layer Normalisation.

    We use the torch.nn.TransformerEncoderLayer block.
    These are then stacked into ``num_layers`` layers.

    Parameters
    ----------
    src_input_dim : ``int``, required.
        The input dimension for the source matrix.
    kb_input_dim : ``int``, required.
        The input dimension for the kb matrix.
    model_dim : ``int``, required.
        The hidden dimension used for the _input_ to self attention layers
        and the _output_ from the feedforward layers.
    feedforward_hidden_dim : ``int``, required.
        The middle dimension of the FeedForward network. The input and output
        dimensions are fixed to ensure sizes match up for the self attention
        layers.
    num_layers : ``int``, required.
        The number of stacked TransformerEncoder blocks.
    num_attention_heads : ``int``, required.
        The number of attention heads to use per layer.
    dropout_prob : ``float``, optional, (default = 0.1)
        The dropout probability between the layers in the block.
    return_kb : ``bool``, optional, (default = False)
        Whether to return the final matrix for the KB as well. By default,
        we only return the SRC matrix. If True, we return a tuple (SRC, KB).
    """

    def __init__(self,
                 src_input_dim: int,
                 kb_input_dim: int,
                 model_dim: int,
                 feedforward_hidden_dim: int,
                 num_layers: int,
                 num_attention_heads: int,
                 dropout_prob: float = 0.1,
                 return_kb: bool = False
                 ) -> None:
        super(RelationalTransformerEncoder, self).__init__()

        self.input_dim = src_input_dim
        self.output_dim = model_dim
        self.num_layers = num_layers
        self.return_kb = return_kb

        self.src_projection = torch.nn.Linear(src_input_dim, model_dim)
        self.kb_projection = torch.nn.Linear(kb_input_dim, model_dim)

        rel_block = RelationTransformerEncoderBlock(
            attention_dim=model_dim,
            model_dim=model_dim,
            num_heads=num_attention_heads,
            feedforward_dim=feedforward_hidden_dim,
            dropout=dropout_prob
        )
        self_block = TransformerEncoderBlock(
            attention_dim=model_dim,
            model_dim=model_dim,
            num_heads=num_attention_heads,
            feedforward_dim=feedforward_hidden_dim,
            dropout=dropout_prob
        )

        self.src_self_blocks = clone_module(self_block, num_layers)
        self.kb_self_blocks = clone_module(self_block, num_layers)

        self.rel_blocks = clone_module(rel_block, num_layers)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim

    @overrides
    def is_bidirectional(self) -> bool:
        return False

    @overrides
    def forward(self,
                src: torch.Tensor,
                kb: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                kb_mask: Optional[torch.Tensor] = None,
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        src = add_positional_features(src)
        src = self.src_projection(src)
        kb = self.kb_projection(kb)

        for i in range(self.num_layers):
            src = self.src_self_blocks[i](src, src_mask)
            kb = self.kb_self_blocks[i](kb, kb_mask)

            src, kb = self.rel_blocks[i](kb, src, kb_mask, src_mask)

        if self.return_kb:
            return src, kb
        return src
