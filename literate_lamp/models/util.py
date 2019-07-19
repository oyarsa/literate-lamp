"Utilities shared by the models."
from typing import Optional

import torch
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.attention import Attention


def seq_over_seq(encoder: Seq2VecEncoder,
                 sentences: torch.Tensor,
                 masks: Optional[torch.Tensor] = None
                 ) -> torch.Tensor:
    num_batch, num_sent, _, _ = sentences.shape
    enc_size = encoder.get_output_dim()
    sentence_encodings = sentences.new_empty(num_batch, num_sent, enc_size)

    for i in range(num_sent):
        sentence_emb = sentences[:, i, :, :]
        if masks is not None:
            sentence_mask = masks[:, i, :]
        else:
            sentence_mask = None
        sentence_encodings[:, i, :] = encoder(sentence_emb, sentence_mask)

    return sentence_encodings


def hierarchical_seq_over_seq(encoder: Seq2SeqEncoder,
                              sentences: torch.Tensor,
                              masks: torch.Tensor) -> torch.Tensor:
    num_batch, num_sent, num_tokens = masks.shape
    enc_size = encoder.get_output_dim()
    sentence_hiddens = sentences.new_empty(
        num_batch, num_sent, num_tokens, enc_size)

    for i in range(num_sent):
        sentence_emb = sentences[:, i, :, :]
        sentence_mask = masks[:, i, :]
        sentence_hiddens[:, i, :, :] = encoder(sentence_emb, sentence_mask)

    return sentence_hiddens


def attention_over_sequence(attention: Attention, sequence: torch.Tensor,
                            vector: torch.Tensor) -> torch.Tensor:
    num_batch, num_sent, num_tokens, _ = sequence.shape
    scores = sequence.new_empty(num_batch, num_sent, num_tokens)

    for i in range(num_sent):
        sequence_item = sequence[:, i, :, :]
        scores[:, i, :] = attention(vector, sequence_item)

    return scores
