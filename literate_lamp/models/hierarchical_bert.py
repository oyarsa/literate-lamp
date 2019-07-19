"Model that builds a representation from the BERT windows."
from typing import Dict, Optional
from pathlib import Path

import torch
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn import util

from models.base_model import BaseModel
from models.util import seq_over_seq
from layers import bert_embeddings


class HierarchicalBert(BaseModel):
    """
    Uses hierarchical RNNs to build a sentence representation (from the BERT
    windows) and then build a document representation from those sentences.
    """

    def __init__(self,
                 bert_path: Path,
                 sentence_encoder: Seq2VecEncoder,
                 document_encoder: Seq2VecEncoder,
                 vocab: Vocabulary,
                 encoder_dropout: float = 0.0,
                 train_bert: bool = False
                 ) -> None:
        # We have to pass the vocabulary to the constructor.
        super().__init__(vocab)
        self.word_embeddings = bert_embeddings(pretrained_model=bert_path,
                                               training=train_bert)

        if encoder_dropout > 0:
            self.encoder_dropout = torch.nn.Dropout(p=encoder_dropout)
        else:
            self.encoder_dropout = lambda x: x

        self.sentence_encoder = sentence_encoder
        self.document_encoder = document_encoder
        self.dense = torch.nn.Linear(
            in_features=document_encoder.get_output_dim(),
            out_features=1
        )

    # This is the computation bit of the model. The arguments of this function
    # are the fields from the `Instance` we created, as that's what's going to
    # be passed to this. We also have the optional `label`, which is only
    # available at training time, used to calculate the loss.
    def forward(self,
                passage_id: Dict[str, torch.Tensor],
                question_id: Dict[str, torch.Tensor],
                bert0: Dict[str, torch.Tensor],
                bert1: Dict[str, torch.Tensor],
                passage: Dict[str, torch.Tensor],
                question: Dict[str, torch.Tensor],
                answer0: Dict[str, torch.Tensor],
                answer1: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        # Every sample in a batch has to have the same size (as it's a tensor),
        # so smaller entries are padded. The mask is used to counteract this
        # padding.
        t0_masks = util.get_text_field_mask(bert0)
        t1_masks = util.get_text_field_mask(bert1)

        # We create the embeddings from the input text
        t0_embs = self.word_embeddings(bert0)
        t1_embs = self.word_embeddings(bert1)

        t0_sentence_encodings = self.encoder_dropout(
            seq_over_seq(self.sentence_encoder, t0_embs,
                         t0_masks))
        t1_sentence_encodings = self.encoder_dropout(
            seq_over_seq(self.sentence_encoder, t1_embs,
                         t1_masks))

        t0_enc_out = self.encoder_dropout(
            self.document_encoder(t0_sentence_encodings, mask=None))
        t1_enc_out = self.encoder_dropout(
            self.document_encoder(t1_sentence_encodings, mask=None))

        logit0 = self.dense(t0_enc_out).squeeze(-1)
        logit1 = self.dense(t1_enc_out).squeeze(-1)

        logits = torch.stack((logit0, logit1), dim=-1)
        # We also compute the class with highest likelihood (our prediction)
        prob = torch.softmax(logits, dim=-1)
        output = {"logits": logits, "prob": prob}

        # Labels are optional. If they're present, we calculate the accuracy
        # and the loss function.
        if label is not None:
            self.accuracy(prob, label)
            output["loss"] = self.loss(logits, label)

        # The output is the dict we've been building, with the logits, loss
        # and the prediction.
        return output
