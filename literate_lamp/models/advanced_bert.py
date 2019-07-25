"Implements the AdvancedBertClassifier class."
from typing import Dict, Optional
from pathlib import Path

import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2vec_encoders import BertPooler, Seq2VecEncoder

from models.base_model import BaseModel
from layers import bert_embeddings


class AdvancedBertClassifier(BaseModel):
    """
    Model similar to the AttentiveClassifier with BERT, but without external
    features.

    SimpleTrian is this with the attention before the encoders.
    """

    def __init__(self,
                 bert_path: Path,
                 encoder: Seq2VecEncoder,
                 vocab: Vocabulary,
                 hidden_dim: int = 100,
                 encoder_dropout: float = 0.0,
                 train_bert: bool = False
                 ) -> None:
        # We have to pass the vocabulary to the constructor.
        super().__init__(vocab)
        self.word_embeddings = bert_embeddings(pretrained_model=bert_path,
                                               training=train_bert)

        self.encoder_dropout: torch.nn.Module
        if encoder_dropout > 0:
            self.encoder_dropout = torch.nn.Dropout(p=encoder_dropout)
        else:
            self.encoder_dropout = torch.nn.Identity()

        self.pooler = BertPooler(pretrained_model=str(bert_path))
        self.dense1 = torch.nn.Linear(
            in_features=self.pooler.get_output_dim(),
            out_features=hidden_dim
        )
        self.encoder = encoder
        self.dense2 = torch.nn.Linear(
            in_features=encoder.get_output_dim(),
            out_features=1
        )

    # This is the computation bit of the model. The arguments of this function
    # are the fields from the `Instance` we created, as that's what's going to
    # be passed to this. We also have the optional `label`, which is only
    # available at training time, used to calculate the loss.
    def forward(self,
                metadata: Dict[str, torch.Tensor],
                bert0: Dict[str, torch.Tensor],
                bert1: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        # Every sample in a batch has to have the same size (as it's a tensor),
        # so smaller entries are padded. The mask is used to counteract this
        # padding.

        # We create the embeddings from the input text
        t0_embs = self.word_embeddings(bert0)
        t1_embs = self.word_embeddings(bert1)

        t0_pooled = self.pooler(t0_embs)
        t1_pooled = self.pooler(t1_embs)

        t0_transformed = self.dense1(t0_pooled)
        t1_transformed = self.dense1(t1_pooled)

        t0_enc_out = self.encoder_dropout(
            self.encoder(t0_transformed, mask=None))
        t1_enc_out = self.encoder_dropout(
            self.encoder(t1_transformed, mask=None))

        logit0 = self.dense2(t0_enc_out).squeeze(-1)
        logit1 = self.dense2(t1_enc_out).squeeze(-1)

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
