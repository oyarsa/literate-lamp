"Just the standard basic BERT for classification."
from typing import Dict, Optional
from pathlib import Path

import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2vec_encoders import BertPooler
from allennlp.nn import util

from models.base_model import BaseModel
from layers import bert_embeddings


class SimpleBertClassifier(BaseModel):
    """
    Model that encodes input using BERT, takes the embedding for the CLS
    token (using BertPooler) and puts the output through a FFN to get the
    probabilities.
    """

    def __init__(self,
                 bert_path: Path,
                 vocab: Vocabulary,
                 train_bert: bool = False
                 ) -> None:
        # We have to pass the vocabulary to the constructor.
        super().__init__(vocab)
        self.word_embeddings = bert_embeddings(pretrained_model=bert_path,
                                               training=train_bert)

        self.pooler = BertPooler(pretrained_model=str(bert_path))

        hidden_dim = self.encoder.get_output_dim()
        self.hidden2logit = torch.nn.Linear(
            in_features=hidden_dim,
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

        # Then we use those embeddings (along with the masks) as inputs for
        # our encoders
        enc0_outs = self.pooler(t0_embs, t0_masks)
        enc1_outs = self.pooler(t1_embs, t1_masks)

        # Finally, we pass each encoded output tensor to the feedforward layer
        # to produce logits corresponding to each class.
        logit0 = self.hidden2logit(enc0_outs).squeeze(-1)
        logit1 = self.hidden2logit(enc1_outs).squeeze(-1)
        logit0, _ = torch.max(logit0, dim=1)
        logit1, _ = torch.max(logit1, dim=1)
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
