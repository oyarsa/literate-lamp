"Just the standard basic XLNet for classification."
from typing import Dict, Optional
from pathlib import Path

import torch
from allennlp.data.vocabulary import Vocabulary

from modules import XLNetPooler
from models import BaseModel
from layers import xlnet_embeddings


class SimpleXLNetClassifier(BaseModel):
    """
    Model that encodes input using BERT, takes the embedding for the CLS
    token (using BertPooler) and puts the output through a FFN to get the
    probabilities.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 config_path: Path,
                 model_path: Path,
                 train_xlnet: bool = False
                 ) -> None:
        # We have to pass the vocabulary to the constructor.
        super().__init__(vocab)
        self.word_embeddings = xlnet_embeddings(
            config_path=config_path,
            model_path=model_path,
            training=train_xlnet
        )

        self.pooler = XLNetPooler(self.word_embeddings.get_output_dim())

        self.output = torch.nn.Linear(
            in_features=self.pooler.get_output_dim(),
            out_features=1
        )

    # This is the computation bit of the model. The arguments of this function
    # are the fields from the `Instance` we created, as that's what's going to
    # be passed to this. We also have the optional `label`, which is only
    # available at training time, used to calculate the loss.
    def forward(self,
                metadata: Dict[str, torch.Tensor],
                string0: Dict[str, torch.Tensor],
                string1: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        # Class token indexes for the pooler
        cls_indexes0 = string0['cls-index']
        cls_indexes1 = string1['cls-index']

        # We create the embeddings from the input text
        t0_embs = self.word_embeddings(string0)
        t1_embs = self.word_embeddings(string1)

        # Then we use those embeddings (along with the masks) as inputs for
        # our encoders
        enc0_outs = self.pooler(t0_embs, cls_indexes0)
        enc1_outs = self.pooler(t1_embs, cls_indexes1)

        logit0 = self.output(enc0_outs).squeeze(-1)
        logit1 = self.output(enc1_outs).squeeze(-1)

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
