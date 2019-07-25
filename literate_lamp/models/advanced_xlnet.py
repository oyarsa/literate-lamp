"Just the standard basic XLNet for classification."
from typing import Dict, Optional

import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask

from models import BaseModel
from layers import LinearAttention


class AdvancedXLNetClassifier(BaseModel):
    """
    Model that encodes input using BERT, takes the embedding for the CLS
    token (using BertPooler) and puts the output through a FFN to get the
    probabilities.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 encoder_dropout: float = 0.5,
                 train_xlnet: bool = False
                 ) -> None:
        # We have to pass the vocabulary to the constructor.
        super().__init__(vocab)
        self.word_embeddings = word_embeddings

        self.encoder = encoder
        self.attn = LinearAttention(self.encoder.get_output_dim())
        self.dropout = torch.nn.Dropout(p=encoder_dropout)

        self.output = torch.nn.Linear(
            in_features=self.attn.get_output_dim(),
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
        t0_masks = get_text_field_mask(string0)
        t1_masks = get_text_field_mask(string1)

        # We create the embeddings from the input text
        t0_embs = self.word_embeddings(string0)
        t1_embs = self.word_embeddings(string1)

        # Then we use those embeddings (along with the masks) as inputs for
        # our encoders
        enc0_hiddens = self.dropout(self.encoder(t0_embs, t0_masks))
        enc1_hiddens = self.dropout(self.encoder(t1_embs, t1_masks))

        enc0_outs = self.attn(enc0_hiddens)
        enc1_outs = self.attn(enc1_hiddens)

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
