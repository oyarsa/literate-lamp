"""
Implementation of a Dynamic Memory Network.
"""
from typing import Dict, Optional

import torch
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.data.vocabulary import Vocabulary

from models.base_model import BaseModel

from models.dmn.input_module import InputModule
from models.dmn.answer_module import AnswerModule
from models.dmn.question_module import QuestionModule
from models.dmn.output_module import OutputModule
from models.dmn.memory_module import MemoryModule


class Dmn(BaseModel):
    """
    Refer to `BaselineClassifier` for details on how this works.

    This one is based on that, but adds attention layers after RNNs.
    These are:
        - self-attention (LinearAttention, ignoring the second tensor)
        for question and answer
        - a BilinearAttention for attenting a question to a passage

    This means that our RNNs are Seq2Seq now, returning the entire hidden
    state as matrices. For the passage-to-question e need a vector for the
    question state, so we use the hidden state matrix weighted by the
    attention result.

    We use the attention vectors as weights for the RNN hidden states.

    After attending the RNN states, we concat them and feed into a FFN.
    """

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 sentence_encoder: Seq2VecEncoder,
                 document_encoder: Seq2VecEncoder,
                 question_encoder: Seq2VecEncoder,
                 answer_encoder: Seq2VecEncoder,
                 passes: int,
                 vocab: Vocabulary,
                 dropout: float = 0.0) -> None:
        # We have to pass the vocabulary to the constructor.
        super().__init__(vocab)

        self.hidden_dim = sentence_encoder.get_output_dim()
        self.passes = passes

        self.input_module = InputModule(
            word_embeddings=word_embeddings,
            sentence_encoder=sentence_encoder,
            document_encoder=document_encoder,
            dropout=dropout
        )
        self.question_module = QuestionModule(
            word_embeddings=word_embeddings,
            encoder=question_encoder
        )
        self.answer_module = AnswerModule(
            word_embeddings=word_embeddings,
            encoder=answer_encoder
        )

        self.memory_module = MemoryModule(self.hidden_dim)
        self.output_module = OutputModule(self.hidden_dim, self.hidden_dim, 1)

    def forward(self,
                metadata: Dict[str, torch.Tensor],
                sentences: Dict[str, torch.Tensor],
                question: Dict[str, torch.Tensor],
                answer0: Dict[str, torch.Tensor],
                answer1: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        facts_encoded = self.input_module(sentences)
        question_encoded = self.question_module(question)
        answer0_encoded = self.answer_module(answer0)
        answer1_encoded = self.answer_module(answer1)
        memory = question_encoded

        for _ in range(self.passes):
            memory = self.memory_module(facts_encoded, question_encoded,
                                        memory)

        out_0 = self.output_module(memory, answer0_encoded)
        out_1 = self.output_module(memory, answer1_encoded)

        logits = torch.stack((out_0, out_1), dim=1)
        # # We softmax to turn those logits into probabilities
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
