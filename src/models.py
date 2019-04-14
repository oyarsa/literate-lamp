from typing import Dict, Optional

import torch

# Base class for the Model we'll implement. Inherits from `torch.nn.Model`,
# but compatible with what the rest of the AllenNLP library expects.
from allennlp.models import Model

# These create text embeddings from a `TextField` input. Since our text data
# is represented using `TextField`s, this makes sense.
# Again, `TextFieldEmbedder` is the abstract class, `BasicTextFieldEmbedder`
# is the implementation (as we're just using simple embeddings, no fancy
# ELMo or BERT so far).
from allennlp.modules.text_field_embedders import TextFieldEmbedder

# `Seq2VecEncoder` is an abstract encoder that takes a sequence and generates
# a vector. This can be an LSTM (although they can also be Seq2Seq if you
# output the hidden state), a Transformer or anything else really, just taking
# NxM -> 1xQ.
# The `PytorchSeq2VecWrapper` is a wrapper for the PyTorch Seq2Vec encoders
# (such as the LSTM we'll use later on), as they don't exactly follow the
# interface the library expects.
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.attention import LinearAttention, BilinearAttention
from allennlp.training.metrics import CategoricalAccuracy

# Holds the vocabulary, learned from the whole data. Also knows the mapping
# from the `TokenIndexer`, mapping the `Token` to an index in the vocabulary
# and vice-versa.
from allennlp.data.vocabulary import Vocabulary

# Some utilities provided by AllenNLP.
#   - `get_text_field_mask` masks the inputs according to the padding.
#   - `clone` creates N copies of a layer.
from allennlp.nn import util


@Model.register('baseline-classifier')
class BaselineClassifier(Model):
    """
    The `Model` class basically needs a `forward` method to be able to process
    the input. It can do whatever we want, though, as long as `forward` can be
    differentiated.

    We're passing abstract classes as inputs, so that our model can use
    different types of embeddings and encoder. This allows us to replace
    the embedding with ELMo or BERT without changing the model code, for
    example, or replace the LSTM with a GRU or Transformer in the same way.

    Refer to the imports as an explanation to these abstract classes.
    """

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 vocab: Vocabulary) -> None:
        # We have to pass the vocabulary to the constructor.
        super().__init__(vocab)
        self.word_embeddings = word_embeddings

        # Our model has different encoders for each of the fields (passage,
        # answer and question). These could theoretically be different for each
        # field, but for now we're using the same. Hence, we clone the provided
        # encoder.
        self.p_encoder, self.q_encoder, self.a_encoder = util.clone(encoder, 3)

        # We're using a hidden layer to build the output from each encoder.
        # As this can't really change, it's not passed as input.
        # The size has to be the size of concatenating the encoder outputs,
        # since that's how we're combining them in the computation. As they're
        # the same, just multiply the first encoder output by 3.
        # The output of the model (which is the output of this layer) has to
        # have size equal to the number of classes.
        hidden_dim = self.p_encoder.get_output_dim() * 4
        self.hidden2logit = torch.nn.Linear(
            in_features=hidden_dim,
            out_features=vocab.get_vocab_size('label')
        )

        # Categorical (as this is a classification task) accuracy
        self.accuracy = CategoricalAccuracy()
        # CrossEntropyLoss is a combinational of LogSoftmax and
        # Negative Log Likelihood. We won't directly use Softmax in training.
        self.loss = torch.nn.CrossEntropyLoss()

    # This is the computation bit of the model. The arguments of this function
    # are the fields from the `Instance` we created, as that's what's going to
    # be passed to this. We also have the optional `label`, which is only
    # available at training time, used to calculate the loss.
    def forward(self,
                passage_id: Dict[str, torch.Tensor],
                question_id: Dict[str, torch.Tensor],
                passage: Dict[str, torch.Tensor],
                question: Dict[str, torch.Tensor],
                answer0: Dict[str, torch.Tensor],
                answer1: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:

        # Every sample in a batch has to have the same size (as it's a tensor),
        # so smaller entries are padded. The mask is used to counteract this
        # padding.
        p_mask = util.get_text_field_mask(passage)
        q_mask = util.get_text_field_mask(question)
        a0_mask = util.get_text_field_mask(answer0)
        a1_mask = util.get_text_field_mask(answer1)

        # We create the embeddings from the input text
        p_emb = self.word_embeddings(passage)
        q_emb = self.word_embeddings(question)
        a0_emb = self.word_embeddings(answer0)
        a1_emb = self.word_embeddings(answer1)
        # Then we use those embeddings (along with the masks) as inputs for
        # our encoders
        p_enc_out = self.p_encoder(p_emb, p_mask)
        q_enc_out = self.q_encoder(q_emb, q_mask)
        a0_enc_out = self.a_encoder(a0_emb, a0_mask)
        a1_enc_out = self.a_encoder(a1_emb, a1_mask)

        # We then concatenate the representations from each encoder
        encoder_out = torch.cat(
            (p_enc_out, q_enc_out, a0_enc_out, a1_enc_out), 1)
        # Finally, we pass each encoded output tensor to the feedforward layer
        # to produce logits corresponding to each class.
        logits = self.hidden2logit(encoder_out)
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

    # This function computes the metrics we want to see during training.
    # For now, we only have the accuracy metric, but we could have a number
    # of different metrics here.
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


@Model.register('attentive-classifier')
class AttentiveClassifier(Model):
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
                 p_encoder: Seq2SeqEncoder,
                 q_encoder: Seq2SeqEncoder,
                 a_encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        # We have to pass the vocabulary to the constructor.
        super().__init__(vocab)
        self.word_embeddings = word_embeddings

        # Our model has different encoders for each of the fields (passage,
        # answer and question).
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        self.a_encoder = a_encoder

        # Attention layers: passage-question, question-self, answer-self
        self.p_q_attn = BilinearAttention(
            vector_dim=self.q_encoder.get_output_dim(),
            matrix_dim=self.p_encoder.get_output_dim(),
        )
        self.q_self_attn = LinearAttention(
            tensor_1_dim=self.q_encoder.get_output_dim(),
            tensor_2_dim=self.q_encoder.get_output_dim(),
            combination='1'
        )
        self.a_self_attn = LinearAttention(
            tensor_1_dim=self.a_encoder.get_output_dim(),
            tensor_2_dim=self.a_encoder.get_output_dim(),
            combination='1'
        )
        self.p_a_bilinear = torch.nn.Linear(
            in_features=self.p_encoder.get_output_dim(),
            out_features=self.a_encoder.get_output_dim()
        )
        self.q_a_bilinear = torch.nn.Linear(
            in_features=self.q_encoder.get_output_dim(),
            out_features=self.a_encoder.get_output_dim()
        )
        # We're using a hidden layer to build the output from each encoder.
        # As this can't really change, it's not passed as input.
        # The size has to be the size of concatenating the encoder outputs,
        # since that's how we're combining them in the computation. As they're
        # the same, just multiply the first encoder output by 3.
        # The output of the model (which is the output of this layer) has to
        # have size equal to the number of classes.
        # hidden_dim = (self.q_encoder.get_output_dim()  # p_q_attn
        #               + self.q_encoder.get_output_dim()  # q_self_attn
        #               + self.a_encoder.get_output_dim())  # a_self_attn
        # hidden_dim = self.p_a_bilinear.get_output_dim()
        # self.hidden2logit = torch.nn.Linear(
        #     in_features=hidden_dim,
        #     out_features=1
        # )

        # Categorical (as this is a classification task) accuracy
        self.accuracy = CategoricalAccuracy()
        # CrossEntropyLoss is a combinational of LogSoftmax and
        # Negative Log Likelihood. We won't directly use Softmax in training.
        self.loss = torch.nn.CrossEntropyLoss()

    # This is the computation bit of the model. The arguments of this function
    # are the fields from the `Instance` we created, as that's what's going to
    # be passed to this. We also have the optional `label`, which is only
    # available at training time, used to calculate the loss.
    def forward(self,
                passage_id: Dict[str, torch.Tensor],
                question_id: Dict[str, torch.Tensor],
                passage: Dict[str, torch.Tensor],
                question: Dict[str, torch.Tensor],
                answer0: Dict[str, torch.Tensor],
                answer1: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:

        # Every sample in a batch has to have the same size (as it's a tensor),
        # so smaller entries are padded. The mask is used to counteract this
        # padding.
        p_mask = util.get_text_field_mask(passage)
        q_mask = util.get_text_field_mask(question)
        a0_mask = util.get_text_field_mask(answer0)
        a1_mask = util.get_text_field_mask(answer1)

        # We create the embeddings from the input text
        p_emb = self.word_embeddings(passage)
        q_emb = self.word_embeddings(question)
        a0_emb = self.word_embeddings(answer0)
        a1_emb = self.word_embeddings(answer1)
        # Then we use those embeddings (along with the masks) as inputs for
        # our encoders
        p_hiddens = self.p_encoder(p_emb, p_mask)
        q_hiddens = self.q_encoder(q_emb, q_mask)
        a0_hiddens = self.a_encoder(a0_emb, a0_mask)
        a1_hiddens = self.a_encoder(a1_emb, a1_mask)

        # print('Hiddens: p, q, a', p_hiddens.shape,
        #       q_hiddens.shape, q_hiddens.shape)
        # print('Masks: p, q, a', p_mask.shape, q_mask.shape, a_mask.shape)

        q_attn = self.q_self_attn(q_hiddens, q_hiddens, q_mask)
        a0_attn = self.a_self_attn(a0_hiddens, a0_hiddens, a0_mask)
        a1_attn = self.a_self_attn(a1_hiddens, a1_hiddens, a1_mask)

        q_weighted = util.weighted_sum(q_hiddens, q_attn)
        a0_weighted = util.weighted_sum(a0_hiddens, a0_attn)
        a1_weighted = util.weighted_sum(a1_hiddens, a1_attn)

        p_q_attn = self.p_q_attn(q_weighted, p_hiddens, p_mask)
        p_weighted = util.weighted_sum(p_hiddens, p_q_attn)

        out_0 = (self.p_a_bilinear(p_weighted) * a0_weighted).sum(dim=1)
        out_0 += (self.q_a_bilinear(q_weighted) * a0_weighted).sum(dim=1)

        out_1 = (self.p_a_bilinear(p_weighted) * a1_weighted).sum(dim=1)
        out_1 += (self.q_a_bilinear(q_weighted) * a1_weighted).sum(dim=1)

        # print('After: pq, q, a', p_q_attn.shape, q_attn.shape, a_attn.shape)
        # print('Weighted: p, q, a', p_weighted.shape, q_weighted.shape,
        #       a_weighted.shape)

        # We then concatenate the representations from each encoder
        # encoder_out = torch.cat((p_weighted, q_weighted, a_weighted), 1)
        # print('Out:', encoder_out.shape)

        # Finally, we pass each encoded output tensor to the feedforward layer
        # to produce logits corresponding to each class.
        # logits = self.hidden2logit(encoder_out)
        logits = torch.stack((out_0, out_1), dim=1)
        # print('Logits:', logits.shape)
        # # We also compute the class with highest likelihood (our prediction)
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

    # This function computes the metrics we want to see during training.
    # For now, we only have the accuracy metric, but we could have a number
    # of different metrics here.
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


@Model.register('attentive-reader')
class AttentiveReader(Model):
    """
    Refer to `BaselineClassifier` for details on how this works.

    This is an implementation of the Attentive Reader, the baselien for the
    SemEval-2018 Task 11.

    It uses a Seq2Seq to encode the passage, and Seq2Vec to encode the question
    and answer. Then it weighs the passage states according to question
    bilinear attention.

    The prediction is another bilinear multiplication of the weighted passage
    and the answer, fed into a sigmoid.

    NOTE: The original had both answers on the input, and the output would
    be a vector of size 2, and the output would be the probability of each
    being right. I can see the theoretical difference, but I'm keeping it
    this way for now, because this is also what other models (TriAN) use.
    """

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 p_encoder: Seq2SeqEncoder,
                 q_encoder: Seq2VecEncoder,
                 a_encoder: Seq2VecEncoder,
                 vocab: Vocabulary) -> None:
        # We have to pass the vocabulary to the constructor.
        super().__init__(vocab)
        self.word_embeddings = word_embeddings

        # Our model has different encoders for each of the fields (passage,
        # answer and question).
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        self.a_encoder = a_encoder

        # Attention layers: passage-question, passage-answer
        self.p_q_attn = BilinearAttention(
            vector_dim=self.q_encoder.get_output_dim(),
            matrix_dim=self.p_encoder.get_output_dim(),
        )
        self.p_a_bilinear = torch.nn.Linear(
            in_features=self.p_encoder.get_output_dim(),
            out_features=self.a_encoder.get_output_dim()
        )

        # Categorical (as this is a classification task) accuracy
        self.accuracy = CategoricalAccuracy()
        # CrossEntropyLoss is a combinational of LogSoftmax and
        # Negative Log Likelihood. We won't directly use Softmax in training.
        self.loss = torch.nn.CrossEntropyLoss()

    # This is the computation bit of the model. The arguments of this function
    # are the fields from the `Instance` we created, as that's what's going to
    # be passed to this. We also have the optional `label`, which is only
    # available at training time, used to calculate the loss.
    def forward(self,
                passage_id: Dict[str, torch.Tensor],
                question_id: Dict[str, torch.Tensor],
                passage: Dict[str, torch.Tensor],
                question: Dict[str, torch.Tensor],
                answer0: Dict[str, torch.Tensor],
                answer1: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:

        # Every sample in a batch has to have the same size (as it's a tensor),
        # so smaller entries are padded. The mask is used to counteract this
        # padding.
        p_mask = util.get_text_field_mask(passage)
        q_mask = util.get_text_field_mask(question)
        a0_mask = util.get_text_field_mask(answer0)
        a1_mask = util.get_text_field_mask(answer1)

        # We create the embeddings from the input text
        p_emb = self.word_embeddings(passage)
        q_emb = self.word_embeddings(question)
        a0_emb = self.word_embeddings(answer0)
        a1_emb = self.word_embeddings(answer1)
        # Then we use those embeddings (along with the masks) as inputs for
        # our encoders
        p_hiddens = self.p_encoder(p_emb, p_mask)
        q_hidden = self.q_encoder(q_emb, q_mask)
        a0_hidden = self.a_encoder(a0_emb, a0_mask)
        a1_hidden = self.a_encoder(a1_emb, a1_mask)

        # print('Hiddens: p, q, a', p_hiddens.shape,
        #       q_hiddens.shape, q_hiddens.shape)
        # print('Masks: p, q, a', p_mask.shape, q_mask.shape, a_mask.shape)

        p_q_attn = self.p_q_attn(q_hidden, p_hiddens, p_mask)
        p_weighted = util.weighted_sum(p_hiddens, p_q_attn)

        out0 = (self.p_a_bilinear(p_weighted) * a0_hidden).sum(dim=1)
        out1 = (self.p_a_bilinear(p_weighted) * a1_hidden).sum(dim=1)

        # print('After: pq, q, a', p_q_attn.shape, q_attn.shape, a_attn.shape)
        # print('Weighted: p, q, a', p_weighted.shape, q_weighted.shape,
        #       a_weighted.shape)

        # We then concatenate the representations from each encoder
        # encoder_out = torch.cat((p_weighted, q_weighted, a_weighted), 1)
        # print('Out:', encoder_out.shape)

        # Finally, we pass each encoded output tensor to the feedforward layer
        # to produce logits corresponding to each class.
        # logits = self.hidden2logit(encoder_out)
        logits = torch.stack((out0, out1), dim=1)
        # We also compute the class with highest likelihood (our prediction)
        prob = torch.softmax(logits, dim=-1)
        output = {"prob": prob}

        # Labels are optional. If they're present, we calculate the accuracy
        # and the loss function.
        if label is not None:
            self.accuracy(prob, label)
            output["loss"] = self.loss(prob, label)

        # The output is the dict we've been building, with the logits, loss
        # and the prediction.
        return output

    # This function computes the metrics we want to see during training.
    # For now, we only have the accuracy metric, but we could have a number
    # of different metrics here.
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
