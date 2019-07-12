"Deep Learning models for Commonsense Question Answering"
from typing import Dict, Optional
from pathlib import Path

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
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, BertPooler
from allennlp.training.metrics import CategoricalAccuracy

# Holds the vocabulary, learned from the whole data. Also knows the mapping
# from the `TokenIndexer`, mapping the `Token` to an index in the vocabulary
# and vice-versa.
from allennlp.data.vocabulary import Vocabulary

# Some utilities provided by AllenNLP.
#   - `get_text_field_mask` masks the inputs according to the padding.
#   - `clone` creates N copies of a layer.
from allennlp.nn import util

from layers import (SequenceAttention, BilinearAttention, LinearSelfAttention,
                    bert_embeddings)


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
                 pos_embeddings: TextFieldEmbedder,
                 ner_embeddings: TextFieldEmbedder,
                 rel_embeddings: TextFieldEmbedder,
                 p_encoder: Seq2SeqEncoder,
                 q_encoder: Seq2SeqEncoder,
                 a_encoder: Seq2SeqEncoder,
                 vocab: Vocabulary,
                 embedding_dropout: float = 0.0,
                 encoder_dropout: float = 0.0) -> None:
        # We have to pass the vocabulary to the constructor.
        super().__init__(vocab)

        self.word_embeddings = word_embeddings
        self.pos_embeddings = pos_embeddings
        self.ner_embeddings = ner_embeddings
        self.rel_embeddings = rel_embeddings

        if embedding_dropout > 0:
            self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)
        else:
            self.embedding_dropout = lambda x: x

        if encoder_dropout > 0:
            self.encoder_dropout = torch.nn.Dropout(p=encoder_dropout)
        else:
            self.encoder_dropout = lambda x: x

        embedding_dim = word_embeddings.get_output_dim()
        self.p_q_match = SequenceAttention(input_dim=embedding_dim)
        self.a_p_match = SequenceAttention(input_dim=embedding_dim)
        self.a_q_match = SequenceAttention(input_dim=embedding_dim)

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
        self.q_self_attn = LinearSelfAttention(
            input_dim=self.q_encoder.get_output_dim()
        )
        self.a_self_attn = LinearSelfAttention(
            input_dim=self.a_encoder.get_output_dim()
        )
        self.p_a_bilinear = torch.nn.Linear(
            in_features=self.p_encoder.get_output_dim(),
            out_features=self.a_encoder.get_output_dim()
        )
        self.q_a_bilinear = torch.nn.Linear(
            in_features=self.q_encoder.get_output_dim(),
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
                passage_pos: Dict[str, torch.Tensor],
                passage_ner: Dict[str, torch.Tensor],
                question_pos: Dict[str, torch.Tensor],
                p_q_rel: Dict[str, torch.Tensor],
                p_a0_rel: Dict[str, torch.Tensor],
                p_a1_rel: Dict[str, torch.Tensor],
                hc_feat: Dict[str, torch.Tensor],
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
        p_emb = self.embedding_dropout(self.word_embeddings(passage))
        q_emb = self.embedding_dropout(self.word_embeddings(question))
        a0_emb = self.embedding_dropout(self.word_embeddings(answer0))
        a1_emb = self.embedding_dropout(self.word_embeddings(answer1))
        # And the POS tags
        p_pos_emb = self.embedding_dropout(self.pos_embeddings(passage_pos))
        q_pos_emb = self.embedding_dropout(self.pos_embeddings(question_pos))
        # And the NER
        p_ner_emb = self.embedding_dropout(self.ner_embeddings(passage_ner))
        # And the relations
        p_q_rel_emb = self.embedding_dropout(self.rel_embeddings(p_q_rel))
        p_a0_rel_emb = self.embedding_dropout(self.rel_embeddings(p_a0_rel))
        p_a1_rel_emb = self.embedding_dropout(self.rel_embeddings(p_a1_rel))

        # We compute the Sequence Attention
        # First the scores
        p_q_scores = self.p_q_match(p_emb, q_emb, q_mask)
        a0_q_scores = self.a_q_match(a0_emb, q_emb, q_mask)
        a1_q_scores = self.a_q_match(a1_emb, q_emb, q_mask)
        a0_p_scores = self.a_p_match(a0_emb, p_emb, p_mask)
        a1_p_scores = self.a_p_match(a1_emb, p_emb, p_mask)

        # Then the weighted inputs
        p_q_match = util.weighted_sum(q_emb, p_q_scores)
        a0_q_match = util.weighted_sum(q_emb, a0_q_scores)
        a1_q_match = util.weighted_sum(q_emb, a1_q_scores)
        a0_p_match = util.weighted_sum(p_emb, a0_p_scores)
        a1_p_match = util.weighted_sum(p_emb, a1_p_scores)

        # We combine the inputs to our encoder
        p_input = torch.cat((p_emb, p_q_match, p_pos_emb, p_ner_emb, hc_feat,
                             p_q_rel_emb, p_a0_rel_emb, p_a1_rel_emb), dim=2)
        a0_input = torch.cat((a0_emb, a0_p_match, a0_q_match), dim=2)
        a1_input = torch.cat((a1_emb, a1_p_match, a1_q_match), dim=2)
        q_input = torch.cat((q_emb, q_pos_emb), dim=2)

        # Then we use those (along with the masks) as inputs for
        # our encoders
        p_hiddens = self.encoder_dropout(self.p_encoder(p_input, p_mask))
        q_hiddens = self.encoder_dropout(self.q_encoder(q_input, q_mask))
        a0_hiddens = self.encoder_dropout(self.a_encoder(a0_input, a0_mask))
        a1_hiddens = self.encoder_dropout(self.a_encoder(a1_input, a1_mask))

        # We compute the self-attention scores
        q_attn = self.q_self_attn(q_hiddens, q_hiddens, q_mask)
        a0_attn = self.a_self_attn(a0_hiddens, a0_hiddens, a0_mask)
        a1_attn = self.a_self_attn(a1_hiddens, a1_hiddens, a1_mask)

        # Then we weight the hidden-states with those scores
        q_weighted = util.weighted_sum(q_hiddens, q_attn)
        a0_weighted = util.weighted_sum(a0_hiddens, a0_attn)
        a1_weighted = util.weighted_sum(a1_hiddens, a1_attn)

        # We weight the text states with a passage-question bilinear attention
        p_q_attn = self.p_q_attn(q_weighted, p_hiddens, p_mask)
        p_weighted = util.weighted_sum(p_hiddens, p_q_attn)

        # Calculate the outputs for each answer, from passage-answer and
        # question-answer attention again
        out_0 = (self.p_a_bilinear(p_weighted) * a0_weighted).sum(dim=1)
        out_0 += (self.q_a_bilinear(q_weighted) * a0_weighted).sum(dim=1)

        out_1 = (self.p_a_bilinear(p_weighted) * a1_weighted).sum(dim=1)
        out_1 += (self.q_a_bilinear(q_weighted) * a1_weighted).sum(dim=1)

        # Output vector is 2-dim vector with both logits
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

        self.emb_dropout = torch.nn.Dropout(p=0.5)

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
        p_emb = self.emb_dropout(self.word_embeddings(passage))
        q_emb = self.emb_dropout(self.word_embeddings(question))
        a0_emb = self.emb_dropout(self.word_embeddings(answer0))
        a1_emb = self.emb_dropout(self.word_embeddings(answer1))
        # Then we use those embeddings (along with the masks) as inputs for
        # our encoders
        p_hiddens = self.p_encoder(p_emb, p_mask)
        q_hidden = self.q_encoder(q_emb, q_mask)
        a0_hidden = self.a_encoder(a0_emb, a0_mask)
        a1_hidden = self.a_encoder(a1_emb, a1_mask)

        # We weight the text hidden states according to text-question attention
        p_q_attn = self.p_q_attn(q_hidden, p_hiddens, p_mask)
        p_weighted = util.weighted_sum(p_hiddens, p_q_attn)

        # We combine the output with a bilinear attention from text to answer
        out0 = (self.p_a_bilinear(p_weighted) * a0_hidden).sum(dim=1)
        out1 = (self.p_a_bilinear(p_weighted) * a1_hidden).sum(dim=1)
        logits = torch.stack((out0, out1), dim=1)
        # We also compute the class with highest likelihood (our prediction)
        prob = torch.softmax(logits, dim=-1)
        output = {"logits": logits, "prob": prob}

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


@Model.register('simple-bert-classifier')
class SimpleBertClassifier(Model):
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

    # This function computes the metrics we want to see during training.
    # For now, we only have the accuracy metric, but we could have a number
    # of different metrics here.
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


@Model.register('advanced-bert-classifier')
class AdvancedBertClassifier(Model):
    """
    Model similar to the AttentiveClassifier with BERT, but without external
    features.

    SimpleTrian is this with the attention before the encoders.
    """

    def __init__(self,
                 bert_path: Path,
                 encoder: Seq2VecEncoder,
                 rel_embeddings: TextFieldEmbedder,
                 vocab: Vocabulary,
                 hidden_dim: int = 100,
                 encoder_dropout: float = 0.0,
                 train_bert: bool = False
                 ) -> None:
        # We have to pass the vocabulary to the constructor.
        super().__init__(vocab)
        self.word_embeddings = bert_embeddings(pretrained_model=bert_path,
                                               training=train_bert)

        # self.rel_embeddings = rel_embeddings

        if encoder_dropout > 0:
            self.encoder_dropout = torch.nn.Dropout(p=encoder_dropout)
        else:
            self.encoder_dropout = lambda x: x

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

    # This function computes the metrics we want to see during training.
    # For now, we only have the accuracy metric, but we could have a number
    # of different metrics here.
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


@Model.register('simple-trian')
class SimpleTrian(Model):
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
                 rel_embeddings: TextFieldEmbedder,
                 p_encoder: Seq2SeqEncoder,
                 q_encoder: Seq2SeqEncoder,
                 a_encoder: Seq2SeqEncoder,
                 vocab: Vocabulary,
                 embedding_dropout: float = 0.0,
                 encoder_dropout: float = 0.0) -> None:
        # We have to pass the vocabulary to the constructor.
        super().__init__(vocab)

        self.word_embeddings = word_embeddings
        self.rel_embeddings = rel_embeddings

        if embedding_dropout > 0:
            self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)
        else:
            self.embedding_dropout = lambda x: x

        if encoder_dropout > 0:
            self.encoder_dropout = torch.nn.Dropout(p=encoder_dropout)
        else:
            self.encoder_dropout = lambda x: x

        embedding_dim = word_embeddings.get_output_dim()
        self.p_q_match = SequenceAttention(input_dim=embedding_dim)
        self.a_p_match = SequenceAttention(input_dim=embedding_dim)
        self.a_q_match = SequenceAttention(input_dim=embedding_dim)

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
        self.q_self_attn = LinearSelfAttention(
            input_dim=self.q_encoder.get_output_dim()
        )
        self.a_self_attn = LinearSelfAttention(
            input_dim=self.a_encoder.get_output_dim()
        )
        self.p_a_bilinear = torch.nn.Linear(
            in_features=self.p_encoder.get_output_dim(),
            out_features=self.a_encoder.get_output_dim()
        )
        self.q_a_bilinear = torch.nn.Linear(
            in_features=self.q_encoder.get_output_dim(),
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
                p_q_rel: Dict[str, torch.Tensor],
                p_a0_rel: Dict[str, torch.Tensor],
                p_a1_rel: Dict[str, torch.Tensor],
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
        p_emb = self.embedding_dropout(self.word_embeddings(passage))
        q_emb = self.embedding_dropout(self.word_embeddings(question))
        a0_emb = self.embedding_dropout(self.word_embeddings(answer0))
        a1_emb = self.embedding_dropout(self.word_embeddings(answer1))
        # And the relations
        p_q_rel_emb = self.embedding_dropout(self.rel_embeddings(p_q_rel))
        p_a0_rel_emb = self.embedding_dropout(self.rel_embeddings(p_a0_rel))
        p_a1_rel_emb = self.embedding_dropout(self.rel_embeddings(p_a1_rel))

        # We compute the Sequence Attention
        # First the scores
        p_q_scores = self.p_q_match(p_emb, q_emb, q_mask)
        a0_q_scores = self.a_q_match(a0_emb, q_emb, q_mask)
        a1_q_scores = self.a_q_match(a1_emb, q_emb, q_mask)
        a0_p_scores = self.a_p_match(a0_emb, p_emb, p_mask)
        a1_p_scores = self.a_p_match(a1_emb, p_emb, p_mask)

        # Then the weighted inputs
        p_q_match = util.weighted_sum(q_emb, p_q_scores)
        a0_q_match = util.weighted_sum(q_emb, a0_q_scores)
        a1_q_match = util.weighted_sum(q_emb, a1_q_scores)
        a0_p_match = util.weighted_sum(p_emb, a0_p_scores)
        a1_p_match = util.weighted_sum(p_emb, a1_p_scores)

        # We combine the inputs to our encoder
        p_input = torch.cat((p_emb, p_q_match, p_q_rel_emb,
                             p_a0_rel_emb, p_a1_rel_emb), dim=2)
        a0_input = torch.cat((a0_emb, a0_p_match, a0_q_match), dim=2)
        a1_input = torch.cat((a1_emb, a1_p_match, a1_q_match), dim=2)
        q_input = q_emb

        # Then we use those (along with the masks) as inputs for
        # our encoders
        p_hiddens = self.encoder_dropout(self.p_encoder(p_input, p_mask))
        q_hiddens = self.encoder_dropout(self.q_encoder(q_input, q_mask))
        a0_hiddens = self.encoder_dropout(self.a_encoder(a0_input, a0_mask))
        a1_hiddens = self.encoder_dropout(self.a_encoder(a1_input, a1_mask))

        # We compute the self-attention scores
        q_attn = self.q_self_attn(q_hiddens, q_hiddens, q_mask)
        a0_attn = self.a_self_attn(a0_hiddens, a0_hiddens, a0_mask)
        a1_attn = self.a_self_attn(a1_hiddens, a1_hiddens, a1_mask)

        # Then we weight the hidden-states with those scores
        q_weighted = util.weighted_sum(q_hiddens, q_attn)
        a0_weighted = util.weighted_sum(a0_hiddens, a0_attn)
        a1_weighted = util.weighted_sum(a1_hiddens, a1_attn)

        # We weight the text states with a passage-question bilinear attention
        p_q_attn = self.p_q_attn(q_weighted, p_hiddens, p_mask)
        p_weighted = util.weighted_sum(p_hiddens, p_q_attn)

        # Calculate the outputs for each answer, from passage-answer and
        # question-answer attention again
        out_0 = (self.p_a_bilinear(p_weighted) * a0_weighted).sum(dim=1)
        out_0 += (self.q_a_bilinear(q_weighted) * a0_weighted).sum(dim=1)

        out_1 = (self.p_a_bilinear(p_weighted) * a1_weighted).sum(dim=1)
        out_1 += (self.q_a_bilinear(q_weighted) * a1_weighted).sum(dim=1)

        # Output vector is 2-dim vector with both logits
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

    # This function computes the metrics we want to see during training.
    # For now, we only have the accuracy metric, but we could have a number
    # of different metrics here.
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


@Model.register('hierarchical-bert')
class HierarchicalBert(Model):
    """
    Uses hierarchical RNNs to build a sentence representation (from the BERT
    windows) and then build a document representation from those sentences.
    """

    def __init__(self,
                 bert_path: Path,
                 sentence_encoder: Seq2VecEncoder,
                 document_encoder: Seq2VecEncoder,
                 rel_embeddings: TextFieldEmbedder,
                 vocab: Vocabulary,
                 encoder_dropout: float = 0.0,
                 train_bert: bool = False
                 ) -> None:
        # We have to pass the vocabulary to the constructor.
        super().__init__(vocab)
        self.word_embeddings = bert_embeddings(pretrained_model=bert_path,
                                               training=train_bert)

        # self.rel_embeddings = rel_embeddings

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

    # This function computes the metrics we want to see during training.
    # For now, we only have the accuracy metric, but we could have a number
    # of different metrics here.
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


def seq_over_seq(encoder: Seq2VecEncoder,
                 sentences: torch.Tensor,
                 masks: torch.Tensor) -> torch.Tensor:
    num_batch, num_sent, _ = masks.shape
    enc_size = encoder.get_output_dim()
    sentence_encodings = sentences.new_empty(num_batch, num_sent, enc_size)

    for i in range(num_sent):
        sentence_emb = sentences[:, i, :, :]
        sentence_mask = masks[:, i, :]
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


@Model.register('advattn-bert-classifier')
class AdvancedAttentionBertClassifier(Model):
    """
    Model similar to the AttentiveClassifier with BERT, but without external
    features.

    SimpleTrian is this with the attention before the encoders.
    """

    def __init__(self,
                 bert_path: Path,
                 encoder: Seq2SeqEncoder,
                 rel_embeddings: TextFieldEmbedder,
                 vocab: Vocabulary,
                 hidden_dim: int = 100,
                 encoder_dropout: float = 0.0,
                 train_bert: bool = False
                 ) -> None:
        # We have to pass the vocabulary to the constructor.
        super().__init__(vocab)
        self.word_embeddings = bert_embeddings(pretrained_model=bert_path,
                                               training=train_bert)

        # self.rel_embeddings = rel_embeddings

        if encoder_dropout > 0:
            self.encoder_dropout = torch.nn.Dropout(p=encoder_dropout)
        else:
            self.encoder_dropout = lambda x: x

        self.pooler = BertPooler(pretrained_model=str(bert_path))
        self.dense1 = torch.nn.Linear(
            in_features=self.pooler.get_output_dim(),
            out_features=hidden_dim
        )
        self.encoder = encoder
        self.self_attn = LinearSelfAttention(
            input_dim=self.encoder.get_output_dim(),
            bias=True
        )
        self.dense2 = torch.nn.Linear(
            in_features=self.encoder.get_output_dim(),
            out_features=1
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

        # We create the embeddings from the input text
        t0_embs = self.word_embeddings(bert0)
        t1_embs = self.word_embeddings(bert1)

        t0_pooled = self.pooler(t0_embs)
        t1_pooled = self.pooler(t1_embs)

        t0_transformed = self.dense1(t0_pooled)
        t1_transformed = self.dense1(t1_pooled)

        t0_enc_hiddens = self.encoder_dropout(
            self.encoder(t0_transformed, mask=None))
        t1_enc_hiddens = self.encoder_dropout(
            self.encoder(t1_transformed, mask=None))

        t0_enc_attn = self.self_attn(t0_enc_hiddens, t0_enc_hiddens)
        t1_enc_attn = self.self_attn(t1_enc_hiddens, t1_enc_hiddens)

        t0_enc_out = util.weighted_sum(t0_enc_hiddens, t0_enc_attn)
        t1_enc_out = util.weighted_sum(t1_enc_hiddens, t1_enc_attn)

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

    # This function computes the metrics we want to see during training.
    # For now, we only have the accuracy metric, but we could have a number
    # of different metrics here.
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


@Model.register('hierarchical-attn-bert')
class HierarchicalAttentionNetwork(Model):
    """
    Uses hierarchical RNNs to build a sentence representation (from the
    windows) and then build a document representation from those sentences.
    """

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 sentence_encoder: Seq2SeqEncoder,
                 document_encoder: Seq2SeqEncoder,
                 relation_encoder: Seq2VecEncoder,
                 rel_embeddings: TextFieldEmbedder,
                 vocab: Vocabulary,
                 encoder_dropout: float = 0.0,
                 train_bert: bool = False
                 ) -> None:
        # We have to pass the vocabulary to the constructor.
        super().__init__(vocab)

        self.word_embeddings = word_embeddings
        self.rel_embeddings = rel_embeddings
        self.relation_encoder = relation_encoder

        if encoder_dropout > 0:
            self.encoder_dropout = torch.nn.Dropout(p=encoder_dropout)
        else:
            self.encoder_dropout = lambda x: x

        self.sentence_encoder = sentence_encoder
        self.sentence_attn = LinearSelfAttention(
            input_dim=self.sentence_encoder.get_output_dim(),
            bias=True
        )
        self.document_encoder = document_encoder
        self.document_attn = LinearSelfAttention(
            input_dim=self.document_encoder.get_output_dim(),
            bias=True
        )

        # dense_input_dim = (document_encoder.get_output_dim() +
        #                    2 * relation_encoder.get_output_dim())
        self.dense = torch.nn.Linear(
            in_features=document_encoder.get_output_dim(),
            out_features=1
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
                bert0: Dict[str, torch.Tensor],
                bert1: Dict[str, torch.Tensor],
                passage: Dict[str, torch.Tensor],
                question: Dict[str, torch.Tensor],
                answer0: Dict[str, torch.Tensor],
                answer1: Dict[str, torch.Tensor],
                p_q_rel: Dict[str, torch.Tensor],
                p_a0_rel: Dict[str, torch.Tensor],
                p_a1_rel: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        # Every sample in a batch has to have the same size (as it's a tensor),
        # so smaller entries are padded. The mask is used to counteract this
        # padding.

        t0_masks = util.get_text_field_mask(bert0, num_wrapping_dims=1)
        t1_masks = util.get_text_field_mask(bert1, num_wrapping_dims=1)

        # We create the embeddings from the input text
        t0_embs = self.word_embeddings(bert0)
        t1_embs = self.word_embeddings(bert1)

        t0_sentence_hiddens = self.encoder_dropout(
            hierarchical_seq_over_seq(self.sentence_encoder, t0_embs,
                                      t0_masks))
        t1_sentence_hiddens = self.encoder_dropout(
            hierarchical_seq_over_seq(self.sentence_encoder, t1_embs,
                                      t1_masks))

        t0_sentence_attns = self.sentence_attn(
            t0_sentence_hiddens, t0_sentence_hiddens)
        t1_sentence_attns = self.sentence_attn(
            t1_sentence_hiddens, t1_sentence_hiddens)

        t0_sentence_encodings = util.weighted_sum(
            t0_sentence_hiddens, t0_sentence_attns)
        t1_sentence_encodings = util.weighted_sum(
            t1_sentence_hiddens, t1_sentence_attns)

        t0_document_hiddens = self.encoder_dropout(self.document_encoder(
            t0_sentence_encodings, mask=None))
        t1_document_hiddens = self.encoder_dropout(self.document_encoder(
            t1_sentence_encodings, mask=None))

        t0_document_attn = self.document_attn(
            t0_document_hiddens, t0_document_hiddens)
        t1_document_attn = self.document_attn(
            t1_document_hiddens, t1_document_hiddens)

        t0_document_encoding = util.weighted_sum(
            t0_document_hiddens, t0_document_attn)
        t1_document_encoding = util.weighted_sum(
            t1_document_hiddens, t1_document_attn)

        # Now, the relations
        p_q_rel_masks = util.get_text_field_mask(p_q_rel)
        p_a0_rel_masks = util.get_text_field_mask(p_a0_rel)
        p_a1_rel_masks = util.get_text_field_mask(p_a1_rel)

        p_q_rel_embs = self.rel_embeddings(p_q_rel)
        p_a0_rel_embs = self.rel_embeddings(p_a0_rel)
        p_a1_rel_embs = self.rel_embeddings(p_a1_rel)

        p_q_encs = self.relation_encoder(p_q_rel_embs, p_q_rel_masks)
        p_a0_encs = self.relation_encoder(p_a0_rel_embs, p_a0_rel_masks)
        p_a1_encs = self.relation_encoder(p_a1_rel_embs, p_a1_rel_masks)

        p_q_enc = p_q_encs.mean(dim=1)
        p_a0_enc = p_a0_encs.mean(dim=1)
        p_a1_enc = p_a1_encs.mean(dim=1)

        t0_final = torch.cat((t0_document_encoding, p_q_enc, p_a0_enc),
                             dim=-1)
        t1_final = torch.cat((t1_document_encoding, p_q_enc, p_a1_enc),
                             dim=-1)

        t0_final = t0_document_encoding
        t1_final = t1_document_encoding
        # Joining everything and getting the result
        logit0 = self.dense(t0_final).squeeze(-1)
        logit1 = self.dense(t1_final).squeeze(-1)

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

    # This function computes the metrics we want to see during training.
    # For now, we only have the accuracy metric, but we could have a number
    # of different metrics here.
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
