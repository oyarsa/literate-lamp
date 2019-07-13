"Reads data from file and extracts features used in the models"
from typing import Iterator, Optional, List, Sequence
import json
from pathlib import Path

import numpy as np
# A sample is represented as an `Instance`, which data as `TextField`s, and
# the target as a `LabelField`.
from allennlp.data import Instance
# As we have three parts in the input (passage, question, answer), we'll have
# three such `TextField`s in an Instance. The `LabelField` will hold 0|1.
from allennlp.data.fields import (TextField, LabelField, MetadataField,
                                  ArrayField, ListField)
# This implements the logic for reading a data file and extracting a list of
# `Instance`s from it.
from allennlp.data.dataset_readers import DatasetReader
# Tokens can be indexed in many ways. The `TokenIndexer` is the abstract class
# for this, but here we'll use the `SingleIdTokenIndexer`, which maps each
# token in the vocabulary to an integer.
from allennlp.data.token_indexers import (PosTagIndexer, NerTagIndexer,
                                          SingleIdTokenIndexer)
# This converts a word into a `Token` object, with fields for POS tags and
# such.
from allennlp.data.tokenizers import WordTokenizer, Token
from allennlp.data.tokenizers.word_splitter import (SpacyWordSplitter,
                                                    BertBasicWordSplitter)
from allennlp.data.token_indexers import PretrainedBertIndexer, TokenIndexer

from conceptnet import ConceptNet, triple_as_sentence
import util


class McScriptReader(DatasetReader):
    """Base class for the readers in this module.
    As they all read from the same type of file, the `_read` function is
    shared. However, they provide different features for different modules,
    so they create different `Instance`s
    """

    def _read(self, file_path: str) -> Iterator[Instance]:
        "Reads a file from `file_path` and returns a list of `Instances`."
        with open(file_path) as datafile:
            data = json.load(datafile)

        instances = data['data']['instance']
        for instance in instances:
            passage_id = instance['@id']
            passage = instance['text']['#text']

            if 'question' not in instance['questions']:
                instance['questions'] = []
            elif isinstance(instance['questions']['question'], list):
                instance['questions'] = instance['questions']['question']
            else:
                instance['questions'] = [instance['questions']['question']]
            questions = instance['questions']

            for question in questions:
                question_id = question['@id']
                question_text = question['@text']

                answers = ["", ""]
                labels = ["", ""]

                for answer_dicts in question['answer']:
                    if answer_dicts['@id'] == '0':
                        index = 0
                    else:
                        index = 1

                    answers[index] = answer_dicts['@text']
                    labels[index] = answer_dicts['@correct']

                assert "" not in answers, "Answers have to be non-empty"
                assert "" not in labels, "Labels have to be non-empty"

                yield self.text_to_instance(passage_id, question_id, passage,
                                            question_text, answers[0],
                                            answers[1], labels[0])


@DatasetReader.register('fulltrian-reader')
class FullTrianReader(McScriptReader):
    """
     DatasetReader for Question Answering data, from a JSON converted from the
     original XML files (using xml2json).

     Each `Instance` will have these fields:
        - `passage_id`: the id of the text
        - `question_id`: the id of question
        - `passage`: the main text from which the question will ask about
        - `question`: the question text
        - `answer0`: the first candidate answer
        - `answer1`: the second candidate answer
        - `label`: 0 if answer0 is the correct one, 1 if answer1 is correct
        - `passage_pos`: POS tags of the passage text
        - `question_pos`: POS tags of the question text
        - `passage_ner`: NER tags of the passage text
        - `p_q_rel`: ConceptNet relations between passage and question
        - `p_a0_rel`: ConceptNet relations between passage and answer 0
        - `p_a1_rel`: ConceptNet relations between passage and answer 1
        - `hc_feat`: handcrafted features (co-occurrences, term frequency)
     """

    # Initialise using a TokenIndexer, if provided. If not, create a new one.
    def __init__(self,
                 word_indexer: Optional[TokenIndexer] = None,
                 is_bert: bool = False,
                 conceptnet_path: Optional[Path] = None):
        super().__init__(lazy=False)
        self.pos_indexers = {"pos_tokens": PosTagIndexer()}
        self.ner_indexers = {"ner_tokens": NerTagIndexer()}
        self.rel_indexers = {
            "rel_tokens": SingleIdTokenIndexer(namespace='rel_tokens')}

        if is_bert:
            splitter = BertBasicWordSplitter()
        else:
            splitter = SpacyWordSplitter()
        self.tokeniser = WordTokenizer(word_splitter=splitter)

        self.word_indexers = {'tokens': word_indexer}
        word_splitter = SpacyWordSplitter(pos_tags=True, ner=True, parse=True)
        self.word_tokeniser = WordTokenizer(word_splitter=word_splitter)
        bert_splitter = BertBasicWordSplitter()
        self.bert_tokeniser = WordTokenizer(word_splitter=bert_splitter)

        if word_indexer is None:
            if is_bert:
                word_indexer = PretrainedBertIndexer(
                    pretrained_model='bert-base-uncased',
                    truncate_long_sequences=False)
            else:
                word_indexer = SingleIdTokenIndexer(lowercase_tokens=True)
        self.word_indexers = {'tokens': word_indexer}

        self.conceptnet = ConceptNet(conceptnet_path=conceptnet_path)

    # Converts the text from each field in the input to `Token`s, and then
    # initialises `TextField` with them. These also take the token indexer
    # we initialised in the constructor, so it can keep track of each token's
    # index.
    # Then we create the `Instance` fields dict from the data. There's an
    # optional label. It's optional because at prediction time (testing) we
    # won't have the label.
    def text_to_instance(self,
                         passage_id: str,
                         question_id: str,
                         passage: str,
                         question: str,
                         answer0: str,
                         answer1: str,
                         label0: Optional[str] = None
                         ) -> Instance:
        passage_tokens = self.word_tokeniser.tokenize(text=passage)
        question_tokens = self.word_tokeniser.tokenize(text=question)
        answer0_tokens = self.word_tokeniser.tokenize(text=answer0)
        answer1_tokens = self.word_tokeniser.tokenize(text=answer1)

        passage_wordpieces = self.tokeniser.tokenize(text=passage)
        question_wordpieces = self.tokeniser.tokenize(text=question)
        answer0_wordpieces = self.tokeniser.tokenize(text=answer0)
        answer1_wordpieces = self.tokeniser.tokenize(text=answer1)

        passage_words = toks2strs(passage_wordpieces)
        question_words = toks2strs(question_wordpieces)
        answer0_words = toks2strs(answer0_wordpieces)
        answer1_words = toks2strs(answer1_wordpieces)

        p_q_relations = strs2toks(self.conceptnet.get_text_query_relations(
            passage_words, question_words))
        p_a0_relations = strs2toks(self.conceptnet.get_text_query_relations(
            passage_words, answer0_words))
        p_a1_relations = strs2toks(self.conceptnet.get_text_query_relations(
            passage_words, answer1_words))

        handcrafted_features = compute_handcrafted_features(
            passage=passage_tokens,
            question=question_tokens,
            answer0=answer0_tokens,
            answer1=answer1_tokens
        )

        fields = {
            "passage_id": MetadataField(passage_id),
            "question_id": MetadataField(question_id),
            "passage": TextField(passage_wordpieces, self.word_indexers),
            "passage_pos": TextField(passage_tokens, self.pos_indexers),
            "passage_ner": TextField(passage_tokens, self.ner_indexers),
            "question": TextField(question_wordpieces, self.word_indexers),
            "question_pos": TextField(question_tokens, self.pos_indexers),
            "answer0": TextField(answer0_wordpieces, self.word_indexers),
            "answer1": TextField(answer1_wordpieces, self.word_indexers),
            "p_q_rel": TextField(p_q_relations, self.rel_indexers),
            "p_a0_rel": TextField(p_a0_relations, self.rel_indexers),
            "p_a1_rel": TextField(p_a1_relations, self.rel_indexers),
            "hc_feat": ArrayField(handcrafted_features)
        }

        if label0 is not None:
            if label0 == "True":
                label = 0
            elif label0 == 'False':
                label = 1
            else:
                raise ValueError('Wrong value for Answer::correct')

            fields["label"] = LabelField(label=label, skip_indexing=True)

        return Instance(fields)


def strs2toks(strings: Sequence[str]) -> List[Token]:
    "Converts each string in the list to a Token"
    return [Token(s) for s in strings]


def toks2strs(tokens: Sequence[Token]) -> List[str]:
    "Converts each Token in the list to a str (using the text attribute)"
    return [t.text for t in tokens]


def compute_handcrafted_features(passage: Sequence[Token],
                                 question: Sequence[Token],
                                 answer0: Sequence[Token],
                                 answer1: Sequence[Token]) -> np.ndarray:
    def is_valid(token: Token) -> bool:
        return not util.is_stopword(token) and not util.is_punctuation(token)

    def co_occurrence(text: Sequence[str], query: Sequence[str]
                      ) -> List[float]:
        query_set = set(q.lower() for q in query)
        return [(is_valid(word) and word in query_set) for word in text]

    p_lemmas = [t.lemma_ for t in passage]
    q_lemmas = [t.lemma_ for t in question]
    a0_lemmas = [t.lemma_ for t in answer0]
    a1_lemmas = [t.lemma_ for t in answer1]

    p_words = toks2strs(passage)
    q_words = toks2strs(question)
    a0_words = toks2strs(answer0)
    a1_words = toks2strs(answer1)

    p_q_co_occ = co_occurrence(p_words, q_words)
    p_a0_co_occ = co_occurrence(p_words, a0_words)
    p_a1_co_occ = co_occurrence(p_words, a1_words)
    # dim: len * 3
    co_occ = np.vstack((p_q_co_occ, p_a0_co_occ, p_a1_co_occ))

    p_q_lem_co_occ = co_occurrence(p_lemmas, q_lemmas)
    p_a0_lem_co_occ = co_occurrence(p_lemmas, a0_lemmas)
    p_a1_lem_co_occ = co_occurrence(p_lemmas, a1_lemmas)
    # dim: len * 3
    lemma_co_occ = np.vstack((p_q_lem_co_occ, p_a0_lem_co_occ,
                              p_a1_lem_co_occ))

    # dim: len * 1
    tf = np.array([util.get_term_frequency(word) for word in passage])

    # dim: len * 3 + len * 3 + len * 1 = len * 7
    features = np.vstack((co_occ, lemma_co_occ, tf)).T
    return features


@DatasetReader.register('simpletrian-reader')
class SimpleMcScriptReader(McScriptReader):
    """
     DatasetReader for Question Answering data, from a JSON converted from the
     original XML files (using xml2json).

     Each `Instance` will have these fields:
         - `passage_id`: the id of the text
         - `question_id`: the id of question
         - `passage`: the main text from which the question will ask about
         - `question`: the question text
         - `answer0`: the first candidate answer
         - `answer1`: the second candidate answer
         - `label`: 0 if answer0 is the correct one, 1 if answer1 is correct
     """

    # Initialise using a TokenIndexer, if provided. If not, create a new one.
    def __init__(self,
                 word_indexer: Optional[TokenIndexer] = None,
                 is_bert: bool = False):
        super().__init__(lazy=False)

        if is_bert:
            splitter = BertBasicWordSplitter()
        else:
            splitter = SpacyWordSplitter()
        self.tokeniser = WordTokenizer(word_splitter=splitter)

        if word_indexer is None:
            if is_bert:
                word_indexer = PretrainedBertIndexer(
                    pretrained_model='bert-base-uncased',
                    truncate_long_sequences=False)
            else:
                word_indexer = SingleIdTokenIndexer(lowercase_tokens=True)
        self.word_indexers = {'tokens': word_indexer}

    # Converts the text from each field in the input to `Token`s, and then
    # initialises `TextField` with them. These also take the token indexer
    # we initialised in the constructor, so it can keep track of each token's
    # index.
    # Then we create the `Instance` fields dict from the data. There's an
    # optional label. It's optional because at prediction time (testing) we
    # won't have the label.
    def text_to_instance(self,
                         passage_id: str,
                         question_id: str,
                         passage: str,
                         question: str,
                         answer0: str,
                         answer1: str,
                         label0: Optional[str] = None
                         ) -> Instance:
        passage_tokens = self.tokeniser.tokenize(text=passage)
        question_tokens = self.tokeniser.tokenize(text=question)
        answer0_tokens = self.tokeniser.tokenize(text=answer0)
        answer1_tokens = self.tokeniser.tokenize(text=answer1)

        fields = {
            "passage_id": MetadataField(passage_id),
            "question_id": MetadataField(question_id),
            "passage": TextField(passage_tokens, self.word_indexers),
            "question": TextField(question_tokens, self.word_indexers),
            "answer0": TextField(answer0_tokens, self.word_indexers),
            "answer1": TextField(answer1_tokens, self.word_indexers),
        }

        if label0 is not None:
            if label0 == "True":
                label = 0
            elif label0 == 'False':
                label = 1
            else:
                raise ValueError('Wrong value for Answer::correct')

            fields["label"] = LabelField(label=label, skip_indexing=True)

        return Instance(fields)


class SimpleTrianReader(McScriptReader):
    """
     DatasetReader for Question Answering data, from a JSON converted from the
     original XML files (using xml2json).

     Each `Instance` will have these fields:
         - `passage_id`: the id of the text
         - `question_id`: the id of question
         - `passage`: the main text from which the question will ask about
         - `question`: the question text
         - `answer0`: the first candidate answer
         - `answer1`: the second candidate answer
         - `p_q_rel`: relations between passage and question
         - `p_a0_rel`: relations between passage and answer 0
         - `p_a1_rel`: relations between passage and answer 1
         - `label`: 0 if answer0 is the correct one, 1 if answer1 is correct
     """

    # Initialise using a TokenIndexer, if provided. If not, create a new one.
    def __init__(self,
                 word_indexer: Optional[TokenIndexer] = None,
                 is_bert: bool = False,
                 conceptnet_path: Optional[Path] = None):
        super().__init__(lazy=False)
        self.rel_indexers = {
            "rel_tokens": SingleIdTokenIndexer(namespace='rel_tokens')}

        if is_bert:
            splitter = BertBasicWordSplitter()
        else:
            splitter = SpacyWordSplitter()
        self.tokeniser = WordTokenizer(word_splitter=splitter)

        if word_indexer is None:
            if is_bert:
                word_indexer = PretrainedBertIndexer(
                    pretrained_model='bert-base-uncased',
                    truncate_long_sequences=False)
            else:
                word_indexer = SingleIdTokenIndexer(lowercase_tokens=True)
        self.word_indexers = {'tokens': word_indexer}

        self.conceptnet = ConceptNet(conceptnet_path=conceptnet_path)

    # Converts the text from each field in the input to `Token`s, and then
    # initialises `TextField` with them. These also take the token indexer
    # we initialised in the constructor, so it can keep track of each token's
    # index.
    # Then we create the `Instance` fields dict from the data. There's an
    # optional label. It's optional because at prediction time (testing) we
    # won't have the label.
    def text_to_instance(self,
                         passage_id: str,
                         question_id: str,
                         passage: str,
                         question: str,
                         answer0: str,
                         answer1: str,
                         label0: Optional[str] = None
                         ) -> Instance:
        passage_tokens = self.tokeniser.tokenize(text=passage)
        question_tokens = self.tokeniser.tokenize(text=question)
        answer0_tokens = self.tokeniser.tokenize(text=answer0)
        answer1_tokens = self.tokeniser.tokenize(text=answer1)

        passage_words = toks2strs(passage_tokens)
        question_words = toks2strs(question_tokens)
        answer0_words = toks2strs(answer0_tokens)
        answer1_words = toks2strs(answer1_tokens)

        p_q_relations = strs2toks(self.conceptnet.get_text_query_relations(
            passage_words, question_words))
        p_a0_relations = strs2toks(self.conceptnet.get_text_query_relations(
            passage_words, answer0_words))
        p_a1_relations = strs2toks(self.conceptnet.get_text_query_relations(
            passage_words, answer1_words))

        fields = {
            "passage_id": MetadataField(passage_id),
            "question_id": MetadataField(question_id),
            "passage": TextField(passage_tokens, self.word_indexers),
            "question": TextField(question_tokens, self.word_indexers),
            "answer0": TextField(answer0_tokens, self.word_indexers),
            "answer1": TextField(answer1_tokens, self.word_indexers),
            "p_q_rel": TextField(p_q_relations, self.rel_indexers),
            "p_a0_rel": TextField(p_a0_relations, self.rel_indexers),
            "p_a1_rel": TextField(p_a1_relations, self.rel_indexers),
        }

        if label0 is not None:
            if label0 == "True":
                label = 0
            elif label0 == 'False':
                label = 1
            else:
                raise ValueError('Wrong value for Answer::correct')

            fields["label"] = LabelField(label=label, skip_indexing=True)

        return Instance(fields)


class RelationBertReader(McScriptReader):
    """
    DatasetReader for Question Answering data, from a JSON converted from the
    original XML files (using xml2json).

    Each `Instance` will have these fields:
        - `passage_id`: the id of the text
        - `question_id`: the id of question
        - `bert0`: input text for first answer
        - `bert1`: input text for second answer
        - `passage`: the main text from which the question will ask about
        - `question`: the question text
        - `answer0`: the first candidate answer
        - `answer1`: the second candidate answer
        - `label`: 0 if answer0 is the correct one, 1 if answer1 is correct

    For `bert0` and `bert1`, the input text is split into windows, as the
    passage text is likely to be bigger than BERT's maximum size.

    Even though the models won't anything besides `bert0` and `bert1`, the
    fields are going to be used for sorting the input to minimise padding.
    This is done in the training function.  It isn't necessary, but changing
    that behaviour would involve too much work.
    """

    # Initialise using a TokenIndexer, if provided. If not, create a new one.
    def __init__(self,
                 is_bert: bool,
                 conceptnet_path: Path,
                 word_indexer: Optional[TokenIndexer] = None):
        super().__init__(lazy=False)

        if is_bert:
            splitter = BertBasicWordSplitter()
        else:
            splitter = SpacyWordSplitter()
        self.tokeniser = WordTokenizer(word_splitter=splitter)

        if word_indexer is None:
            if is_bert:
                word_indexer = PretrainedBertIndexer(
                    pretrained_model='bert-base-uncased',
                    truncate_long_sequences=True)
            else:
                word_indexer = SingleIdTokenIndexer(lowercase_tokens=True)
        self.word_indexers = {'tokens': word_indexer}

        # self.rel_indexers = {
        #     "rel_tokens": SingleIdTokenIndexer(namespace='rel_tokens')}
        self.conceptnet = ConceptNet(conceptnet_path=conceptnet_path)

    # Converts the text from each field in the input to `Token`s, and then
    # initialises `TextField` with them. These also take the token indexer
    # we initialised in the constructor, so it can keep track of each token's
    # index.
    # Then we create the `Instance` fields dict from the data. There's an
    # optional label. It's optional because at prediction time (testing) we
    # won't have the label.
    def text_to_instance(self,
                         passage_id: str,
                         question_id: str,
                         passage: str,
                         question: str,
                         answer0: str,
                         answer1: str,
                         label0: Optional[str] = None
                         ) -> Instance:
        if hasattr(self.word_indexers['tokens'], 'max_pieces'):
            max_pieces = self.word_indexers['tokens'].max_pieces
        else:
            max_pieces = 512

        bert0 = bert_sliding_window(question, answer0, passage, max_pieces)
        bert1 = bert_sliding_window(question, answer1, passage, max_pieces)

        bert0_tokens = [self.tokeniser.tokenize(text=b) for b in bert0]
        bert1_tokens = [self.tokeniser.tokenize(text=b) for b in bert1]

        bert0_fields = [TextField(b, self.word_indexers) for b in bert0_tokens]
        bert1_fields = [TextField(b, self.word_indexers) for b in bert1_tokens]

        passage_tokens = self.tokeniser.tokenize(text=passage[:max_pieces])
        question_tokens = self.tokeniser.tokenize(text=question)
        answer0_tokens = self.tokeniser.tokenize(text=answer0)
        answer1_tokens = self.tokeniser.tokenize(text=answer1)

        passage_words = toks2strs(passage_tokens)
        question_words = toks2strs(question_tokens)
        answer0_words = toks2strs(answer0_tokens)
        answer1_words = toks2strs(answer1_tokens)

        # p_q_relations = strs2toks(self.conceptnet.get_text_query_relations(
        #     passage_words, question_words))
        # p_a0_relations = strs2toks(self.conceptnet.get_text_query_relations(
        #     passage_words, answer0_words))
        # p_a1_relations = strs2toks(self.conceptnet.get_text_query_relations(
        #     passage_words, answer1_words))

        p_q_relations = relation_sentences(
            self.conceptnet, passage_words, question_words)
        p_a0_relations = relation_sentences(
            self.conceptnet, passage_words, answer0_words)
        p_a1_relations = relation_sentences(
            self.conceptnet, passage_words, answer1_words)

        p_a0_relations += p_q_relations
        p_a1_relations += p_q_relations

        p_q_tokens = [self.tokeniser.tokenize(text=b) for b in p_q_relations]
        p_a0_tokens = [self.tokeniser.tokenize(text=b) for b in p_a0_relations]
        p_a1_tokens = [self.tokeniser.tokenize(text=b) for b in p_a1_relations]

        p_q_fields = [TextField(b, self.word_indexers) for b in p_q_tokens]
        p_a0_fields = [TextField(b, self.word_indexers) for b in p_a0_tokens]
        p_a1_fields = [TextField(b, self.word_indexers) for b in p_a1_tokens]

        fields = {
            "passage_id": MetadataField(passage_id),
            "question_id": MetadataField(question_id),
            "bert0": ListField(bert0_fields),
            "bert1": ListField(bert1_fields),
            "passage": TextField(passage_tokens, self.word_indexers),
            "question": TextField(question_tokens, self.word_indexers),
            "answer0": TextField(answer0_tokens, self.word_indexers),
            "answer1": TextField(answer1_tokens, self.word_indexers),
            # "p_q_rel": TextField(p_q_relations, self.rel_indexers),
            # "p_a0_rel": TextField(p_a0_relations, self.rel_indexers),
            # "p_a1_rel": TextField(p_a1_relations, self.rel_indexers),
            "p_q_rel": ListField(p_q_fields),
            "p_a0_rel": ListField(p_a0_fields),
            "p_a1_rel": ListField(p_a1_fields),
        }

        if label0 is not None:
            if label0 == "True":
                label = 0
            elif label0 == 'False':
                label = 1
            else:
                raise ValueError('Wrong value for Answer::correct')

            fields["label"] = LabelField(label=label, skip_indexing=True)

        return Instance(fields)


class SimpleBertReader(McScriptReader):
    """
    DatasetReader for Question Answering data, from a JSON converted from the
    original XML files (using xml2json).

    Each `Instance` will have these fields:
        - `passage_id`: the id of the text
        - `question_id`: the id of question
        - `bert0`: input text for first answer
        - `bert1`: input text for second answer
        - `passage`: the main text from which the question will ask about
        - `question`: the question text
        - `answer0`: the first candidate answer
        - `answer1`: the second candidate answer
        - `label`: 0 if answer0 is the correct one, 1 if answer1 is correct

    For `bert0` and `bert1`, the input text is split into windows, as the
    passage text is likely to be bigger than BERT's maximum size.

    Even though the models won't anything besides `bert0` and `bert1`, the
    fields are going to be used for sorting the input to minimise padding.
    This is done in the training function.  It isn't necessary, but changing
    that behaviour would involve too much work.
    """

    # Initialise using a TokenIndexer, if provided. If not, create a new one.
    def __init__(self,
                 word_indexer: Optional[TokenIndexer] = None):
        super().__init__(lazy=False)

        splitter = BertBasicWordSplitter()
        self.tokeniser = WordTokenizer(word_splitter=splitter)

        if word_indexer is None:
            word_indexer = PretrainedBertIndexer(
                pretrained_model='bert-base-uncased',
                truncate_long_sequences=False)
        self.word_indexers = {'tokens': word_indexer}

    # Converts the text from each field in the input to `Token`s, and then
    # initialises `TextField` with them. These also take the token indexer
    # we initialised in the constructor, so it can keep track of each token's
    # index.
    # Then we create the `Instance` fields dict from the data. There's an
    # optional label. It's optional because at prediction time (testing) we
    # won't have the label.

    def text_to_instance(self,
                         passage_id: str,
                         question_id: str,
                         passage: str,
                         question: str,
                         answer0: str,
                         answer1: str,
                         label0: Optional[str] = None
                         ) -> Instance:
        max_pieces = self.word_indexers['tokens'].max_pieces
        bert0 = bert_sliding_window(question, answer0, passage, max_pieces)
        bert1 = bert_sliding_window(question, answer1, passage, max_pieces)

        bert0_tokens = [self.tokeniser.tokenize(text=b) for b in bert0]
        bert1_tokens = [self.tokeniser.tokenize(text=b) for b in bert1]

        bert0_fields = [TextField(b, self.word_indexers) for b in bert0_tokens]
        bert1_fields = [TextField(b, self.word_indexers) for b in bert1_tokens]

        passage_tokens = self.tokeniser.tokenize(text=passage[:max_pieces])
        question_tokens = self.tokeniser.tokenize(text=question)
        answer0_tokens = self.tokeniser.tokenize(text=answer0)
        answer1_tokens = self.tokeniser.tokenize(text=answer1)

        fields = {
            "passage_id": MetadataField(passage_id),
            "question_id": MetadataField(question_id),
            "bert0": ListField(bert0_fields),
            "bert1": ListField(bert1_fields),
            "passage": TextField(passage_tokens, self.word_indexers),
            "question": TextField(question_tokens, self.word_indexers),
            "answer0": TextField(answer0_tokens, self.word_indexers),
            "answer1": TextField(answer1_tokens, self.word_indexers),
        }

        if label0 is not None:
            if label0 == "True":
                label = 0
            elif label0 == 'False':
                label = 1
            else:
                raise ValueError('Wrong value for Answer::correct')

            fields["label"] = LabelField(label=label, skip_indexing=True)

        return Instance(fields)


def bert_sliding_window(question: str, answer: str, passage: str,
                        max_wordpieces: int, stride: Optional[int] = None,
                        ) -> List[str]:
    pieces = []
    special_tokens = 4  # [CLS] + 3 [SEP]
    window_size = max_wordpieces - len(question) - len(answer) - special_tokens

    if stride is None:
        stride = window_size

    for i in range(0, len(passage), stride):
        window = passage[i:i + window_size]
        piece = f'{question} [SEP] {answer} [SEP] {window}'
        pieces.append(piece)

    return pieces


def relation_sentences(conceptnet: ConceptNet, text: Sequence[str],
                       query: Sequence[str]) -> List[str]:
    triples = conceptnet.get_all_text_query_triples(text, query)
    sentences = [triple_as_sentence(t) for t in triples]
    return sentences
