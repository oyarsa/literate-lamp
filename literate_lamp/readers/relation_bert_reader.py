from typing import Optional
from pathlib import Path

from allennlp.data import Instance
from allennlp.data.tokenizers.word_splitter import (SpacyWordSplitter,
                                                    BertBasicWordSplitter)
from allennlp.data.token_indexers import PretrainedBertIndexer, TokenIndexer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.fields import (TextField, LabelField, MetadataField,
                                  ListField)

from readers.mc_script_reader import McScriptReader
from conceptnet import ConceptNet
from readers.util import (toks2strs, bert_sliding_window,
                   relation_sentences)


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
        q_a0_relations = relation_sentences(
            self.conceptnet, question_words, answer0_words)
        q_a1_relations = relation_sentences(
            self.conceptnet, question_words, answer1_words)
        # p_p_relations = relation_sentences(
        #     self.conceptnet, passage_words, passage_words)
        # q_q_relations = relation_sentences(
        #     self.conceptnet, question_words, question_words)
        # a0_a0_relations = relation_sentences(
        #     self.conceptnet, answer0_words, answer0_words)
        # a1_a1_relations = relation_sentences(
        #     self.conceptnet, answer1_words, answer1_words)

        # common_relations = p_q_relations + p_p_relations + q_q_relations
        # p_a0_relations += common_relations + q_a0_relations + a0_a0_relations
        # p_a1_relations += common_relations + q_a1_relations + a1_a1_relations
        # p_a0_rel_set = set(p_a0_relations)
        # p_a1_rel_set = set(p_a1_relations)

        p_a0_rel_set = set(p_a0_relations + p_q_relations + q_a0_relations)
        p_a1_rel_set = set(p_a1_relations + p_q_relations + q_a1_relations)

        p_a0_tokens = (self.tokeniser.tokenize(text=b) for b in p_a0_rel_set)
        p_a1_tokens = (self.tokeniser.tokenize(text=b) for b in p_a1_rel_set)

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
