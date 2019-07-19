from typing import Optional
from pathlib import Path

from allennlp.data import Instance
from allennlp.data.tokenizers.word_splitter import (SpacyWordSplitter,
                                                    BertBasicWordSplitter)
from allennlp.data.token_indexers import PretrainedBertIndexer, TokenIndexer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.fields import TextField, LabelField, MetadataField

from readers.mc_script_reader import McScriptReader
from conceptnet import ConceptNet
from readers.util import strs2toks, toks2strs


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
