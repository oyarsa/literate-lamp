"Reader that builds XLNet strings in conjuction with querying ConceptNet."
from typing import Optional
from pathlib import Path

from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.fields import (TextField, LabelField, MetadataField,
                                  ListField)

from conceptnet import ConceptNet
from readers.base_reader import BaseReader
from readers.util import pieces2strs, xlnet_input_string, relation_sentences
from modules import XLNetWordSplitter, XLNetIndexer


class RelationXLNetReader(BaseReader):
    """
    DatasetReader for Question Answering data, from a JSON converted from the
    original XML files (using xml2json).

    Each `Instance` will have these fields:
        - `passage_id`: the id of the text
        - `question_id`: the id of question
        - `string0`: input text for first answer
        - `string1`: input text for second answer
        - `rel0`: list of relation sentences for the first answer
        - `rel1`: list of relation sentences for the second answer
        - `label`: 0 if answer0 is the correct one, 1 if answer1 is correct

    For `bert0` and `bert1`, the input text is split into windows, as the
    passage text is likely to be bigger than BERT's maximum size.
    """
    keys = [
        ("string0", "tokens_length"),
        ("string1", "tokens_length")
    ]

    # Initialise using a TokenIndexer, if provided. If not, create a new one.
    def __init__(self,
                 vocab_file: Path,
                 conceptnet_path: Path,
                 word_indexer: Optional[TokenIndexer] = None):
        super().__init__(lazy=False)

        splitter = XLNetWordSplitter(vocab_file=str(vocab_file))
        self.tokeniser = WordTokenizer(word_splitter=splitter)

        if word_indexer is None:
            word_indexer = XLNetIndexer(vocab_file=str(vocab_file))
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
                         question_type: str,
                         passage: str,
                         question: str,
                         answer0: str,
                         answer1: str,
                         label0: Optional[str] = None
                         ) -> Instance:
        metadata = {
            'passage_id': passage_id,
            'question_id': question_id,
            'question_type': question_type,
        }

        string0 = xlnet_input_string(question, answer0, passage)
        string1 = xlnet_input_string(question, answer1, passage)

        tokens0 = self.tokeniser.tokenize(text=string0)
        tokens1 = self.tokeniser.tokenize(text=string1)

        passage_tokens = self.tokeniser.tokenize(text=passage)
        question_tokens = self.tokeniser.tokenize(text=question)
        answer0_tokens = self.tokeniser.tokenize(text=answer0)
        answer1_tokens = self.tokeniser.tokenize(text=answer1)

        passage_words = pieces2strs(passage_tokens)
        question_words = pieces2strs(question_tokens)
        answer0_words = pieces2strs(answer0_tokens)
        answer1_words = pieces2strs(answer1_tokens)

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

        rel0_set = set(p_a0_relations + p_q_relations + q_a0_relations)
        rel1_set = set(p_a1_relations + p_q_relations + q_a1_relations)

        rel0_tokens = (self.tokeniser.tokenize(text=b) for b in rel0_set)
        rel1_tokens = (self.tokeniser.tokenize(text=b) for b in rel1_set)

        rel0_fields = [TextField(b, self.word_indexers) for b in rel0_tokens]
        rel1_fields = [TextField(b, self.word_indexers) for b in rel1_tokens]

        fields = {
            "metadata": MetadataField(metadata),
            "string0": TextField(tokens0, self.word_indexers),
            "string1": TextField(tokens1, self.word_indexers),
            "rel0": ListField(rel0_fields),
            "rel1": ListField(rel1_fields),
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
