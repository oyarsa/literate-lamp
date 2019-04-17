"Reads data from file and extracts features used in the models"
from typing import Iterator, Optional, List
import json
from pathlib import Path

# A sample is represented as an `Instance`, which data as `TextField`s, and
# the target as a `LabelField`.
from allennlp.data import Instance
# As we have three parts in the input (passage, question, answer), we'll have
# three such `TextField`s in an Instance. The `LabelField` will hold 0|1.
from allennlp.data.fields import TextField, LabelField, MetadataField
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
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.token_indexers import PretrainedBertIndexer, TokenIndexer

from conceptnet import ConceptNet


@DatasetReader.register('mcscript-reader')
class McScriptReader(DatasetReader):
    """
     DatasetReader for Question Answering data, from a JSON converted from the
     original XML files (using xml2json).

     Each `Instance` will have 4 fields:
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
                 conceptnet_path: Optional[Path] = None):
        super().__init__(lazy=False)
        self.pos_indexers = {"pos_tokens": PosTagIndexer()}
        self.ner_indexers = {"ner_tokens": NerTagIndexer()}
        self.rel_indexers = {
            "rel_tokens": SingleIdTokenIndexer(namespace='rel_tokens')}

        word_splitter = SpacyWordSplitter(pos_tags=True, ner=True, parse=True)
        self.word_tokeniser = WordTokenizer(word_splitter=word_splitter)

        word_indexer = word_indexer or PretrainedBertIndexer(
            pretrained_model='bert-base-uncased')
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
            "passage_pos": TextField(passage_tokens, self.pos_indexers),
            "passage_ner": TextField(passage_tokens, self.ner_indexers),
            "question": TextField(question_tokens, self.word_indexers),
            "question_pos": TextField(question_tokens, self.pos_indexers),
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

    # Reads a file from `file_path` and returns a list of `Instances`.
    def _read(self, file_path: str) -> Iterator[Instance]:
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


def strs2toks(strings: List[str]) -> List[Token]:
    "Converts each string in the list to a Token"
    return [Token(s) for s in strings]


def toks2strs(tokens: List[Token]) -> List[str]:
    "Converts each Token in the list to a str (using the text attribute)"
    return [t.text for t in tokens]
