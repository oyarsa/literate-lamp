from typing import Iterator, Dict, Optional
import json
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
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
# This converts a word into a `Token` object, with fields for POS tags and
# such.
# TODO: Explore tokenisers from this package.
#   Currently we're doing just 'split', which is enough (as the data is laid
#   out accordingly), but using a proper tokeniser could give us more
#   information.
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter


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
                 token_indexers: Optional[Dict[str, TokenIndexer]] = None,
                 lowercase_tokens: bool = False) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer(lowercase_tokens=lowercase_tokens)}
        word_splitter = SpacyWordSplitter(
            pos_tags=False, parse=False, ner=False)
        self.tokeniser = WordTokenizer(word_splitter=word_splitter)

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
        # passage_tokens = [Token(word) for word in passage.split()]
        # question_tokens = [Token(word) for word in question.split()]
        # answer_tokens = [Token(word) for word in answer.split()]
        passage_tokens = self.tokeniser.tokenize(text=passage)
        question_tokens = self.tokeniser.tokenize(text=question)
        answer0_tokens = self.tokeniser.tokenize(text=answer0)
        answer1_tokens = self.tokeniser.tokenize(text=answer1)

        fields = {
            "passage_id": MetadataField(passage_id),
            "question_id": MetadataField(question_id),
            "passage": TextField(passage_tokens, self.token_indexers),
            "question": TextField(question_tokens, self.token_indexers),
            "answer0": TextField(answer0_tokens, self.token_indexers),
            "answer1": TextField(answer1_tokens, self.token_indexers),
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
        with open(file_path) as f:
            data = json.load(f)

        instances = data['data']['instance']
        for instance in instances:
            passage_id = instance['@id']
            passage = instance['text']['#text']

            if 'question' not in instance['questions']:
                instance['questions'] = []
            elif type(instance['questions']['question']) is list:
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
