from typing import Iterator, Dict, Optional
# A sample is represented as an `Instance`, which data as `TextField`s, and
# the target as a `LabelField`.
from allennlp.data import Instance
# As we have three parts in the input (passage, question, answer), we'll have
# three such `TextField`s in an Instance. The `LabelField` will hold 0|1.
from allennlp.data.fields import TextField, LabelField
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
    DatasetReader for Question Answering data, one sentence per line, like

        Passage|Question|Answer|Label

    Each `Instance` will have 4 fields:
        - `Passage`: the main text from which the question will ask about
        - `Question`: the question text
        - `Answer`: a candidate answer, can be true or false
        - `Label`: 1 if the answer is true, 0 otherwise
    """

    # Initialise using a TokenIndexer, if provided. If not, create a new one.
    def __init__(self, token_indexers: Optional[Dict[str, TokenIndexer]] = None
                 ) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer()}
        word_splitter = SpacyWordSplitter(pos_tags=True, parse=True, ner=True)
        self.tokeniser = WordTokenizer(word_splitter=word_splitter)

    # Converts the text from each field in the input to `Token`s, and then
    # initialises `TextField` with them. These also take the token indexer
    # we initialised in the constructor, so it can keep track of each token's
    # index.
    # Then we create the `Instance` fields dict from the data. There's an
    # optional label. It's optional because at prediction time (testing) we
    # won't have the label.
    def text_to_instance(self,
                         passage: str,
                         question: str,
                         answer: str,
                         label: Optional[str] = None
                         ) -> Instance:
        # passage_tokens = [Token(word) for word in passage.split()]
        # question_tokens = [Token(word) for word in question.split()]
        # answer_tokens = [Token(word) for word in answer.split()]
        passage_tokens = self.tokeniser.tokenize(text=passage)
        question_tokens = self.tokeniser.tokenize(text=question)
        answer_tokens = self.tokeniser.tokenize(text=answer)

        fields = {
            "passage": TextField(passage_tokens, self.token_indexers),
            "question": TextField(question_tokens, self.token_indexers),
            "answer": TextField(answer_tokens, self.token_indexers),
        }

        if label is not None:
            fields["label"] = LabelField(label=int(label), skip_indexing=True)

        return Instance(fields)

    # Reads a file from `file_path` and returns a list of `Instances`.
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            next(f)  # Skip header line
            for line in f:
                passage, question, answer, label = line.strip().split('|')
                yield self.text_to_instance(passage, question, answer, label)
