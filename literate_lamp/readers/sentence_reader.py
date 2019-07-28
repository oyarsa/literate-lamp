"""
Reader that builds input for the DMN model. Especially, breaks up the text
passage into sentences.
"""
from typing import Optional
from pathlib import Path

from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.fields import (
    TextField, LabelField, MetadataField, ListField)

import readers.util as util
from readers.base_reader import BaseReader


class SentenceReader(BaseReader):
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
    keys = [
        ("sentences", "list_tokens_length"),
        ("question", "tokens_length")
    ]

    # Initialise using a TokenIndexer, if provided. If not, create a new one.
    def __init__(self,
                 embedding_type: str,
                 xlnet_vocab_file: Optional[Path] = None,
                 word_indexer: Optional[TokenIndexer] = None):
        super().__init__(lazy=False)

        self.sentenciser = util.get_sentencizer()
        self.tokeniser = util.get_tokenizer(embedding_type=embedding_type,
                                            xlnet_vocab_file=xlnet_vocab_file)

        if word_indexer is None:
            word_indexer = util.get_indexer(embedding_type=embedding_type,
                                            xlnet_vocab_file=xlnet_vocab_file)
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

        passage_sentences = util.split_sentences(self.sentenciser, passage)
        passage_sentences_tokens = [self.tokeniser.tokenize(sentence) for
                                    sentence in passage_sentences]
        passage_fields = [TextField(tokens, self.word_indexers)
                          for tokens in passage_sentences_tokens]

        question_tokens = self.tokeniser.tokenize(question)
        answer0_tokens = self.tokeniser.tokenize(answer0)
        answer1_tokens = self.tokeniser.tokenize(answer1)

        fields = {
            "metadata": MetadataField(metadata),
            "sentences": ListField(passage_fields),
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
