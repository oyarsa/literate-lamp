"Reader that builds the BERT strings for models that don't use relations."
from typing import Optional

from allennlp.data import Instance
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter
from allennlp.data.token_indexers import PretrainedBertIndexer, TokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.fields import (TextField, LabelField, MetadataField,
                                  ListField)

from readers.base_reader import BaseReader
from readers.util import bert_sliding_window


class SimpleBertReader(BaseReader):
    """
    DatasetReader for Question Answering data, from a JSON converted from the
    original XML files (using xml2json).

    Each `Instance` will have these fields:
        - `metadata`: text id, question id and question type
        - `bert0`: input text for first answer
        - `bert1`: input text for second answer
        - `label`: 0 if answer0 is the correct one, 1 if answer1 is correct

    For `bert0` and `bert1`, the input text is split into windows, as the
    passage text is likely to be bigger than BERT's maximum size.
    """
    keys = [
        ("bert0", "list_num_tokens"),
        ("bert1", "list_num_tokens")
    ]

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

        max_pieces = self.word_indexers['tokens'].max_pieces
        bert0 = bert_sliding_window(question, answer0, passage, max_pieces)
        bert1 = bert_sliding_window(question, answer1, passage, max_pieces)

        bert0_tokens = [self.tokeniser.tokenize(text=b) for b in bert0]
        bert1_tokens = [self.tokeniser.tokenize(text=b) for b in bert1]

        bert0_fields = [TextField(b, self.word_indexers) for b in bert0_tokens]
        bert1_fields = [TextField(b, self.word_indexers) for b in bert1_tokens]

        fields = {
            "metadata": MetadataField(metadata),
            "bert0": ListField(bert0_fields),
            "bert1": ListField(bert1_fields),
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
