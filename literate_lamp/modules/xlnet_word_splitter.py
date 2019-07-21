from typing import List, Dict

from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.word_splitter import WordSplitter
from pytorch_transformers import XLNetTokenizer


class PretrainedXLNetTokenizer:
    """
    In some instances you may want to load the same XLNet model twice
    (e.g. to use as a token embedder and also as a pooling layer).
    This factory provides a cache so that you don't actually have to load the
    model twice.
    """
    _cache: Dict[str, XLNetTokenizer] = {}

    @classmethod
    def load(cls, vocab_file: str, cache_model: bool = True) -> XLNetTokenizer:
        if vocab_file in cls._cache:
            return PretrainedXLNetTokenizer._cache[vocab_file]

        model = XLNetTokenizer(vocab_file=vocab_file)
        if cache_model:
            cls._cache[vocab_file] = model

        return model


class XLNetWordSplitter(WordSplitter):
    """
    The ``BasicWordSplitter`` from the BERT implementation.
    This is used to split a sentence into words.
    Then the ``BertTokenIndexer`` converts each word into wordpieces.
    """

    def __init__(self,
                 vocab_file: str,
                 do_lower_case: bool = False
                 ) -> None:
        self.tokeniser = PretrainedXLNetTokenizer.load(vocab_file)

    def split_words(self, sentence: str) -> List[Token]:
        return [Token(t) for t in self.tokeniser.tokenize(sentence)]
