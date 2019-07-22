# pylint: disable=no-self-use
from typing import Dict, List, Union
import logging

from overrides import overrides

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer

from modules.xlnet_word_splitter import PretrainedXLNetTokenizer

logger = logging.getLogger(__name__)


class XLNetIndexer(TokenIndexer[int]):
    """
    A token indexer that does the wordpiece-tokenization (e.g. for BERT
    embeddings).  If you are using one of the pretrained BERT models, you'll
    want to use the ``PretrainedBertIndexer`` subclass rather than this base
    class.
    Parameters
    ----------
    vocab : ``Dict[str, int]``
        The mapping {wordpiece -> id}.  Note this is not an AllenNLP
        ``Vocabulary``.
    namespace : str, optional (default: "wordpiece")
        The namespace in the AllenNLP ``Vocabulary`` into which the wordpieces
        will be loaded.
    end_tokens : ``str``, optional (default=``[CLS]``)
        These are appended to the tokens provided to ``tokens_to_indices``.
    sep_token : ``str``, optional (default=``[SEP]``)
        This token indicates the segments in the sequence.
    token_min_padding_length : ``int``, optional (default=``0``)
        See :class:`TokenIndexer`.
    """

    def __init__(self,
                 namespace: str = "tokens",
                 vocab_file: str = 'xlnet-base-cased',
                 sep_token: str = '<sep>',
                 cls_token: str = '<cls>',
                 pad_token: Union[int, str] = 0,
                 token_min_padding_length: int = 0) -> None:
        super().__init__(token_min_padding_length)

        self.biggest = 0
        self.namespace = namespace
        self.tokeniser = PretrainedXLNetTokenizer.load(vocab_file)
        self.sep_token = sep_token
        self.cls_token = cls_token
        self.pad_token = pad_token

        self.cls_token_id = self.tokeniser.convert_tokens_to_ids(cls_token)
        self.sep_token_id = self.tokeniser.convert_tokens_to_ids(sep_token)

    @overrides
    def count_vocab_items(self,
                          token: Token,
                          counter: Dict[str, Dict[str, int]]) -> None:
        # If we only use pretrained models, we don't need to do anything here.
        pass

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[int]]:
        token_ids = self.tokeniser.convert_tokens_to_ids(
            t.text for t in tokens)
        total_length = len(token_ids) + 2
        self.biggest = max(total_length, self.biggest)

        input_ids = []
        segment_ids = []
        current_segment = 0

        for tid in token_ids:
            input_ids.append(tid)

            segment_ids.append(current_segment)
            if tid == self.sep_token_id:
                current_segment = 1

        input_ids.append(self.sep_token_id)
        # Every SEP is part of the preceding segment
        segment_ids.append(current_segment)

        input_ids.append(self.cls_token_id)
        segment_ids.append(0)  # CLS is part of first segment
        cls_index = len(input_ids) - 1

        mask = [1] * len(input_ids)

        return {
            index_name: input_ids,
            'cls-index': [cls_index],
            f'{index_name}-type-ids': segment_ids,
            'mask': mask,
        }

    @overrides
    def get_padding_token(self) -> int:
        return 0

    @overrides
    def get_padding_lengths(self, token: int) -> Dict[str, int]:
        return {}

    @overrides
    def pad_token_sequence(self,
                           tokens: Dict[str, List[int]],
                           desired_num_tokens: Dict[str, int],
                           padding_lengths: Dict[str, int]
                           ) -> Dict[str, List[int]]:
        return {key: pad_sequence_to_length(val, desired_num_tokens[key])
                for key, val in tokens.items()}

    @overrides
    def get_keys(self, index_name: str) -> List[str]:
        """
        We need to override this because the indexer generates multiple keys.
        """
        return [index_name, 'cls-index', f'{index_name}-type-ids', 'mask']
