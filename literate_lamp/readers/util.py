"Utilities shared by the readers."
from typing import List, Sequence, Optional
from pathlib import Path

import numpy as np
from allennlp.data.tokenizers import Token, WordTokenizer
from allennlp.data.tokenizers.word_splitter import (SpacyWordSplitter,
                                                    BertBasicWordSplitter,
                                                    WordSplitter)
from allennlp.data.token_indexers import PretrainedBertIndexer, TokenIndexer
from allennlp.data.token_indexers import SingleIdTokenIndexer

from conceptnet import ConceptNet, triple_as_sentence
from util import is_stopword, is_punctuation, get_term_frequency
from modules import XLNetWordSplitter, XLNetIndexer


def strs2toks(strings: Sequence[str]) -> List[Token]:
    "Converts each string in the list to a Token"
    return [Token(s) for s in strings]


def toks2strs(tokens: Sequence[Token]) -> List[str]:
    "Converts each Token in the list to a str (using the text attribute)"
    return [t.text for t in tokens]


def compute_handcrafted_features(passage: Sequence[Token],
                                 question: Sequence[Token],
                                 answer0: Sequence[Token],
                                 answer1: Sequence[Token]) -> np.ndarray:
    def is_valid(token: Token) -> bool:
        return not is_stopword(token) and not is_punctuation(token)

    def co_occurrence(text: Sequence[str], query: Sequence[str]
                      ) -> List[float]:
        query_set = set(q.lower() for q in query)
        return [(is_valid(word) and word in query_set) for word in text]

    p_lemmas = [t.lemma_ for t in passage]
    q_lemmas = [t.lemma_ for t in question]
    a0_lemmas = [t.lemma_ for t in answer0]
    a1_lemmas = [t.lemma_ for t in answer1]

    p_words = toks2strs(passage)
    q_words = toks2strs(question)
    a0_words = toks2strs(answer0)
    a1_words = toks2strs(answer1)

    p_q_co_occ = co_occurrence(p_words, q_words)
    p_a0_co_occ = co_occurrence(p_words, a0_words)
    p_a1_co_occ = co_occurrence(p_words, a1_words)
    # dim: len * 3
    co_occ = np.vstack((p_q_co_occ, p_a0_co_occ, p_a1_co_occ))

    p_q_lem_co_occ = co_occurrence(p_lemmas, q_lemmas)
    p_a0_lem_co_occ = co_occurrence(p_lemmas, a0_lemmas)
    p_a1_lem_co_occ = co_occurrence(p_lemmas, a1_lemmas)
    # dim: len * 3
    lemma_co_occ = np.vstack((p_q_lem_co_occ, p_a0_lem_co_occ,
                              p_a1_lem_co_occ))

    # dim: len * 1
    tf = np.array([get_term_frequency(word) for word in passage])

    # dim: len * 3 + len * 3 + len * 1 = len * 7
    features = np.vstack((co_occ, lemma_co_occ, tf)).T
    return features


def bert_sliding_window(question: str, answer: str, passage: str,
                        max_wordpieces: int, stride: Optional[int] = None,
                        ) -> List[str]:
    pieces = []
    special_tokens = 4  # [CLS] + 3 [SEP]
    window_size = max_wordpieces - len(question) - len(answer) - special_tokens

    if stride is None:
        stride = window_size

    for i in range(0, len(passage), stride):
        window = passage[i:i + window_size]
        piece = f'{question} [SEP] {answer} [SEP] {window}'
        pieces.append(piece)

    return pieces


def xlnet_input_string(question: str, answer: str, passage: str) -> str:
    return f'{question} [SEP] {answer} [SEP] {passage}'


def relation_sentences(conceptnet: ConceptNet, text: Sequence[str],
                       query: Sequence[str]) -> List[str]:
    triples = conceptnet.get_all_text_query_triples(text, query)
    sentences = [triple_as_sentence(t) for t in triples]
    return sentences


def get_tokenizer(embedding_type: str, xlnet_vocab_file: Path) -> WordSplitter:
    if embedding_type == 'bert':
        splitter = BertBasicWordSplitter()
    elif embedding_type == 'glove':
        splitter = SpacyWordSplitter()
    elif embedding_type == 'xlnet':
        splitter = XLNetWordSplitter(vocab_file=str(xlnet_vocab_file))
    return WordTokenizer(word_splitter=splitter)


def get_indexer(embedding_type: str,
                xlnet_vocab_file: Path,
                max_seq_length: int) -> TokenIndexer:
    if embedding_type == 'bert':
        return PretrainedBertIndexer(
            pretrained_model='bert-base-uncased',
            truncate_long_sequences=False)
    if embedding_type == 'glove':
        return SingleIdTokenIndexer(lowercase_tokens=True)
    if embedding_type == 'xlnet':
        return XLNetIndexer(
            max_seq_length=max_seq_length,
            vocab_file=str(xlnet_vocab_file)
        )
