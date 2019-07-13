"""
This module implements the ConceptNet class, to interact with the knowledge
base and make queries for relations between words and sentences.
"""
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Sequence, Set, Tuple
from pathlib import Path
import re


class ConceptNet:
    """
    Loads a database of(relation, word, word) triples. This database
    can be queried with two words to obtain the relationship between them.
    We can also query a sentence against another and obtain a list of relations
    """
    NULL_REL = '<NULL>'

    def __init__(self, conceptnet_path: Optional[Path] = None) -> None:
        """
        Loads(relation, word1, word2) triples from `triples_path`
        and then accepts queries for relations between words.
        """
        self._relations: DefaultDict[str, Dict[str, str]] = defaultdict(dict)

        if conceptnet_path is None:
            print('[conceptnet.py/ConceptNet] No ConceptNet data provided')
            return

        with open(conceptnet_path, encoding='utf-8') as infile:
            for line in infile:
                relation, word1, word2 = line.strip().split('\t')
                # Relation is reflexive
                self._relations[word1][word2] = relation
                self._relations[word2][word1] = relation

    def get_relation(self, word1: str, word2: str) -> str:
        """
        Lowercases word1 and word2, replacing spaces by underscores(which is
        what ConceptNet uses as separators). Then queries the data if there is
        a relation between word1 and word2.

        This is reflexive, so order doesn't matter.
        """
        word1 = '_'.join(word1.lower().split())
        word2 = '_'.join(word2.lower().split())

        if word1 in self._relations:
            return self._relations[word1].get(word2, ConceptNet.NULL_REL)
        return ConceptNet.NULL_REL

    def get_all_text_query_triples(self, text: Sequence[str],
                                   query: Sequence[str]
                                   ) -> Set[Tuple[str, str, str]]:
        triples: Set[Tuple[str, str, str]] = set()

        for text_word in text:
            for query_word in query:
                if text_word == query_word:
                    continue
                relation = self.get_relation(text_word, query_word)
                if relation == ConceptNet.NULL_REL:
                    continue
                triples.add((text_word, relation, query_word))

        if not triples:
            triples.add(("No", "Relation", "Found"))

        return triples

    def get_text_query_relations(self, text: Sequence[str],
                                 query: Sequence[str]) -> List[str]:
        """
        Gets a list of relations. For each word in text, we see if there's
        a relation for any word in query. If there is, we use the first we
        find as the relation for that word.
        """
        relations = [ConceptNet.NULL_REL] * len(text)
        query_set = set(q.lower() for q in query)

        for i, text_word in enumerate(text):
            for query_word in query_set:
                # Attempt to get a relation
                relation = self.get_relation(text_word, query_word)
                # If we did find one, we'll stop looking
                if relation != ConceptNet.NULL_REL:
                    relations[i] = relation
                    break

        return relations


def triple_as_sentence(triple: Tuple[str, str, str]) -> str:
    head, relation, tail = triple
    parts = re.findall('[A-Z][^A-Z]*', relation)
    relation = " ".join(parts)
    return f'{head} {relation} {tail}'
