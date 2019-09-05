#!/usr/bin/env python3

from typing import List, TypeVar, Iterable, Set

from allennlp.data import Instance

import args
import common
from readers import SimpleMcScriptReader


ARGS = args.get_args()
common.set_args(ARGS)

T = TypeVar('T')


def flatten(l: Iterable[List[T]]) -> List[T]:
    "Turns a list of lists into a list. Reduces one level of nesting."
    flat_list = [item for sublist in l for item in sublist]
    return flat_list


class TextStats:
    """
    Given a list of texts (each a list of strings), calculates statistics
    on length.
    """

    def __init__(self, instance_texts: List[List[str]]):
        instance_sizes = [len(instance) for instance in instance_texts]

        self.mean_size = sum(instance_sizes) / len(instance_sizes)
        self.max_size = max(instance_sizes)
        self.min_size = min(instance_sizes)

    def __repr__(self) -> str:
        return (f'Min: {self.min_size}. Max: {self.max_size}. '
                f'Mean: {self.mean_size}')


class DatasetStats:
    """
    Given a list of instances (the samples in a dataset), calculate the
    statistics over each field.
    """

    def __init__(self, instances: List[Instance]):
        questions = extract_field('question', instances)
        passages = extract_field('passage', instances)
        answers = extract_field('answer0', instances)
        answers += extract_field('answer1', instances)

        vocab: Set[str] = set(flatten(questions + passages + answers))

        self.passage_stats = TextStats(passages)
        self.question_stats = TextStats(questions)
        self.answer_stats = TextStats(answers)
        self.vocab_size = len(vocab)

    def __repr__(self) -> str:
        return (
            f'Passages: {self.passage_stats}\n'
            f'Questions: {self.question_stats}\n'
            f'Answers: {self.answer_stats}\n'
            f'Vocabulary: {self.vocab_size}\n'
        )


def extract_field(field: str, instances: List[Instance]) -> List[List[str]]:
    """
    Extracts `field` for each instance in `instances`, and then extract
    the tokens from that field.
    """
    instance_tokens = [instance[field].tokens for instance in instances]
    strings = [[token.text for token in tokens]
               for tokens in instance_tokens]
    return strings


def main() -> None:
    paths = [ARGS.TRAIN_DATA_PATH, ARGS.VAL_DATA_PATH, ARGS.TEST_DATA_PATH]
    reader = SimpleMcScriptReader()
    instances: List[Instance] = flatten(reader.read(path) for path in paths)
    stats = DatasetStats(instances)
    print(stats)


if __name__ == '__main__':
    main()
