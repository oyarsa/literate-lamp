#!/usr/bin/env python3
import sys
from typing import Tuple, TextIO


def clean_word(string: str) -> Tuple[str, str]:
    pieces = string.split('/')
    language = pieces[2]
    word = pieces[3]
    return language, word


def process_file(input_file: TextIO, output_file: TextIO) -> None:
    for line in input_file:
        fields = line.split('\t')
        relation = fields[1]
        word1 = fields[2]
        word2 = fields[3]

        relation = relation.split('/')[2]
        lang1, word1 = clean_word(word1)
        lang2, word2 = clean_word(word2)

        if lang1 != 'en' or lang2 != 'en':  # We only accept English words
            continue

        print('{}\t{}\t{}'.format(relation, word1, word2), file=output_file)


if __name__ == '__main__':
    if any('help' in arg or '-h' in arg for arg in sys.argv):
        print('Arguments: path_to_conceptnet output_path')
        exit(0)
    if len(sys.argv) >= 3:
        concetpnet_path = sys.argv[1]
        output_path = sys.argv[2]
    else:
        print('Missing arguments: path_to_conceptnet output_path')
        exit(1)

    with open(concetpnet_path, encoding='utf-8') as input_file:
        with open(output_path, 'w', encoding='utf-8') as output_file:
            process_file(input_file=input_file, output_file=output_file)
