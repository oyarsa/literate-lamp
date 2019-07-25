#!/usr/bin/env python3
"""
Script to run a baseline model for the SemEval-2018 Task 11 problem (also
COIN 2019 Task 1).
The dataset has a Passage, a Question and a Candidate answer. The target is to
predict if the answer is correct or not.
The baseline uses GloVe to build embeddings for each of the three texts, which
are then encoded using (different) LSTMs. The encoded vectors are then
concatenated and fed into a feed-forward layer that output class probabilities.

This script builds the model, trains it, generates predictions and saves it.
Then it checks if the saving went correctly.
"""
import pickle
import random
from typing import Dict, Tuple, Callable, List
from collections import defaultdict
from pprint import pprint
from pathlib import Path

import torch
import numpy as np
from allennlp.training.util import evaluate as allen_eval
from allennlp.models import Model
from allennlp.data.iterators import BucketIterator
from allennlp.data.fields import ListField
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.instance import Instance
from allennlp.common import JsonDict

import args
import readers
import common
from predictor import McScriptPredictor
from util import example_input, is_cuda, train_model, load_data, tf2str

ARGS = args.get_args()


def print_base_instance(instance: Instance, prediction: torch.Tensor) -> None:
    passage = tf2str(instance['passage'])
    question = tf2str(instance['question'])
    answer1 = tf2str(instance['answer0'])
    answer2 = tf2str(instance['answer1'])
    label = instance['label'].label
    print_instance(passage, question, answer1, answer2, prediction, label)


def print_xlnet_instance(instance: Instance,
                         probability: torch.Tensor
                         ) -> None:
    def clean(string: str) -> str:
        return string.replace("▁", "")

    string0 = instance['string0']
    passage, question_answer0 = tf2str(string0).split('<sep>')
    string1 = instance['string1']
    _, question_answer1 = tf2str(string1).split('<sep>')
    label = instance['label']
    prediction = probability.argmax()

    print('PASSAGE:\n', '\t', clean(passage), sep='')
    print('QUESTION+ANSWER0:\n', '\t', clean(question_answer0), sep='')
    print('QUESTION+ANSWER1:\n', '\t', clean(question_answer1), sep='')
    print('PREDICTION:', prediction, probability)
    print('CORRECT:', label.label)


def process_bert_list(fields: ListField) -> Tuple[str, str, str]:
    windows = []

    for field in fields:
        text = tf2str(field)
        split = text.split('[SEP]')
        question, answer, window = split
        windows.append(window)

    passage = " ".join(windows)
    return passage, question, answer


def print_bert_instance(instance: Instance, prediction: torch.Tensor) -> None:
    bert0 = instance['bert0']
    passage, question, answer0 = process_bert_list(bert0)
    bert1 = instance['bert1']
    _, _, answer1 = process_bert_list(bert1)
    label = instance['label'].label

    print_instance(passage, question, answer0, answer1, prediction, label)


def error_analysis(model: Model,
                   test_data: List[Instance],
                   sample_size: int = 10) -> None:
    base_readers = ['simple', 'simple-trian', 'full-trian']
    xlnet_readers = ['relation-xl', 'simple-xl', 'extended-xl']
    bert_readers = ['simple-bert', 'relation-bert']
    _, reader_type = common.get_modelfn_reader()

    print('#'*5, 'ERROR ANALYSIS', '#'*5)

    wrongs = []
    for instance in test_data:
        label = instance['label'].label

        output = model.forward_on_instance(instance)
        probability = output['prob']
        prediction = probability.argmax()

        if prediction != label:
            wrongs.append((instance, probability))

    wrongs = random.sample(wrongs, sample_size)
    for i, (wrong, predicted) in enumerate(wrongs):
        print(f'{i})')
        if reader_type in base_readers:
            print_base_instance(wrong, predicted)
        elif reader_type in xlnet_readers:
            print_xlnet_instance(wrong, predicted)
        elif reader_type in bert_readers:
            print_bert_instance(wrong, predicted)
        print('#'*10)
        print()


def print_instance(passage: str,
                   question: str,
                   answer1: str,
                   answer2: str,
                   probability: torch.Tensor,
                   label: int
                   ) -> None:
    print('PASSAGE:\n', '\t', passage, sep='')
    print('QUESTION:\n', '\t', question, sep='')
    print('ANSWERS:')
    print('\t0:', answer1)
    print('\t1:', answer2)
    prediction = probability.argmax()
    print('PREDICTION:', prediction, probability)
    print('CORRECT:', label)


def make_prediction(model: Model,
                    reader: readers.BaseReader,
                    verbose: bool = False
                    ) -> JsonDict:
    "Create a predictor to run our model and get predictions."
    model.eval()
    predictor = McScriptPredictor(model, reader)

    if verbose:
        print()
        print('#'*5, 'EXAMPLE', '#'*5)

    passage, question, answer1, label1 = example_input(0)
    _, _, answer2, _ = example_input(1)
    result = predictor.predict("", "", "", passage, question, answer1, answer2)
    prediction = np.argmax(result['prob'])

    if verbose:
        label = 1 if label1 == 1 else 2
        print_instance(passage, question, answer1,
                       answer2, prediction+1, label)

    return result


def split_list(data: List[Instance]) -> Dict[str, List[Instance]]:
    output: Dict[str, List[Instance]] = defaultdict(list)

    for sample in data:
        qtype = sample['metadata']['question_type']
        output[qtype].append(sample)

    return output


def evaluate(model: Model,
             reader: readers.BaseReader,
             test_data: List[Instance]
             ) -> None:
    vocab = Vocabulary.from_instances(test_data)
    iterator = BucketIterator(batch_size=ARGS.BATCH_SIZE,
                              sorting_keys=reader.keys)
    # Our data should be indexed using the vocabulary we learned.
    iterator.index_with(vocab)

    data_types = split_list(test_data)
    results: Dict[str, Tuple[int, float]] = {}

    print()
    print('#'*5, 'PER TYPE EVALUATION', '#'*5)
    for qtype, data in data_types.items():
        num_items = len(data)
        print(f'Type: {qtype} ({num_items})')

        metrics = allen_eval(model, data, iterator, ARGS.CUDA_DEVICE, "")
        print()

        accuracy = metrics['accuracy']
        results[qtype] = (num_items, accuracy)


def test_load(build_model_fn: Callable[[Vocabulary], Model],
              reader: readers.BaseReader,
              save_path: Path,
              original_prediction: JsonDict,
              cuda_device: int) -> None:
    "Test if we can load the model and if its prediction matches the original."
    print('\n>>>>Testing if the model saves and loads correctly')
    # Reload vocabulary
    with open(save_path / 'vocabulary.pickle', 'rb') as vocab_file:
        model = build_model_fn(pickle.load(vocab_file))
    # Recreate the model.
    # Load the state from the file
    with open(save_path / 'model.th', 'rb') as model_file:
        model.load_state_dict(torch.load(model_file))
    model.eval()
    # We've loaded the model. Let's move it to the GPU again if available.
    if cuda_device > -1:
        model.cuda(cuda_device)

    # Try predicting again and see if we get the same results (we should).
    predictor = McScriptPredictor(model, dataset_reader=reader)
    passage, question, answer0, _ = example_input(0)

    _, _, answer1, _ = example_input(1)
    prediction = predictor.predict(
        passage_id="",
        question_id="",
        question_type="",
        passage=passage,
        question=question,
        answer0=answer0,
        answer1=answer1
    )
    np.testing.assert_array_almost_equal(
        original_prediction['logits'], prediction['logits'])
    print('Success.')


def run_model() -> None:
    "Execute model according to the configuration"
    print('#'*5, 'PARAMETERS', '#'*5)
    pprint(ARGS)
    print('#'*10, '\n\n')

    # Which model to use?
    build_fn, reader_type = common.get_modelfn_reader()
    reader = common.create_reader(reader_type)

    def optimiser(model: Model) -> torch.optim.Optimizer:
        return torch.optim.Adamax(model.parameters(), lr=2e-3)

    # Create SAVE_FOLDER if it doesn't exist
    ARGS.SAVE_PATH.mkdir(exist_ok=True, parents=True)
    train_dataset = load_data(data_path=ARGS.TRAIN_DATA_PATH,
                              reader=reader,
                              pre_processed_path=ARGS.TRAIN_PREPROCESSED_PATH)
    val_dataset = load_data(data_path=ARGS.VAL_DATA_PATH,
                            reader=reader,
                            pre_processed_path=ARGS.VAL_PREPROCESSED_PATH)
    test_dataset = load_data(data_path=ARGS.TEST_DATA_PATH,
                             reader=reader,
                             pre_processed_path=ARGS.TEST_PREPROCESSED_PATH)

    model = train_model(build_fn,
                        train_data=train_dataset,
                        val_data=val_dataset,
                        test_data=test_dataset,
                        save_path=ARGS.SAVE_PATH,
                        num_epochs=ARGS.NUM_EPOCHS,
                        batch_size=ARGS.BATCH_SIZE,
                        optimiser_fn=optimiser,
                        cuda_device=ARGS.CUDA_DEVICE,
                        sorting_keys=reader.keys)

    evaluate(model, reader, test_dataset)
    result = make_prediction(model, reader, verbose=False)
    error_analysis(model, test_dataset)

    print('Save path', ARGS.SAVE_PATH)

    cuda_device = 0 if is_cuda(model) else -1
    test_load(build_fn, reader, ARGS.SAVE_PATH, result, cuda_device)


if __name__ == '__main__':
    torch.manual_seed(ARGS.RANDOM_SEED)
    random.seed(ARGS.RANDOM_SEED)

    run_model()
