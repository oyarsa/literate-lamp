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
from typing import Callable
from pprint import pprint
from pathlib import Path

import torch
import numpy as np
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.common import JsonDict

import args
import readers
import common
from predictor import McScriptPredictor
from util import example_input, is_cuda, train_model, load_data

ARGS = args.get_args()


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
        common.print_instance(passage, question, answer1,
                              answer2, prediction+1, label)

    return result


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

    common.evaluate(model, reader, test_dataset)
    result = make_prediction(model, reader, verbose=False)
    common.error_analysis(model, test_dataset)

    print('Save path', ARGS.SAVE_PATH)

    cuda_device = 0 if is_cuda(model) else -1
    test_load(build_fn, reader, ARGS.SAVE_PATH, result, cuda_device)


if __name__ == '__main__':
    torch.manual_seed(ARGS.RANDOM_SEED)
    random.seed(ARGS.RANDOM_SEED)

    run_model()
