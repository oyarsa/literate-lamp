#!/usr/bin/env python3
import argparse
import pickle
import random
import sys
from pathlib import Path

import torch

import args
import common
import util


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate or perform error analysis with pre-trained "
                    "models. If you want to train a model, use train.py"
    )
    parser.add_argument('command', choices=['evaluate', 'analysis'],
                        help="Whether to evaluate performance on test set or "
                             "perform error analysis.")
    parser.add_argument('--path', type=str, required=True,
                        help="Path to the model folder.")
    parser.add_argument('--data', type=str,
                        help="Path to the testing data file.")
    parser.add_argument('--sample_size', type=int, default=10,
                        help="Number of items to draw for error analysis.")

    if '--' in sys.argv:
        first, second = util.split_list(sys.argv[1:], '--')
    else:
        first, second = sys.argv[1:], []
    opts = parser.parse_args(first)
    ARGS = args.get_args(second)
    common.set_args(ARGS)

    util.print_args(vars(opts))
    util.print_args(ARGS)

    model_path = Path(opts.path)
    build_fn, reader_type = common.get_modelfn_reader()
    reader = common.create_reader(reader_type)

    with open(model_path / 'vocabulary.pickle', 'rb') as vocab_file:
        model = build_fn(pickle.load(vocab_file))
    # Recreate the model.
    # Load the state from the file
    with open(model_path / 'model.th', 'rb') as model_file:
        model.load_state_dict(torch.load(model_file))
    model.eval()

    if isinstance(ARGS.CUDA_DEVICE, int):
        cuda_device = ARGS.CUDA_DEVICE
    elif isinstance(args.ARGS.CUDA_DEVICE, list):
        cuda_device = ARGS.CUDA_DEVICE[0]

    # We've loaded the model. Let's move it to the GPU again if available.
    if cuda_device > -1:
        model.cuda(cuda_device)

    if opts.data:
        data_path = Path(opts.data)
        pre_processed_path = None
    else:
        data_path = ARGS.TEST_DATA_PATH
        pre_processed_path = ARGS.TEST_PREPROCESSED_PATH

    test_data = util.load_data(
        data_path=data_path,
        reader=reader,
        pre_processed_path=pre_processed_path
    )

    if opts.command == 'evaluate':
        common.evaluate(model, reader, test_data)
    if opts.command == 'analysis':
        common.error_analysis(model, test_data, sample_size=opts.sample_size)

    torch.manual_seed(ARGS.RANDOM_SEED)
    random.seed(ARGS.RANDOM_SEED)

    pass


if __name__ == '__main__':
    main()
