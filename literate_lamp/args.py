"Processes command line arguments and configuration."
import sys
import argparse
from typing import cast, List, Optional
from pathlib import Path

from util import (DotDict, parse_cuda, get_experiment_name,
                  get_preprocessed_name)


def get_args(arguments: Optional[List[str]] = None) -> DotDict:
    "Processes command line arguments and configuration."
    parser = argparse.ArgumentParser(description="Train a model.")

    usage = """
USAGE:
    run.py CONFIG MODEL [EMBEDDING_TYPE] [CUDA_DEVICE] [NAME] [ENCODER] [TTYPE]

ARGS:
    CONFIG: configuration to use. One of: small, large
    MODEL: model to run. One of: baseline, attentive, reader
    EMBEDDING_TYPE: word embeddings for the text. One of: glove, bert.
    CUDA_DEVICE: device to run the training. -1 for CPU, >=0 for GPU.
    NAME: name for model being trained (used in saving)
    ENCODER: which encoder to use (lstm, gru, transformer, positional)
    TYYPE: transformer type (allen or custom)
"""
    if arguments is None:
        arguments = sys.argv[1:]

    parser.add_argument('--config', type=str, choices=['small', 'large'],
                        default='small',
                        help="Configuration to use, small or large.")
    parser.add_argument('--model', type=str, default='zero-trian',
                        help="Model to run.")
    parser.add_argument('--embedding', type=str, default='glove',
                        choices=['glove', 'bert', 'xlnet'],
                        help="Embedding to use, GloVe, BERT or XLNet")
    parser.add_argument('--cuda', type=str, default="0",
                        help="GPU(s) to use. If multiple, separated by "
                             "comma. If single, just use the gpu number. If "
                             "CPU is desired, use -1.\n"
                             "Examples:\n"
                             "\t--gpu 0,1,2\n"
                             "\t--gpu 0")
    parser.add_argument('--name', type=str, default=None,
                        help="Name for this model.")
    parser.add_argument('--encoder', type=str, default='lstm',
                        choices=['lstm', 'transformer'],
                        help="Encoder type, one of lstm or transformer")
    parser.add_argument('--transformer', type=str, default='custom',
                        choices=['allen', 'custom'],
                        help="If encoder is transformer, choose which one to "
                             "use, allen or custom.")

    res = parser.parse_args(arguments)
    args = DotDict()

    config = res.config
    args.MODEL = res.model

    args.NER_EMBEDDING_DIM = 8
    args.REL_EMBEDDING_DIM = 10
    args.POS_EMBEDDING_DIM = 12
    args.HANDCRAFTED_DIM = 7
    args.EMBEDDING_TYPE = res.embedding

    args.CUDA_DEVICE = parse_cuda(res.cuda)
    args.MODEL_NAME = res.name

    data_folder = Path('data')
    # Proper configuration path for the External folder. The data one is
    # going to be part of the repo, so this is fine for now, but External isn't
    # always going to be.
    external_folder = Path('..', 'External')

    xlnet_use_window = True

    if config == 'large':
        # Path to our dataset
        args.TRAIN_DATA_PATH = data_folder / 'mctrain-data.json'
        args.VAL_DATA_PATH = data_folder / 'mcdev-data.json'
        args.TEST_DATA_PATH = data_folder / 'mctest-data.json'
        # Path to our embeddings
        args.GLOVE_PATH = external_folder / 'glove.840B.300d.txt'
        # Size of our embeddings
        args.GLOVE_EMBEDDING_DIM = 300
        # Size of our hidden layers (for each encoder)
        args.HIDDEN_DIM = 100
        args.TRANSFORMER_DIM = 512
        # Size of minibatch
        args.BATCH_SIZE = 24
        # Number of epochs to train model
        args.NUM_EPOCHS = 30
        # Size of the input window for XLNet
        if xlnet_use_window:
            args.xlnet_window_size = 512
        else:
            args.xlnet_window_size = None
    elif config == 'small':
        # Path to our dataset
        args.TRAIN_DATA_PATH = data_folder / 'small-train.json'
        args.VAL_DATA_PATH = data_folder / 'small-dev.json'
        args.TEST_DATA_PATH = data_folder / 'small-test.json'
        # Path to our embeddings
        args.GLOVE_PATH = external_folder / 'glove.6B.50d.txt'
        # Size of our embeddings
        args.GLOVE_EMBEDDING_DIM = 50
        # Size of our hidden layers (for each encoder)
        args.HIDDEN_DIM = 50
        args.TRANSFORMER_DIM = 128
        # Size of minibatch
        args.BATCH_SIZE = 2
        # Number of epochs to train model
        args.NUM_EPOCHS = 1
        # Size of the input window for XLNet
        if xlnet_use_window:
            args.xlnet_window_size = 128
        else:
            args.xlnet_window_size = None

    args.BERT_PATH = external_folder / 'bert-base-uncased.tar.gz'
    args.CONCEPTNET_PATH = external_folder / 'conceptnet.csv'
    args.xlnet_vocab_path = data_folder / 'xlnet-base-cased-spiece.model'
    args.xlnet_config_path = data_folder / 'xlnet-base-cased-config.json'
    args.xlnet_model_path = external_folder / \
        'xlnet-base-cased-pytorch_model.bin'

    # Path to save the Model and Vocabulary
    args.SAVE_FOLDER = Path('experiments')
    args.SAVE_PATH = args.SAVE_FOLDER / \
        get_experiment_name(args.MODEL, config,
                            args.EMBEDDING_TYPE, args.MODEL_NAME)
    print('Save path', args.SAVE_PATH)

    def preprocessed_name(split_type: str) -> str:
        "Gets the pre-processed pickle filename from the configuration."
        name = get_preprocessed_name(split_type, args.MODEL, config,
                                     args.EMBEDDING_TYPE)
        return cast(str, name)

    # Path to save pre-processed input
    args.TRAIN_PREPROCESSED_NAME = preprocessed_name('train')
    args.VAL_PREPROCESSED_NAME = preprocessed_name('val')
    args.TEST_PREPROCESSED_NAME = preprocessed_name('test')

    args.TRAIN_PREPROCESSED_PATH = (external_folder /
                                    args.TRAIN_PREPROCESSED_NAME)
    args.VAL_PREPROCESSED_PATH = external_folder / args.VAL_PREPROCESSED_NAME
    args.TEST_PREPROCESSED_PATH = external_folder / args.TEST_PREPROCESSED_NAME
    print('Pre-processed data path:', args.TRAIN_PREPROCESSED_PATH)

    # Random seed (for reproducibility)
    args.RANDOM_SEED = 1234

    # Model Configuration
    # Use LSTM, GRU or Transformer
    args.ENCODER_TYPE = res.encoder
    args.WHICH_TRANSFORMER = res.transformer
    args.BIDIRECTIONAL = True
    args.RNN_LAYERS = 1
    args.RNN_DROPOUT = 0.5 if args.ENCODER_TYPE != 'transformer' else 0
    args.EMBEDDDING_DROPOUT = 0.5 if args.EMBEDDING_TYPE == 'glove' else 0

    # Number of passes the DMN will make over the input
    args.DMN_PASSES = 2

    # Whether to fine tune the embeddings (specifically BERT and XLNet)
    args.finetune_embeddings = False

    return args
