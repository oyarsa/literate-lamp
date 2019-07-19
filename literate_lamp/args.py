"Processes command line arguments and configuration."
import sys
from typing import cast
from pathlib import Path

from util import (DotDict, parse_cuda, get_experiment_name,
                  get_preprocessed_name)


def get_args() -> DotDict:
    "Processes command line arguments and configuration."
    usage = """
USAGE:
    run.py CONFIG MODEL [EMBEDDING_TYPE] [CUDA_DEVICE] [NAME] [ENCODER] [TTYPE]

ARGS:
    CONFIG: configuration to use. One of: small, large
    MODEL: model to run. One of: baseline, attentive, reader
    EMBEDDING_TYPE: word embeddings for the text. One of: glove, bert.
    CUDA_DEVICE: device to run the training. -1 for CPU, >=0 for GPU.
    NAME: name for model being trained (used in saving)
    ENCODER: which encoder to use (lstm, gru, transformer)
    TYYPE: transformer type (allen or custom)
"""
    if any('help' in arg or '-h' in arg for arg in sys.argv):
        print(usage)
        exit(0)

    args = DotDict()

    default_config = 'small'  # Can be: _large_ or _small_
    config = sys.argv[1] if len(sys.argv) >= 2 else default_config

    # Which model to use: 'baseline', 'reader', 'simple-bert', 'advanced-bert',
    #  or 'attentive'.
    default_model = 'attentive'
    args.MODEL = sys.argv[2] if len(sys.argv) >= 3 else default_model

    args.NER_EMBEDDING_DIM = 8
    args.REL_EMBEDDING_DIM = 10
    args.POS_EMBEDDING_DIM = 12
    args.HANDCRAFTED_DIM = 7
    default_embedding_type = 'glove'  # can also be 'bert'
    args.EMBEDDING_TYPE = sys.argv[3] if len(
        sys.argv) >= 4 else default_embedding_type

    args.CUDA_DEVICE = parse_cuda(sys.argv[4]) if len(sys.argv) >= 5 else 0
    args.MODEL_NAME = sys.argv[5] if len(sys.argv) >= 6 else None

    data_folder = Path('data')
    # Proper configuration path for the External folder. The data one is
    # going to be part of the repo, so this is fine for now, but External isn't
    # always going to be.
    external_folder = Path('..', 'External')

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
        args.HIDDEN_DIM = 50
        args.TRANSFORMER_DIM = 512
        # Size of minibatch
        args.BATCH_SIZE = 24
        # Number of epochs to train model
        args.NUM_EPOCHS = 30
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

    args.BERT_PATH = external_folder / 'bert-base-uncased.tar.gz'
    args.CONCEPTNET_PATH = external_folder / 'conceptnet.csv'

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
    args.ENCODER_TYPE = sys.argv[6] if len(sys.argv) >= 7 else 'lstm'
    args.WHICH_TRANSFORMER = sys.argv[7] if len(sys.argv) >= 8 else 'allen'
    args.BIDIRECTIONAL = True
    args.RNN_LAYERS = 1
    args.RNN_DROPOUT = 0.5 if args.ENCODER_TYPE != 'transformer' else 0
    args.EMBEDDDING_DROPOUT = 0.5 if args.EMBEDDING_TYPE != 'bert' else 0

    # What encoder to use to join the relation embeddings into a single vector.
    args.RELATION_ENCODER = 'cnn'

    return args
