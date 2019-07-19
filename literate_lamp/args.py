import sys
from typing import cast
from pathlib import Path

from util import (DotDict, parse_cuda, get_experiment_name,
                  get_preprocessed_name)


def get_args() -> DotDict:
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

    f = DotDict()

    DEFAULT_CONFIG = 'small'  # Can be: _large_ or _small_
    CONFIG = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_CONFIG

    # Which model to use: 'baseline', 'reader', 'simple-bert', 'advanced-bert',
    #  or 'attentive'.
    DEFAULT_MODEL = 'attentive'
    f.MODEL = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_MODEL

    f.NER_EMBEDDING_DIM = 8
    f.REL_EMBEDDING_DIM = 10
    f.POS_EMBEDDING_DIM = 12
    f.HANDCRAFTED_DIM = 7
    DEFAULT_EMBEDDING_TYPE = 'glove'  # can also be 'bert'
    f.EMBEDDING_TYPE = sys.argv[3] if len(
        sys.argv) >= 4 else DEFAULT_EMBEDDING_TYPE

    f.CUDA_DEVICE = parse_cuda(sys.argv[4]) if len(sys.argv) >= 5 else 0
    f.MODEL_NAME = sys.argv[5] if len(sys.argv) >= 6 else None

    DATA_FOLDER = Path('data')
    # Proper configuration path for the External folder. The data one is
    # going to be part of the repo, so this is fine for now, but External isn't
    # always going to be.
    EXTERNAL_FOLDER = Path('..', 'External')

    if CONFIG == 'large':
        # Path to our dataset
        f.TRAIN_DATA_PATH = DATA_FOLDER / 'mctrain-data.json'
        f.VAL_DATA_PATH = DATA_FOLDER / 'mcdev-data.json'
        f.TEST_DATA_PATH = DATA_FOLDER / 'mctest-data.json'
        # Path to our embeddings
        f.GLOVE_PATH = EXTERNAL_FOLDER / 'glove.840B.300d.txt'
        # Size of our embeddings
        f.GLOVE_EMBEDDING_DIM = 300
        # Size of our hidden layers (for each encoder)
        f.HIDDEN_DIM = 50
        f.TRANSFORMER_DIM = 512
        # Size of minibatch
        f.BATCH_SIZE = 24
        # Number of epochs to train model
        f.NUM_EPOCHS = 30
    elif CONFIG == 'small':
        # Path to our dataset
        f.TRAIN_DATA_PATH = DATA_FOLDER / 'small-train.json'
        f.VAL_DATA_PATH = DATA_FOLDER / 'small-dev.json'
        f.TEST_DATA_PATH = DATA_FOLDER / 'small-test.json'
        # Path to our embeddings
        f.GLOVE_PATH = EXTERNAL_FOLDER / 'glove.6B.50d.txt'
        # Size of our embeddings
        f.GLOVE_EMBEDDING_DIM = 50
        # Size of our hidden layers (for each encoder)
        f.HIDDEN_DIM = 50
        f.TRANSFORMER_DIM = 128
        # Size of minibatch
        f.BATCH_SIZE = 2
        # Number of epochs to train model
        f.NUM_EPOCHS = 5

    f.BERT_PATH = EXTERNAL_FOLDER / 'bert-base-uncased.tar.gz'
    f.CONCEPTNET_PATH = EXTERNAL_FOLDER / 'conceptnet.csv'

    # Path to save the Model and Vocabulary
    f.SAVE_FOLDER = Path('experiments')
    f.SAVE_PATH = f.SAVE_FOLDER / \
        get_experiment_name(f.MODEL, CONFIG, f.EMBEDDING_TYPE, f.MODEL_NAME)
    print('Save path', f.SAVE_PATH)

    def preprocessed_name(split_type: str) -> str:
        "Gets the pre-processed pickle filename from the configuration."
        name = get_preprocessed_name(split_type, f.MODEL, CONFIG,
                                     f.EMBEDDING_TYPE)
        return cast(str, name)

    # Path to save pre-processed input
    f.TRAIN_PREPROCESSED_NAME = preprocessed_name('train')
    f.VAL_PREPROCESSED_NAME = preprocessed_name('val')
    f.TEST_PREPROCESSED_NAME = preprocessed_name('test')

    f.TRAIN_PREPROCESSED_PATH = EXTERNAL_FOLDER / f.TRAIN_PREPROCESSED_NAME
    f.VAL_PREPROCESSED_PATH = EXTERNAL_FOLDER / f.VAL_PREPROCESSED_NAME
    f.TEST_PREPROCESSED_PATH = EXTERNAL_FOLDER / f.TEST_PREPROCESSED_NAME
    print('Pre-processed data path:', f.TRAIN_PREPROCESSED_PATH)

    # Random seed (for reproducibility)
    f.RANDOM_SEED = 1234

    # Model Configuration
    # Use LSTM, GRU or Transformer
    f.ENCODER_TYPE = sys.argv[6] if len(sys.argv) >= 7 else 'lstm'
    f.WHICH_TRANSFORMER = sys.argv[7] if len(sys.argv) >= 8 else 'allen'
    f.BIDIRECTIONAL = True
    f.RNN_LAYERS = 1
    f.RNN_DROPOUT = 0.5 if f.ENCODER_TYPE != 'transformer' else 0
    f.EMBEDDDING_DROPOUT = 0.5 if f.EMBEDDING_TYPE != 'bert' else 0

    # What encoder to use to join the relation embeddings into a single vector.
    f.RELATION_ENCODER = 'cnn'

    return f
