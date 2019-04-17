"Utility functions for the other modules"
from typing import Tuple, List, Callable, Optional
import pickle
import datetime
import os

import torch
from torch.optim import SGD, Optimizer

from allennlp.models import Model
from allennlp.data import Instance
from allennlp.data.vocabulary import Vocabulary

# Configurable trainer so we don't have to write our own training loop.
from allennlp.training.trainer import Trainer
from allennlp.training.util import evaluate

# Training is done in batches, this creates sorted batches from
# a `DatasetReader`.
from allennlp.data.iterators import BucketIterator

# This is useful if the path is remote. It downloads the file on the first
# time it's ran, and uses the cached result in the next times. If it's a local
# file, it does nothing.
from allennlp.common.file_utils import cached_path
from allennlp.common.params import Params

from allennlp.training.learning_rate_schedulers import LearningRateScheduler

from allennlp.data.token_indexers import (PretrainedBertIndexer,
                                          SingleIdTokenIndexer)
from allennlp.data.dataset_readers import DatasetReader

from reader import McScriptReader


def visualise_model(model: Model) -> None:
    "Print model layers and number of parameters"
    print('#'*5, 'MODEL', '#'*5)
    print(model)

    # Number of parameters
    print()
    print('#'*5, 'PARAMETERS', '#'*5)
    trainable_parameters = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
    print('Trainable parameters:', trainable_parameters)


def example_input(index: int = 0) -> Tuple[str, str, str, str]:
    "Returns an example data tuple. Currently we have tuples 0 and 1."
    examples = [
        "I called to my dog and got the leash off of the hook on the hall . My dog came quickly and I attached his leash to his collar . I put my phone and house keys into my pocket . I walked with my dog to the park across the street from the house and went to the paved walking path . We walked the length of the walking path twice . I listened to my dog to make sure he was n't getting overheated . I greeted people we passed by . I made sure that my dog did not approach anyone who did not want to pet my dog by keeping a firm hold of his leash . Once we completed two laps , we walked back to our house .|why did they lock the door?|Because there was a monster outside.|0",  # NOQA
        "I called to my dog and got the leash off of the hook on the hall . My dog came quickly and I attached his leash to his collar . I put my phone and house keys into my pocket . I walked with my dog to the park across the street from the house and went to the paved walking path . We walked the length of the walking path twice . I listened to my dog to make sure he was n't getting overheated . I greeted people we passed by . I made sure that my dog did not approach anyone who did not want to pet my dog by keeping a firm hold of his leash . Once we completed two laps , we walked back to our house .|why did they lock the door?|Because the dog and owner left for a walk.|1"  # NOQA
    ]
    if index >= len(examples):
        raise IndexError('Example index ({})out of range (we have {} examples)'
                         .format(index, len(examples)))
    passage, question, answer, label = examples[index].split('|')
    return passage, question, answer, label


def is_cuda(model: Model) -> bool:
    "Decide if `model` is hosted on the GPU (True) or CPU (False)"
    return bool(next(model.parameters()).is_cuda)


def train_val_test_split(
        dataset: List[Instance], train_size: float
) -> Tuple[List[Instance], List[Instance], List[Instance]]:
    """
    Split `dataset` into 3 parts: train, validation and test.
    The size of the training set is `train_size`% of the whole dataset.
    The remaining is split in half for validation and test.
    No shuffling is done here.
    """

    train_size = int(train_size * len(dataset))

    val_size = (len(dataset) - train_size) // 2

    train_dataset = dataset[:train_size]
    validation_dataset = dataset[train_size:train_size+val_size]
    test_dataset = dataset[train_size + val_size:]
    return train_dataset, validation_dataset, test_dataset


def create_reader(embedding_type: str = 'bert',
                  conceptnet_path: Optional[str] = None) -> DatasetReader:
    " Creates a new reader and reads from data_path"
    if embedding_type == 'glove':
        word_indexer = SingleIdTokenIndexer(lowercase_tokens=True)
    elif embedding_type == 'bert':
        word_indexer = PretrainedBertIndexer(
            pretrained_model='bert-base-uncased')
    else:
        raise ValueError('Invalid embedding type')

    reader = McScriptReader(conceptnet_path=conceptnet_path,
                            word_indexer=word_indexer)
    return reader


def load_data(reader: Optional[DatasetReader] = None,
              data_path: Optional[str] = None,
              pre_processed_path: Optional[str] = None
              ) -> List[Instance]:
    """
    Load data from file in data_path using McScriptReader.
    A pre-processed pickle will be used if provided.
    The reader is initialised according to the embedding_type (BERT requires
    different indexing from GloVe).
    If provided, ConceptNet tuples are loaded and used to build relations
    in the reader.

    Returns a tuple with the reader used (and properly configured for the data
    read) and the dataset it self.
    """
    # We load pre-processed data to save time (we don't need to tokenise or
    # do parsing/POS-tagging/NER).
    if pre_processed_path is not None and os.path.isfile(pre_processed_path):
        print('>> Reading input from pre-processed file')
        with open(pre_processed_path, 'rb') as preprocessed_file:
            dataset = pickle.load(preprocessed_file)
    else:
        # It shouldn't be, since we don't have a pre-processed file we need
        # to read from the original data.
        if data_path is None or reader is None:
            raise ValueError('Please provide either a pre-processed file or '
                             'a path to the dataset. If the path is provided, '
                             'a DatasetReader is also needed to read it.')
        # Reads from our data. We're used `cached_path`, but data is currently
        # local, so it doesn't really do anything.
        print('>> Reading input from data file')
        dataset = reader.read(cached_path(data_path))
        if pre_processed_path is not None:
            with open(pre_processed_path, 'wb') as preprocessed_file:
                pickle.dump(dataset, preprocessed_file)

    return dataset


def train_model(build_model_fn: Callable[[Vocabulary], Model],
                dataset: List[Instance],
                save_path: Optional[str] = None,
                use_cuda: bool = True,
                batch_size: int = 2,
                patience: int = 10,
                num_epochs: int = 1,
                optimiser_fn: Optional[Callable[[Model], Optimizer]] = None,
                grad_norm_clip: float = 10.0) -> Model:
    "Train and save our baseline model."

    # Splits our dataset into training (80%), validation (10%) and test (10%).
    train_data, val_data, test_data = train_val_test_split(dataset, 0.8)

    # Create a vocabulary from our whole dataset. (for GloVe embeddings)
    vocab = Vocabulary.from_instances(dataset)
    print(vocab)

    model = build_model_fn(vocab)
    visualise_model(model)

    # Next let's check if we have access to a GPU, and use if we want to.
    if torch.cuda.is_available() and use_cuda:
        cuda_device = 0
        # Since we do, we move our model to GPU 0.
        model = model.cuda(cuda_device)
    else:
        # In this case we don't, so we specify -1 to fall back to the CPU.
        # (Where the model already resides.)
        cuda_device = -1

    # We need an optimiser to train the model. This is simple SGD, to which he
    # pass our model's parameter list, and initialise the learning rate.
    if optimiser_fn is None:
        optimiser = SGD(model.parameters(), lr=0.1)
    else:
        optimiser = optimiser_fn(model)

    params = Params(params={
        "type": "reduce_on_plateau",
        "factor": 0.5,
        "patience": 5,
        "verbose": True,
    })
    lr_scheduler = LearningRateScheduler.from_params(optimiser, params)

    # Our trainer needs an iterator to go through our data. This creates
    # batches, sorting them by the number of tokens in each text field, so we
    # have samples with similar number of tokens to minimise padding.
    iterator = BucketIterator(batch_size=batch_size, sorting_keys=[
        ("passage", "num_tokens"),
        ("question", "num_tokens"),
        ("answer0", "num_tokens"),
        ("answer1", "num_tokens")])
    # Our data should be indexed using the vocabulary we learned.
    iterator.index_with(vocab)

    # Initialise the trainer with the paramters we created.
    # Patience is how many epochs without improvement we'll tolerate.
    # We also let the trainer know about CUDA availability.
    if save_path is not None:
        serialization_dir = os.path.join(save_path, 'training')
    trainer = Trainer(model=model,
                      optimizer=optimiser,
                      learning_rate_scheduler=lr_scheduler,
                      iterator=iterator,
                      train_dataset=train_data,
                      validation_dataset=val_data,
                      patience=patience,
                      num_epochs=num_epochs,
                      cuda_device=cuda_device,
                      grad_norm=grad_norm_clip,
                      serialization_dir=serialization_dir,
                      summary_interval=10)

    # Execute training loop.
    trainer.train()

    print()
    print('#'*5, 'EVALUATION', '#'*5)
    metrics = evaluate(model, test_data, iterator, cuda_device, "")
    for key, metric in metrics.items():
        print(key, ':', metric)
    print()

    # To save the model, we need to save the vocabulary and the model weights.
    # Saving weights (model state)
    # TODO: Use `Path` here instead of strings.
    if save_path is not None:
        with open(os.path.join(save_path, 'model.th'), 'wb') as model_file:
            torch.save(model.state_dict(), model_file)
        # Saving vocabulary data (namespaces and tokens)
        with open(os.path.join(save_path, 'vocabulary.pickle'), 'wb') as vfile:
            pickle.dump(vocab, vfile)

    return model


def get_experiment_name(model: str, config: str) -> str:
    """
    Sets up the name for the experiment based on the model type, the
    configuration being used and the current date and time.
    """
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    name = f'{model}.{config}.{date}'
    return name
