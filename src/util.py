from typing import Tuple, List, Callable, Optional

import torch
from torch.optim import SGD, Optimizer

from allennlp.models import Model
from allennlp.data import Instance
from allennlp.data.vocabulary import Vocabulary

# Configurable trainer so we don't have to write our own training loop.
from allennlp.training.trainer import Trainer

# Training is done in batches, this creates sorted batches from
# a `DatasetReader`.
from allennlp.data.iterators import BucketIterator

# This is useful if the path is remote. It downloads the file on the first
# time it's ran, and uses the cached result in the next times. If it's a local
# file, it does nothing.
from allennlp.common.file_utils import cached_path

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


def example_input() -> Tuple[str, str, str, str]:
    # Test string
    test = "I called to my dog and got the leash off of the hook on the hall . My dog came quickly and I attached his leash to his collar . I put my phone and house keys into my pocket . I walked with my dog to the park across the street from the house and went to the paved walking path . We walked the length of the walking path twice . I listend to my dog to make sure he was n't getting overheated . I greeted people we passed by . I made sure that my dog did not approach anyone who did not want to pet my dog by keeping a firm hold of his leash . Once we completed two laps , we walked back to our house .|why did they lock the door?|Because there was a monster outside.|0"  # NOQA
    passage, question, answer, label = test.split('|')
    return passage, question, answer, label


def is_cuda(model: Model) -> bool:
    return bool(next(model.parameters()).is_cuda)


def train_val_split(dataset: List[Instance], val_size: float
                    ) -> Tuple[List[Instance], List[Instance]]:
    train_size = int((1 - val_size) * len(dataset))
    train_dataset = dataset[:train_size]
    validation_dataset = dataset[train_size:]
    return train_dataset, validation_dataset


def train_model(build_model_fn: Callable[[Vocabulary], Model],
                data_path: str,
                save_path: Optional[str] = None,
                use_cuda: bool = True,
                batch_size: int = 2,
                patience: int = 10,
                num_epochs: int = 1,
                optimiser: Optional[Optimizer] = None) -> Model:
    "Train and save our baseline model."
    # Creates a new reader
    reader = McScriptReader()
    # Reads from our data. We're used `cached_path`, but data is currently
    # local, so it doesn't really do anything.
    dataset = reader.read(cached_path(data_path))

    # Splits our dataset into training (80%) and validation (20%).
    train_dataset, validation_dataset = train_val_split(dataset, 0.2)

    # Create a vocabulary from our whole dataset.
    vocab = Vocabulary.from_instances(dataset)
    print('Vocabsize', vocab.get_vocab_size('tokens'))

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
    if optimiser is None:
        optimiser = SGD(model.parameters(), lr=0.1)

    # Our trainer needs an iterator to go through our data. This creates
    # batches, sorting them by the number of tokens in each text field, so we
    # have samples with similar number of tokens to minimise padding.
    iterator = BucketIterator(batch_size=batch_size, sorting_keys=[
        ("passage", "num_tokens"),
        ("question", "num_tokens"),
        ("answer", "num_tokens")])
    # Our data should be indexed using the vocabulary we learned.
    iterator.index_with(vocab)

    # Initialise the trainer with the paramters we created.
    # Patience is how many epochs without improvement we'll tolerate.
    # We also let the trainer know about CUDA availability.
    trainer = Trainer(model=model,
                      optimizer=optimiser,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      patience=patience,
                      num_epochs=num_epochs,
                      cuda_device=cuda_device)

    # Execute training loop.
    trainer.train()

    # To save the model, we need to save the vocabulary and the model weights.
    # Saving weights (model state)
    # TODO: Use `Path` here instead of strings.
    if save_path is not None:
        with open(save_path + 'model.th', 'wb') as f:
            torch.save(model.state_dict(), f)
        # Saving vocabulary data (namespaces and tokens)
        vocab.save_to_files(save_path + 'vocabulary')

    return model
