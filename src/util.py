from typing import Tuple, List, Callable, Optional
import pickle
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

# These create text embeddings from a `TextField` input. Since our text data
# is represented using `TextField`s, this makes sense.
# Again, `TextFieldEmbedder` is the abstract class, `BasicTextFieldEmbedder`
# is the implementation (as we're just using simple embeddings, no fancy
# ELMo or BERT so far).
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
# This is the actual neural layer for the embedding. This will be passed into
# the embedder above.
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import (
    PytorchSeq2VecWrapper, Seq2VecEncoder)

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
    return bool(next(model.parameters()).is_cuda)


def train_val_test_split(
        dataset: List[Instance], train_size: float
) -> Tuple[List[Instance], List[Instance], List[Instance]]:

    train_size = int(train_size * len(dataset))
    val_size = (len(dataset) - train_size) // 2
    train_dataset = dataset[:train_size]
    validation_dataset = dataset[train_size:train_size+val_size]
    test_dataset = dataset[train_size + val_size:]
    return train_dataset, validation_dataset, test_dataset


def train_model(build_model_fn: Callable[[Vocabulary], Model],
                data_path: str,
                save_path: Optional[str] = None,
                use_cuda: bool = True,
                batch_size: int = 2,
                patience: int = 10,
                num_epochs: int = 1,
                optimiser: Optional[Optimizer] = None,
                pre_processed_path: Optional[str] = None) -> Model:
    "Train and save our baseline model."

    # We load pre-processed data to save time (we don't need to tokenise or
    # do parsing/POS-tagging/NER).
    if pre_processed_path is not None and os.path.isfile(pre_processed_path):
        with open(pre_processed_path, 'rb') as preprocessed_file:
            dataset = pickle.load(preprocessed_file)
    else:
        # Creates a new reader
        reader = McScriptReader()
        # Reads from our data. We're used `cached_path`, but data is currently
        # local, so it doesn't really do anything.
        dataset = reader.read(cached_path(data_path))
        if pre_processed_path is not None:
            with open(pre_processed_path, 'wb') as preprocessed_file:
                pickle.dump(dataset, preprocessed_file)

    # Splits our dataset into training (80%), validation (10%) and test (10%).
    train_data, val_data, test_data = train_val_test_split(dataset, 0.8)

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
                      train_dataset=train_data,
                      validation_dataset=val_data,
                      patience=patience,
                      num_epochs=num_epochs,
                      cuda_device=cuda_device)

    # Execute training loop.
    trainer.train()

    print('#'*5, 'EVALUATION', '#'*5)
    metrics = evaluate(model, test_data, iterator, cuda_device, "")
    for key, metric in metrics.items():
        print(key, ':', metric)

    # To save the model, we need to save the vocabulary and the model weights.
    # Saving weights (model state)
    # TODO: Use `Path` here instead of strings.
    if save_path is not None:
        with open(save_path + 'model.th', 'wb') as f:
            torch.save(model.state_dict(), f)
        # Saving vocabulary data (namespaces and tokens)
        vocab.save_to_files(save_path + 'vocabulary')

    return model


def glove_embeddings(vocab: Vocabulary, file_path: str, dimension: int
                     ) -> BasicTextFieldEmbedder:
    "Pre-trained embeddings using GloVe"
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=dimension,
                                trainable=False,
                                pretrained_file=file_path)
    # TODO: Not exactly sure how this one works
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    return word_embeddings


def lstm_encoder(input_dim: int, output_dim: int) -> Seq2VecEncoder:
    """
    Our encoder is going to be an LSTM. We have to wrap it for AllenNLP,
    though.
    """
    return PytorchSeq2VecWrapper(torch.nn.LSTM(
        input_dim, output_dim, batch_first=True))
