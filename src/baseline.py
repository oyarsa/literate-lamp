#!/usr/bin/env python3
import torch
import torch.optim as optim
import numpy as np

# This is useful if the path is remote. It downloads the file on the first
# time it's ran, and uses the cached result in the next times. If it's a local
# file, it does nothing.
from allennlp.common.file_utils import cached_path

# Holds the vocabulary, learned from the whole data. Also knows the mapping
# from the `TokenIndexer`, mapping the `Token` to an index in the vocabulary
# and vice-versa.
from allennlp.data.vocabulary import Vocabulary

# These create text embeddings from a `TextField` input. Since our text data
# is represented using `TextField`s, this makes sense.
# Again, `TextFieldEmbedder` is the abstract class, `BasicTextFieldEmbedder`
# is the implementation (as we're just using simple embeddings, no fancy
# ELMo or BERT so far).
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
# This is the actual neural layer for the embedding. This will be passed into
# the embedder above.
from allennlp.modules.token_embedders import Embedding
# `Seq2VecEncoder` is an abstract encoder that takes a sequence and generates
# a vector. This can be an LSTM (although they can also be Seq2Seq if you
# output the hidden state), a Transformer or anything else really, just taking
# NxM -> 1xQ.
# The `PytorchSeq2VecWrapper` is a wrapper for the PyTorch Seq2Vec encoders
# (such as the LSTM we'll use later on), as they don't exactly follow the
# interface the library expects.
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper

# Training is done in batches, this creates sorted batches from
# a `DatasetReader`
from allennlp.data.iterators import BucketIterator

# Configurable trainer so we don't have to write the training loop.
from allennlp.training.trainer import Trainer

from models import BaselineClassifier
from predictor import QaPredictor
from reader import QaDatasetReader

# Manual seeding for reproducibility.
torch.manual_seed(1)

# We're done with the code, now it's just instantiating stuff and running it.

# Path to our dataset
DATA_PATH = './data/small.csv'
# Path to our embeddings
GLOVE_PATH = '../External/glove.840B.300d.txt'
# Size of our embeddings
EMBEDDING_DIM = 300
# Size of our hidden layers (for each encoder)
HIDDEN_DIM = 100

# Creates a new reader
reader = QaDatasetReader()
# Reads from our data. We're used `cached_path`, but data is currently
# local, so it doesn't really do anything.
dataset = reader.read(cached_path(DATA_PATH))

# Splits our dataset into training (80%) and validation (20%).
train_size = int(0.8 * len(dataset))
train_dataset = dataset[:train_size]
validation_dataset = dataset[train_size:]

# Create a vocabulary from our whole dataset.
vocab = Vocabulary.from_instances(dataset)

print('Vocabsize', vocab.get_vocab_size('tokens'))
# Pre-trained embeddings using GloVe
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM,
                            trainable=False,
                            pretrained_file=GLOVE_PATH)
# TODO: Not exactly sure how this one works
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

# Our encoder is going to be an LSTM. We have to wrap it for AllenNLP, though.
lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(
    EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

# Instantiate modele with our embedding, encoder and vocabulary
model = BaselineClassifier(word_embeddings, lstm, vocab)

# Visualise model
print('#'*5, 'MODEL', '#'*5)
print(model)

# Number of parameters
print()
print('#'*5, 'PARAMETERS', '#'*5)
trainable_parameters = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
print('Trainable parameters:', trainable_parameters)

# Next let's check if we have access to a GPU.
if torch.cuda.is_available():
    cuda_device = 0
    # Since we do, we move our model to GPU 0.
    model = model.cuda(cuda_device)
else:
    # In this case we don't, so we specify -1 to fall back to the CPU. (Where
    # the model already resides.)
    cuda_device = -1

# We need an optimiser to train the model. This is simple SGD, to which he
# pass our model's paramter list, and initialise the learning rate.
optimiser = optim.SGD(model.parameters(), lr=0.1)

# Our trainer needs an iterator to go through our data. This creates batches,
# sorting them by the number of tokens in each text field, so we have
# samples with similar number of tokens to minimise padding.
iterator = BucketIterator(batch_size=2, sorting_keys=[
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
                  # patience=10,
                  num_epochs=1,
                  cuda_device=cuda_device)

# Execute training loop.
trainer.train()

# Create a predictor to run our model and get predictions.
predictor = QaPredictor(model, dataset_reader=reader)
# Test string
test = "I called to my dog and got the leash off of the hook on the hall . My dog came quickly and I attached his leash to his collar . I put my phone and house keys into my pocket . I walked with my dog to the park across the street from the house and went to the paved walking path . We walked the length of the walking path twice . I listend to my dog to make sure he was n't getting overheated . I greeted people we passed by . I made sure that my dog did not approach anyone who did not want to pet my dog by keeping a firm hold of his leash . Once we completed two laps , we walked back to our house .|why did they lock the door?|Because there was a monster outside.|0"  # NOQA
passage, question, answer, label = test.split('|')
# Execute prediction. Gets output dict from the model.
prediction = predictor.predict(
    passage=passage,
    question=question,
    answer=answer
)
# Predicted class
class_ = prediction['class']
print('Label:', label, '-- Predicted:', class_)

# To save the model, we need to save the vocabulary and the model weights.
# Saving weights (model state)
with open("/tmp/model.th", 'wb') as f:
    torch.save(model.state_dict(), f)
# Saving vocabulary data (namespaces and tokens)
vocab.save_to_files("/tmp/vocabulary")

# Reload vocabulary
vocab2 = Vocabulary.from_files("/tmp/vocabulary")
# Recreate the model. Normally we'd need to reinstantiate the embeddings
# and the LSTM too.
model2 = BaselineClassifier(word_embeddings, lstm, vocab2)
# Load the state from the file
with open("/tmp/model.th", 'rb') as f:
    model2.load_state_dict(torch.load(f))
# We now have the loaded model. Let's move it to the GPU again if available.
if cuda_device > -1:
    model2.cuda(cuda_device)

# Try predicting again and see if we get the same results (we should).
predictor2 = QaPredictor(model2, dataset_reader=reader)
prediction2 = predictor2.predict(
    passage=passage,
    question=question,
    answer=answer
)
np.testing.assert_array_almost_equal(
    prediction['logits'], prediction2['logits'])
