# In AllenNLP we use type annotations for just about everything.
from typing import Iterator, List, Dict, Optional

# AllenNLP is built on top of PyTorch, so we use its code freely.
import torch
import torch.optim as optim
import numpy as np

# In AllenNLP we represent each training example as an
# <code>Instance</code> containing <code>Field</code>s of various types.
# Here each example will have a <code>TextField</code> containing the sentence,
# and a <code>SequenceLabelField</code> containing the corresponding part-of-speech tags.
from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField

# Typically to solve a problem like this using AllenNLP,
# you'll have to implement two classes. The first is a
# <a href ="https://allenai.github.io/allennlp-docs/api/allennlp.data.dataset_readers.html">DatasetReader</a>,
# which contains the logic for reading a file of data and producing a stream of <code>Instance</code>s.
from allennlp.data.dataset_readers import DatasetReader

# Frequently we'll want to load datasets or models from URLs.
# The <code>cached_path</code> helper downloads such files,
# caches them locally, and returns the local path. It also
# accepts local file paths (which it just returns as-is).
from allennlp.common.file_utils import cached_path

# There are various ways to represent a word as one or more indices.
# For example, you might maintain a vocabulary of unique words and
# give each word a corresponding id. Or you might have one id per
# character in the word and represent each word as a sequence of ids.
# AllenNLP uses a has a <code>TokenIndexer</code> abstraction for this representation.
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

# Whereas a <code>TokenIndexer</code> represents a rule for
# how to turn a token into indices, a <code>Vocabulary</code>
# contains the corresponding mappings from strings to integers.
# For example, your token indexer might specify to represent a
# token as a sequence of character ids, in which case the
# <code>Vocabulary</code> would contain the mapping {character -> id}.
# In this particular example we use a <code>SingleIdTokenIndexer</code>
# that assigns each token a unique id, and so the <code>Vocabulary</code>
# will just contain a mapping {token -> id} (as well as the reverse mapping).
from allennlp.data.vocabulary import Vocabulary

# Besides <code>DatasetReader</code>, the other class you'll typically
# need to implement is <code>Model</code>, which is a PyTorch <code>Module</code>
# that takes tensor inputs and produces a dict of tensor outputs
# (including the training <code>loss</code> you want to optimize).
from allennlp.models import Model

# As mentioned above, our model will consist of an embedding layer,
# followed by a LSTM, then by a feedforward layer. AllenNLP includes
# abstractions for all of these that smartly handle padding and batching,
# as well as various utility functions.
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask, clone

# We'll want to track accuracy on the training and validation datasets.
from allennlp.training.metrics import CategoricalAccuracy

# In our training we'll need a <code>DataIterator</code>s that can intelligently batch our data.
from allennlp.data.iterators import BucketIterator

# And we'll use AllenNLP's full-featured <code>Trainer</code>.
from allennlp.training.trainer import Trainer

# Finally, we'll want to make predictions on new inputs, more about this below.
from allennlp.predictors import Predictor

from allennlp.common import JsonDict

import re

torch.manual_seed(1)


# Our first order of business is to implement our <code>DatasetReader</code>
# subclass.
class QaDatasetReader(DatasetReader):
    """
    DatasetReader for Question Answering data, one sentence per line, like

        Instance#Question#Answer\tLabel

    Where `Label` is 1 is Answer is correct, 0 otherwise.
    """
    # The only parameter our <code>DatasetReader</code> needs is a dict of
    # <code>TokenIndexer</code>s that specify how to convert tokens into
    # indices.  By default we'll just generate a single index for each token
    # (which we'll call "tokens") that's just a unique id for each distinct
    # token. (This is just the standard "word to index" mapping you'd use in
    # most NLP tasks.)

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer()}

    # <code>DatasetReader.text_to_instance</code> takes the inputs
    # corresponding to a training example (in this case the tokens of the
    # sentence and the corresponding part-of-speech tags), instantiates the
    # corresponding <a
    # (in this case a <code>TextField</code> for the sentence and a
    # <code>SequenceLabelField</code> for its tags), and returns the
    # <code>Instance</code> containing those fields.  Notice that the tags are
    # optional, since we'd like to be able to create instances
    # from unlabeled data to make predictions on them.
    def text_to_instance(self,
                         passage: List[Token],
                         question: List[Token],
                         answer: List[Token],
                         label: Optional[str] = None
                         ) -> Instance:
        fields = {
            "passage": TextField(passage, self.token_indexers),
            "question": TextField(question, self.token_indexers),
            "answer": TextField(answer, self.token_indexers),
        }

        if label is not None:
            fields["label"] = LabelField(label=label)

        return Instance(fields)

    # The other piece we have to implement is <code>_read</code>,
    # which takes a filename and produces a stream of <code>Instance</code>s.
    # Most of the work has already been done in <code>text_to_instance</code>.
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            next(f)  # Skip header line
            for line in f:
                text, label = line.strip().split('\t')

                passage, question, answer = text.split('#')
                passage_tokens = [Token(word) for word in passage.split()]
                question_tokens = [Token(word) for word in question.split()]
                answer_tokens = [Token(word) for word in answer.split()]

                yield self.text_to_instance(passage_tokens, question_tokens,
                                            answer_tokens, label)

# The other class you'll basically always have to implement is
# <code>Model</code>, which is a subclass of <code>torch.nn.Module</code>.  How
# it works is largely up to you, it mostly just needs a <code>forward</code>
# method that takes tensor inputs and produces a dict of tensor outputs that
# includes the loss you'll use to train the model. As mentioned above, our
# model will consist of an embedding layer, a sequence encoder, and a
# feedforward network.


class LstmClassifier(Model):
    # One thing that might seem unusual is that we're going pass in the
    # embedder and the sequence encoder as constructor parameters. This allows
    # us to experiment with different embedders and encoders without having to
    # change the model code.
    def __init__(self,
                 # The embedding layer is specified as an AllenNLP
                 # <code>TextFieldEmbedder</code> which represents a general
                 # way of turning tokens into tensors.  (Here we know that we
                 # want to represent each unique word with a learned tensor,
                 # but using the general class allows us to easily experiment
                 # with different types of embeddings, for example <a href =
                 # "https://allennlp.org/elmo">ELMo</a>.)
                 word_embeddings: TextFieldEmbedder,
                 # Similarly, the encoder is specified as a general
                 # <code>Seq2SeqEncoder</code> even though we know we want to
                 # use an LSTM. Again, this makes it easy to experiment with
                 # other sequence encoders, for example a Transformer.
                 encoder: Seq2VecEncoder,
                 # Every AllenNLP model also expects a <code>Vocabulary</code>,
                 # which contains the namespaced mappings of tokens to indices
                 # and labels to indices.
                 vocab: Vocabulary) -> None:
        # Notice that we have to pass the vocab to the base class constructor.
        super().__init__(vocab)
        self.word_embeddings = word_embeddings

        encoders = clone(encoder, 3)

        self.p_encoder = encoders[0]
        self.q_encoder = encoders[1]
        self.a_encoder = encoders[2]
        # The feed forward layer is not passed in as a parameter, but is
        # constructed by us.  Notice that it looks at the encoder to find the
        # correct input dimension and looks at the vocabulary (and, in
        # particular, at the label -> index mapping) to find the correct output
        # dimension.
        hidden_dim = self.p_encoder.get_output_dim() * 3
        self.hidden2prob = torch.nn.Linear(
            in_features=hidden_dim,
            out_features=vocab.get_vocab_size('label'))
        # The last thing to notice is that we also instantiate a
        # <code>CategoricalAccuracy</code> metric, which we'll use to track
        # accuracy during each training and validation epoch.
        self.accuracy = CategoricalAccuracy()
        self.loss = torch.nn.CrossEntropyLoss()

    # Next we need to implement <code>forward</code>, which is where the actual
    # computation happens. Each <code>Instance</code> in your dataset will get
    # (batched with other instances and) fed into <code>forward</code>.  The
    # <code>forward</code> method expects dicts of tensors as input, and it
    # expects their names to be the names of the fields in your
    # <code>Instance</code>.  In this case we have a sentence field and
    # (possibly) a labels field, so we'll construct our <code>forward</code>
    # accordingly:
    def forward(self,
                passage: Dict[str, torch.Tensor],
                question: Dict[str, torch.Tensor],
                answer: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        # AllenNLP is designed to operate on batched inputs, but different
        # input sequences have different lengths.  Behind the scenes AllenNLP
        # is padding the shorter inputs so that the batch has uniform shape,
        # which means our computations need to use a mask to exclude the
        # padding.  Here we just use the utility function
        # <code>get_text_field_mask</code>, which returns a tensor of 0s and 1s
        # corresponding to the padded and unpadded locations.

        p_mask = get_text_field_mask(passage)
        q_mask = get_text_field_mask(question)
        a_mask = get_text_field_mask(answer)
        # We start by passing the <code>sentence</code> tensor (each sentence a
        # sequence of token ids) to the <code>word_embeddings</code> module,
        # which converts each sentence into a sequence of embedded tensors.
        p_emb = self.word_embeddings(passage)
        q_emb = self.word_embeddings(question)
        a_emb = self.word_embeddings(answer)
        # We next pass the embedded tensors (and the mask) to the LSTM, which
        # produces a sequence of encoded outputs.
        p_enc_out = self.p_encoder(p_emb, p_mask)
        q_enc_out = self.p_encoder(q_emb, q_mask)
        a_enc_out = self.p_encoder(a_emb, a_mask)

        encoder_out = torch.cat((p_enc_out, q_enc_out, a_enc_out), 1)
        # Finally, we pass each encoded output tensor to the feedforward layer
        # to produce logits corresponding to the various tags.
        class_logit = self.hidden2prob(encoder_out)
        output = {"class_logits": class_logit}

        # As before, the labels were optional, as we might want to run this
        # model to make predictions on unlabeled data. If we do have labels,
        # then we use them to update our accuracy metric and compute the "loss"
        # that goes in our output.
        if label is not None:
            self.accuracy(class_logit, label)
            output["loss"] = self.loss(class_logit, label)

        return output

    # We included an accuracy metric that gets updated each forward pass.  That
    # means we need to override a <code>get_metrics</code> method that pulls
    # the data out of it. Behind the scenes, the
    # <code>CategoricalAccuracy</code> metric is storing the number of
    # predictions and the number of correct predictions, updating those counts
    # during each call to forward. Each call to get_metric returns the
    # calculated accuracy and (optionally) resets the counts, which is what
    # allows us to track accuracy anew for each epoch.
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


class BinaryPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict['sentence']
        tokens = [Token(t) for t in re.split(' |#', sentence)]
        return self._dataset_reader.text_to_instance(tokens)


# Now that we've implemented a <code>DatasetReader</code> and
# <code>Model</code>, we're ready to train. We first need an instance of our
# dataset reader.
reader = QaDatasetReader()
# Which we can use to read in the training data and validation data.
# Here we read them in from a URL, but you could read them in from local files
# if your data was local.  We use <code>cached_path</code> to cache the files
# locally (and to hand <code>reader.read</code> the path to the local cached
# version.)
dataset = reader.read(cached_path(
    '../Work/Merging_Data/small.csv'))
train_dataset = dataset[:int(0.8 * len(dataset))]
validation_dataset = dataset[int(0.8 * len(dataset)):]

for i in train_dataset[:3] + validation_dataset[:2]:
    print(i)

# Once we've read in the datasets, we use them to create our
# <code>Vocabulary</code> (that is, the mapping[s] from tokens / labels to
# ids).
vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

# Now we need to construct the model.
# We'll choose a size for our embedding layer and for the hidden layer of our
# LSTM.
EMBEDDING_DIM = 50
HIDDEN_DIM = 100

# For embedding the tokens we'll just use the
# <code>BasicTextFieldEmbedder</code> which takes a mapping from index names to
# embeddings. If you go back to where we defined our
# <code>DatasetReader</code>, the default parameters included a single index
# called "tokens", so our mapping just needs an embedding corresponding to that
# index. We use the <code>Vocabulary</code> to find how many embeddings we need
# and our <code>EMBEDDING_DIM</code> parameter to specify the output dimension.
# It's also possible to start with pre-trained embeddings (for example, GloVe
# vectors), but there's no need to do that on this tiny toy dataset.
print('Vocabsize', vocab.get_vocab_size('tokens'))
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
# We next need to specify the sequence encoder. The need for
# <code>PytorchSeq2SeqWrapper</code> here is slightly unfortunate (and if you
# use <a href =
# "https://github.com/allenai/allennlp/blob/master/tutorials/tagger/README.md#using-config-files">configuration
# files</a> you won't need to worry about it) but here it's required to add
# some extra functionality (and a cleaner interface) to the built-in PyTorch
# module. In AllenNLP we do everything batch first, so we specify that as well.
lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(
    EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

# Finally, we can instantiate the model.
model = LstmClassifier(word_embeddings, lstm, vocab)

# Next let's check if we have access to a GPU.
# if torch.cuda.is_available():
#     cuda_device = 0
#     # Since we do, we move our model to GPU 0.
#     model = model.cuda(cuda_device)
# else:
#     # In this case we don't, so we specify -1 to fall back to the CPU. (Where
#     # the model already resides.)
#     cuda_device = -1
cuda_device = -1

# Now we're ready to train the model. The first thing we'll need is an
# optimizer.  We can just use PyTorch's stochastic gradient descent.
optimizer = optim.SGD(model.parameters(), lr=0.1)

# And we need a <code>DataIterator</code> that handles batching for our
# datasets.  The <code>BucketIterator</code> sorts instances by the specified
# fields in order to create batches with similar sequence lengths.  Here we
# indicate that we want to sort the instances by the number of tokens in the
# sentence field.
iterator = BucketIterator(batch_size=2, sorting_keys=[
    ("passage", "num_tokens"),
    ("question", "num_tokens"),
    ("answer", "num_tokens")])
# We also specify that the iterator should make sure its instances are indexed
# using our vocabulary; that is, that their strings have been converted to
# integers using the mapping we previously created.
iterator.index_with(vocab)

# Now we instantiate our <code>Trainer</code> and run it.  Here we tell it to
# run for 1000 epochs and to stop training early if it ever spends 10 epochs
# without the validation metric improving.  The default validation metric is
# loss (which improves by getting smaller), but it's also possible to specify a
# different metric and direction (e.g. accuracy should get bigger).
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  patience=10,
                  num_epochs=50,
                  cuda_device=cuda_device)

# When we launch it it will print a progress bar for each epoch
# that includes both the "loss" and the "accuracy" metric.
# If our model is good, the loss should go down and the accuracy up as we
# train.
print(model)
trainer.train()

# As in the original PyTorch tutorial, we'd like to look at the predictions our
# model generates.  AllenNLP contains a <code>Predictor</code> abstraction that
# takes inputs, converts them to instances, feeds them through your model, and
# returns JSON-serializable results. Often you'd need to implement your own
# Predictor, but AllenNLP already has a <code>SentenceTaggerPredictor</code>
# that works perfectly here, so we can use it.  It requires our model (for
# making predictions) and a dataset reader (for creating instances).
predictor = BinaryPredictor(model, dataset_reader=reader)
# It has a <code>predict</code> method that just needs a sentence and returns
# (a JSON-serializable version of) the output dict from forward.  Here
# <code>tag_logits</code> will be a (5, 3) array of logits, corresponding to
# the 3 possible tags for each of the 5 words.
# tag_logits = predictor.predict("The dog ate the apple")['tag_logits']
# To get the actual "predictions" we can just take the <code>argmax</code>.
# tag_ids = np.argmax(tag_logits, axis=-1)
# And then use our vocabulary to find the predicted tags.
# print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])

# Finally, we'd like to be able to save our model and reload it later.
# We'll need to save two things. The first is the model weights.
# Here's how to save the model.
with open("/tmp/model.th", 'wb') as f:
    torch.save(model.state_dict(), f)
# And the second is the vocabulary.
vocab.save_to_files("/tmp/vocabulary")

# We only saved the model weights, so we actually have to recreate the same
# model structure using code if we want to reuse them.  First, let's reload the
# vocabulary into a new variable.

# And here's how to reload the model.
vocab2 = Vocabulary.from_files("/tmp/vocabulary")
# And then let's recreate the model (if we were doing this in a different file
# we would of course have to re-instantiate the word embeddings and lstm as
# well).
model2 = LstmClassifier(word_embeddings, lstm, vocab2)
# After which we have to load its state.
with open("/tmp/model.th", 'rb') as f:
    model2.load_state_dict(torch.load(f))
# Here we move the loaded model to the GPU that we used previously. This is
# necessary because we moved <code>word_embeddings</code> and <code>lstm</code>
# with the original model earlier. All of a model's parameters need to be on
# the same device.
if cuda_device > -1:
    model2.cuda(cuda_device)

# And now we should get the same predictions.
predictor2 = BinaryPredictor(model2, dataset_reader=reader)
# tag_logits2 = predictor2.predict("The dog ate the apple")['tag_logits']
# np.testing.assert_array_almost_equal(tag_logits2, tag_logits)
