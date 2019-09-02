# literate-lamp

`literate-lamp` is a Natural Language Processing project focused on Question
Answering, targeting the [MCScript](https://arxiv.org/pdf/1803.05223.pdf) 
dataset.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the
required libraries from `requirements.txt`.

It is recommended that you create a virtual environment first. This release
was tested with Python 3.7, but should work with Python 3.6 too.

```bash
pip install -r requirements.txt
```

## Usage

Use `train.py` to train a new model.

```bash
$ ./literate-lamp/train.py -h

usage: train.py [-h] [--config {small,large}] [--model MODEL]
                [--embedding {glove,bert,xlnet}] [--cuda CUDA] [--name NAME]
                [--encoder {lstm,transformer}] [--transformer {allen,custom}]

Train a model.

optional arguments:
  -h, --help            show this help message and exit
  --config {small,large}
                        Configuration to use, small or large.
  --model MODEL         Model to run.
  --embedding {glove,bert,xlnet}
                        Embedding to use, GloVe, BERT or XLNet
  --cuda CUDA           GPU(s) to use. If multiple, separated by comma. If
                        single, just use the gpu number. If CPU is desired,
                        use -1. Examples: --gpu 0,1,2 --gpu 0.
  --name NAME           Name for this model.
  --encoder {lstm,transformer}
                        Encoder type, one of lstm or transformer
  --transformer {allen,custom}
                        If encoder is transformer, choose which one to use,
                        allen or custom.
```

Use `play.py` to play with a trained model. You can use it to perform error
analysis and evaluation.
```bash
$ ./literate-lamp/play.py -h

usage: play.py [-h] --path PATH [--data DATA] [--sample_size SAMPLE_SIZE]
               {evaluate,analysis}

Evaluate or perform error analysis with pre-trained models. If you want to
train a model, use train.py

positional arguments:
  {evaluate,analysis}   Whether to evaluate performance on test set or perform
                        error analysis.

optional arguments:
  -h, --help            show this help message and exit
  --path PATH           Path to the model folder.
  --data DATA           Path to the testing data file.
  --sample_size SAMPLE_SIZE
                        Number of items to draw for error analysis.

```

## License
[GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
