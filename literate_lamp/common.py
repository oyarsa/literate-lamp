from typing import Callable, Tuple

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model

import args
import models
import readers
from layers import (lstm_encoder, gru_encoder, lstm_seq2seq, gru_seq2seq,
                    glove_embeddings, learned_embeddings, bert_embeddings,
                    transformer_seq2seq, xlnet_embeddings,
                    RelationalTransformerEncoder)
ARGS = args.get_args()


def get_word_embeddings(vocabulary: Vocabulary) -> TextFieldEmbedder:
    "Instatiates the word embeddings based on config."
    if ARGS.EMBEDDING_TYPE == 'glove':
        return glove_embeddings(vocabulary, ARGS.GLOVE_PATH,
                                ARGS.GLOVE_EMBEDDING_DIM, training=True)
    if ARGS.EMBEDDING_TYPE == 'bert':
        return bert_embeddings(pretrained_model=ARGS.BERT_PATH,
                               training=ARGS.finetune_embeddings)
    if ARGS.EMBEDDING_TYPE == 'xlnet':
        return xlnet_embeddings(config_path=ARGS.xlnet_config_path,
                                model_path=ARGS.xlnet_model_path,
                                training=ARGS.finetune_embeddings,
                                window_size=ARGS.xlnet_window_size)
    raise ValueError(
        f'Invalid word embedding type: {ARGS.EMBEDDING_TYPE}')


def build_relational_xl(vocabulary: Vocabulary) -> Model:
    """
    Builds the RelationalXL.

    Parameters
    ---------
    vocab : Vocabulary built from the problem dataset.

    Returns
    -------
    A `RelationalXL` model ready to be trained.
    """
    word_embeddings = get_word_embeddings(vocabulary)
    embedding_dim = word_embeddings.get_output_dim()

    if ARGS.ENCODER_TYPE == 'lstm':
        seq_fn = lstm_seq2seq
        encoder_fn = lstm_encoder
    elif ARGS.ENCODER_TYPE == 'gru':
        seq_fn = gru_seq2seq
        encoder_fn = gru_encoder
    else:
        raise ValueError('Invalid RNN type')

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = ARGS.RNN_DROPOUT if ARGS.RNN_LAYERS > 1 else 0

    if ARGS.ENCODER_TYPE in ['lstm', 'gru']:
        relation_encoder = encoder_fn(input_dim=embedding_dim,
                                      output_dim=ARGS.HIDDEN_DIM,
                                      num_layers=ARGS.RNN_LAYERS,
                                      bidirectional=ARGS.BIDIRECTIONAL,
                                      dropout=dropout)
        text_encoder = seq_fn(input_dim=embedding_dim,
                              output_dim=ARGS.HIDDEN_DIM,
                              num_layers=ARGS.RNN_LAYERS,
                              bidirectional=ARGS.BIDIRECTIONAL,
                              dropout=dropout)

    # Instantiate modele with our embedding, encoder and vocabulary
    model = models.RelationalXL(
        word_embeddings=word_embeddings,
        text_encoder=text_encoder,
        relation_encoder=relation_encoder,
        vocab=vocabulary,
        encoder_dropout=0.5
    )

    return model


def build_advanced_xlnet(vocab: Vocabulary) -> Model:
    """
    Builds the AdvancedXLNetClassifier.

    Parameters
    ---------
    vocab : Vocabulary built from the problem dataset.

    Returns
    -------
    A `AdvancedXLNetClassifier` model ready to be trained.
    """
    word_embeddings = get_word_embeddings(vocab)
    if ARGS.ENCODER_TYPE == 'lstm':
        encoder_fn = lstm_seq2seq
    elif ARGS.ENCODER_TYPE == 'gru':
        encoder_fn = gru_seq2seq
    else:
        raise ValueError('Invalid RNN type')

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = ARGS.RNN_DROPOUT if ARGS.RNN_LAYERS > 1 else 0

    if ARGS.ENCODER_TYPE in ['lstm', 'gru']:
        encoder = encoder_fn(input_dim=768,
                             output_dim=ARGS.HIDDEN_DIM,
                             num_layers=ARGS.RNN_LAYERS,
                             bidirectional=ARGS.BIDIRECTIONAL,
                             dropout=dropout)

    # Instantiate modele with our embedding, encoder and vocabulary
    model = models.AdvancedXLNetClassifier(
        word_embeddings=word_embeddings,
        encoder=encoder,
        vocab=vocab,
        encoder_dropout=0.5,
        train_xlnet=ARGS.finetune_embeddings
    )

    return model


def build_simple_xlnet(vocabulary: Vocabulary) -> Model:
    return models.SimpleXLNetClassifier(
        vocab=vocabulary,
        config_path=ARGS.xlnet_config_path,
        model_path=ARGS.xlnet_model_path,
        train_xlnet=ARGS.finetune_embeddings
    )


def build_dcmn(vocabulary: Vocabulary) -> Model:
    """
    Builds the DCMN.

    Parameters
    ---------
    vocabulary : Vocabulary built from the problem dataset.

    Returns
    -------
    A `DCMN` model ready to be trained.
    """
    word_embeddings = get_word_embeddings(vocabulary)

    model = models.Dcmn(
        word_embeddings=word_embeddings,
        vocab=vocabulary,
        embedding_dropout=ARGS.RNN_DROPOUT
    )

    return model


def build_rel_han(vocab: Vocabulary) -> Model:
    """
    Builds the RelationHan.

    Parameters
    ---------
    vocab : Vocabulary built from the problem dataset.

    Returns
    -------
    A `RelationHan` model ready to be trained.
    """
    word_embeddings = get_word_embeddings(vocab)

    if ARGS.ENCODER_TYPE == 'lstm':
        encoder_fn = lstm_seq2seq
    elif ARGS.ENCODER_TYPE == 'gru':
        encoder_fn = gru_seq2seq
    elif ARGS.ENCODER_TYPE == 'transformer':
        # Transformer has to be handled differently, but the 2 RNNs can share
        pass
    else:
        raise ValueError('Invalid RNN type')

    embedding_dim = word_embeddings.get_output_dim()

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = ARGS.RNN_DROPOUT if ARGS.RNN_LAYERS > 1 else 0

    if ARGS.ENCODER_TYPE in ['lstm', 'gru']:
        sentence_encoder = encoder_fn(input_dim=embedding_dim,
                                      output_dim=ARGS.HIDDEN_DIM,
                                      num_layers=ARGS.RNN_LAYERS,
                                      bidirectional=ARGS.BIDIRECTIONAL,
                                      dropout=dropout)
        document_encoder = encoder_fn(
            input_dim=sentence_encoder.get_output_dim(),
            output_dim=ARGS.HIDDEN_DIM,
            num_layers=1,
            bidirectional=ARGS.BIDIRECTIONAL,
            dropout=dropout)
        relation_encoder = encoder_fn(
            input_dim=sentence_encoder.get_output_dim(),
            output_dim=ARGS.HIDDEN_DIM,
            num_layers=1,
            bidirectional=ARGS.BIDIRECTIONAL,
            dropout=dropout)
    elif ARGS.ENCODER_TYPE == 'transformer':
        sentence_encoder = transformer_seq2seq(
            input_dim=embedding_dim,
            model_dim=ARGS.TRANSFORMER_DIM,
            num_layers=6,
            num_attention_heads=4,
            feedforward_hidden_dim=ARGS.TRANSFORMER_DIM,
            ttype=ARGS.WHICH_TRANSFORMER
        )
        document_encoder = transformer_seq2seq(
            input_dim=sentence_encoder.get_output_dim(),
            model_dim=ARGS.TRANSFORMER_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=ARGS.TRANSFORMER_DIM,
            ttype=ARGS.WHICH_TRANSFORMER
        )
        relation_encoder = transformer_seq2seq(
            input_dim=sentence_encoder.get_output_dim(),
            model_dim=ARGS.TRANSFORMER_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=ARGS.TRANSFORMER_DIM,
            ttype=ARGS.WHICH_TRANSFORMER
        )

    document_relation_encoder = RelationalTransformerEncoder(
        src_input_dim=sentence_encoder.get_output_dim(),
        kb_input_dim=relation_encoder.get_output_dim(),
        model_dim=ARGS.TRANSFORMER_DIM,
        feedforward_hidden_dim=ARGS.TRANSFORMER_DIM,
        num_layers=3,
        num_attention_heads=8,
        dropout_prob=0.1,
        return_kb=False
    )

    # Instantiate model with our embedding, encoder and vocabulary
    model = models.RelationalHan(
        relation_encoder=relation_encoder,
        document_relation_encoder=document_relation_encoder,
        word_embeddings=word_embeddings,
        sentence_encoder=sentence_encoder,
        document_encoder=document_encoder,
        vocab=vocab,
        encoder_dropout=ARGS.RNN_DROPOUT
    )

    return model


def build_relational_transformer(vocab: Vocabulary) -> Model:
    """
    Builds the RelationalTransformerModel.

    Parameters
    ---------
    vocab : Vocabulary built from the problem dataset.

    Returns
    -------
    A `RelationalTransformerModel` model ready to be trained.
    """
    word_embeddings = get_word_embeddings(vocab)
    rel_embeddings = word_embeddings

    if ARGS.ENCODER_TYPE == 'lstm':
        encoder_fn = lstm_seq2seq
    elif ARGS.ENCODER_TYPE == 'gru':
        encoder_fn = gru_seq2seq
    elif ARGS.ENCODER_TYPE == 'transformer':
        # Transformer has to be handled differently, but the 2 RNNs can share
        pass
    else:
        raise ValueError('Invalid RNN type')

    embedding_dim = word_embeddings.get_output_dim()

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = ARGS.RNN_DROPOUT if ARGS.RNN_LAYERS > 1 else 0

    if ARGS.ENCODER_TYPE in ['lstm', 'gru']:
        sentence_encoder = encoder_fn(input_dim=embedding_dim,
                                      output_dim=ARGS.HIDDEN_DIM,
                                      num_layers=ARGS.RNN_LAYERS,
                                      bidirectional=ARGS.BIDIRECTIONAL,
                                      dropout=dropout)
        relation_sentence_encoder = encoder_fn(
            input_dim=embedding_dim,
            output_dim=ARGS.HIDDEN_DIM,
            num_layers=ARGS.RNN_LAYERS,
            bidirectional=ARGS.BIDIRECTIONAL,
            dropout=dropout)
    elif ARGS.ENCODER_TYPE == 'transformer':
        sentence_encoder = transformer_seq2seq(
            input_dim=embedding_dim,
            model_dim=ARGS.TRANSFORMER_DIM,
            num_layers=6,
            num_attention_heads=4,
            feedforward_hidden_dim=ARGS.TRANSFORMER_DIM,
            ttype=ARGS.WHICH_TRANSFORMER
        )
        relation_sentence_encoder = transformer_seq2seq(
            input_dim=embedding_dim,
            model_dim=ARGS.TRANSFORMER_DIM,
            num_layers=6,
            num_attention_heads=4,
            feedforward_hidden_dim=ARGS.TRANSFORMER_DIM,
            ttype=ARGS.WHICH_TRANSFORMER
        )

    relational_encoder = RelationalTransformerEncoder(
        src_input_dim=sentence_encoder.get_output_dim(),
        kb_input_dim=relation_sentence_encoder.get_output_dim(),
        model_dim=ARGS.TRANSFORMER_DIM,
        feedforward_hidden_dim=ARGS.TRANSFORMER_DIM,
        num_layers=3,
        num_attention_heads=8,
        dropout_prob=0.1,
        return_kb=False
    )

    # Instantiate modele with our embedding, encoder and vocabulary
    model = models.RelationalTransformerModel(
        word_embeddings=word_embeddings,
        sentence_encoder=sentence_encoder,
        relational_encoder=relational_encoder,
        relation_sentence_encoder=relation_sentence_encoder,
        rel_embeddings=rel_embeddings,
        vocab=vocab,
        encoder_dropout=ARGS.RNN_DROPOUT
    )

    return model


def build_hierarchical_attn_net(vocab: Vocabulary) -> Model:
    """
    Builds the HierarchicalAttentionNetwork.

    Parameters
    ---------
    vocab : Vocabulary built from the problem dataset.

    Returns
    -------
    A `HierarchicalAttentionNetwork` model ready to be trained.
    """
    word_embeddings = get_word_embeddings(vocab)

    if ARGS.ENCODER_TYPE == 'lstm':
        encoder_fn = lstm_seq2seq
    elif ARGS.ENCODER_TYPE == 'gru':
        encoder_fn = gru_seq2seq
    elif ARGS.ENCODER_TYPE == 'transformer':
        # Transformer has to be handled differently, but the 2 RNNs can share
        pass
    else:
        raise ValueError('Invalid RNN type')

    embedding_dim = word_embeddings.get_output_dim()

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = ARGS.RNN_DROPOUT if ARGS.RNN_LAYERS > 1 else 0

    if ARGS.ENCODER_TYPE in ['lstm', 'gru']:
        sentence_encoder = encoder_fn(input_dim=embedding_dim,
                                      output_dim=ARGS.HIDDEN_DIM,
                                      num_layers=ARGS.RNN_LAYERS,
                                      bidirectional=ARGS.BIDIRECTIONAL,
                                      dropout=dropout)
        document_encoder = encoder_fn(
            input_dim=sentence_encoder.get_output_dim(),
            output_dim=ARGS.HIDDEN_DIM,
            num_layers=1,
            bidirectional=ARGS.BIDIRECTIONAL,
            dropout=dropout)
    elif ARGS.ENCODER_TYPE == 'transformer':
        sentence_encoder = transformer_seq2seq(
            input_dim=embedding_dim,
            model_dim=ARGS.TRANSFORMER_DIM,
            num_layers=6,
            num_attention_heads=4,
            feedforward_hidden_dim=ARGS.TRANSFORMER_DIM,
            ttype=ARGS.WHICH_TRANSFORMER
        )
        document_encoder = transformer_seq2seq(
            input_dim=sentence_encoder.get_output_dim(),
            model_dim=ARGS.TRANSFORMER_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=ARGS.TRANSFORMER_DIM,
            ttype=ARGS.WHICH_TRANSFORMER
        )

    # Instantiate modele with our embedding, encoder and vocabulary
    model = models.HierarchicalAttentionNetwork(
        word_embeddings=word_embeddings,
        sentence_encoder=sentence_encoder,
        document_encoder=document_encoder,
        vocab=vocab,
        encoder_dropout=ARGS.RNN_DROPOUT
    )

    return model


def build_advanced_attn_bert(vocab: Vocabulary) -> Model:
    """
    Builds the AdvancedBertClassifier.

    Parameters
    ---------
    vocab : Vocabulary built from the problem dataset.

    Returns
    -------
    A `AdvancedBertClassifier` model ready to be trained.
    """
    if ARGS.ENCODER_TYPE == 'lstm':
        encoder_fn = lstm_seq2seq
    elif ARGS.ENCODER_TYPE == 'gru':
        encoder_fn = gru_seq2seq
    elif ARGS.ENCODER_TYPE == 'transformer':
        # Transformer has to be handled differently, but the 2 RNNs can share
        pass
    else:
        raise ValueError('Invalid RNN type')

    hidden_dim = 100

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = ARGS.RNN_DROPOUT if ARGS.RNN_LAYERS > 1 else 0

    if ARGS.ENCODER_TYPE in ['lstm', 'gru']:
        encoder = encoder_fn(input_dim=hidden_dim,
                             output_dim=ARGS.HIDDEN_DIM,
                             num_layers=ARGS.RNN_LAYERS,
                             bidirectional=ARGS.BIDIRECTIONAL,
                             dropout=dropout)
    elif ARGS.ENCODER_TYPE == 'transformer':
        encoder = transformer_seq2seq(
            input_dim=hidden_dim,
            model_dim=ARGS.HIDDEN_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=1024
        )

    # Instantiate modele with our embedding, encoder and vocabulary
    model = models.AdvancedAttentionBertClassifier(
        bert_path=ARGS.BERT_PATH,
        encoder=encoder,
        vocab=vocab,
        encoder_dropout=0,
        hidden_dim=hidden_dim
    )

    return model


def build_hierarchical_bert(vocab: Vocabulary) -> Model:
    """
    Builds the HierarchicalBert.

    Parameters
    ---------
    vocab : Vocabulary built from the problem dataset.

    Returns
    -------
    A `HierarchicalBert` model ready to be trained.
    """
    if ARGS.ENCODER_TYPE == 'lstm':
        encoder_fn = lstm_encoder
    elif ARGS.ENCODER_TYPE == 'gru':
        encoder_fn = gru_encoder
    else:
        raise ValueError('Invalid RNN type')

    bert = bert_embeddings(ARGS.BERT_PATH)
    embedding_dim = bert.get_output_dim()

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = ARGS.RNN_DROPOUT if ARGS.RNN_LAYERS > 1 else 0

    if ARGS.ENCODER_TYPE in ['lstm', 'gru']:
        sentence_encoder = encoder_fn(input_dim=embedding_dim,
                                      output_dim=ARGS.HIDDEN_DIM,
                                      num_layers=ARGS.RNN_LAYERS,
                                      bidirectional=ARGS.BIDIRECTIONAL,
                                      dropout=dropout)
        document_encoder = encoder_fn(
            input_dim=sentence_encoder.get_output_dim(),
            output_dim=ARGS.HIDDEN_DIM,
            num_layers=1,
            bidirectional=ARGS.BIDIRECTIONAL,
            dropout=dropout)

    # Instantiate modele with our embedding, encoder and vocabulary
    model = models.HierarchicalBert(
        bert_path=ARGS.BERT_PATH,
        sentence_encoder=sentence_encoder,
        document_encoder=document_encoder,
        vocab=vocab,
        encoder_dropout=ARGS.RNN_DROPOUT
    )

    return model


def build_zero_trian(vocab: Vocabulary) -> Model:
    """
    Builds the ZeroTriAN classifier without the extra features (NER, POS, HC)
    nor the relations..

    Parameters
    ---------
    vocab : Vocabulary built from the problem dataset.

    Returns
    -------
    A `ZeroTrian` model ready to be trained.
    """
    word_embeddings = get_word_embeddings(vocab)

    if ARGS.ENCODER_TYPE == 'lstm':
        encoder_fn = lstm_seq2seq
    elif ARGS.ENCODER_TYPE == 'gru':
        encoder_fn = gru_seq2seq
    elif ARGS.ENCODER_TYPE == 'transformer':
        # Transformer has to be handled differently, but the two RNNs can share
        pass
    else:
        raise ValueError('Invalid RNN type')

    embedding_dim = word_embeddings.get_output_dim()

    # p_emb + p_q_weighted + p_q_rel + 2*p_a_rel
    p_input_size = 2*embedding_dim
    # q_emb
    q_input_size = embedding_dim
    # a_emb + a_q_match + a_p_match
    a_input_size = 3 * embedding_dim

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = ARGS.RNN_DROPOUT if ARGS.RNN_LAYERS > 1 else 0

    if ARGS.ENCODER_TYPE in ['lstm', 'gru']:
        p_encoder = encoder_fn(input_dim=p_input_size,
                               output_dim=ARGS.HIDDEN_DIM,
                               num_layers=ARGS.RNN_LAYERS,
                               bidirectional=ARGS.BIDIRECTIONAL,
                               dropout=dropout)
        q_encoder = encoder_fn(input_dim=q_input_size,
                               output_dim=ARGS.HIDDEN_DIM,
                               num_layers=1,
                               bidirectional=ARGS.BIDIRECTIONAL)
        a_encoder = encoder_fn(input_dim=a_input_size,
                               output_dim=ARGS.HIDDEN_DIM,
                               num_layers=1,
                               bidirectional=ARGS.BIDIRECTIONAL)
    elif ARGS.ENCODER_TYPE == 'transformer':
        p_encoder = transformer_seq2seq(
            input_dim=p_input_size,
            model_dim=ARGS.HIDDEN_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=1024
        )
        q_encoder = transformer_seq2seq(
            input_dim=q_input_size,
            model_dim=ARGS.HIDDEN_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=512
        )
        a_encoder = transformer_seq2seq(
            input_dim=a_input_size,
            model_dim=ARGS.HIDDEN_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=512
        )

    # Instantiate modele with our embedding, encoder and vocabulary
    model = models.ZeroTrian(
        word_embeddings=word_embeddings,
        p_encoder=p_encoder,
        q_encoder=q_encoder,
        a_encoder=a_encoder,
        vocab=vocab,
        embedding_dropout=0,
        encoder_dropout=ARGS.RNN_DROPOUT
    )

    return model


def build_simple_trian(vocab: Vocabulary) -> Model:
    """
    Builds the TriAN classifier without the extra features (NER, POS, HC).

    Parameters
    ---------
    vocab : Vocabulary built from the problem dataset.

    Returns
    -------
    A `Trian` model ready to be trained.
    """
    word_embeddings = get_word_embeddings(vocab)
    rel_embeddings = learned_embeddings(vocab, ARGS.REL_EMBEDDING_DIM,
                                        'tokens')

    if ARGS.ENCODER_TYPE == 'lstm':
        encoder_fn = lstm_seq2seq
    elif ARGS.ENCODER_TYPE == 'gru':
        encoder_fn = gru_seq2seq
    elif ARGS.ENCODER_TYPE == 'transformer':
        # Transformer has to be handled differently, but the two RNNs can share
        pass
    else:
        raise ValueError('Invalid RNN type')

    embedding_dim = word_embeddings.get_output_dim()

    # p_emb + p_q_weighted + p_q_rel + 2*p_a_rel
    p_input_size = (2*embedding_dim + + 3*ARGS.REL_EMBEDDING_DIM)
    # q_emb
    q_input_size = embedding_dim
    # a_emb + a_q_match + a_p_match
    a_input_size = 3 * embedding_dim

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = ARGS.RNN_DROPOUT if ARGS.RNN_LAYERS > 1 else 0

    if ARGS.ENCODER_TYPE in ['lstm', 'gru']:
        p_encoder = encoder_fn(input_dim=p_input_size,
                               output_dim=ARGS.HIDDEN_DIM,
                               num_layers=ARGS.RNN_LAYERS,
                               bidirectional=ARGS.BIDIRECTIONAL,
                               dropout=dropout)
        q_encoder = encoder_fn(input_dim=q_input_size,
                               output_dim=ARGS.HIDDEN_DIM,
                               num_layers=1,
                               bidirectional=ARGS.BIDIRECTIONAL)
        a_encoder = encoder_fn(input_dim=a_input_size,
                               output_dim=ARGS.HIDDEN_DIM,
                               num_layers=1,
                               bidirectional=ARGS.BIDIRECTIONAL)
    elif ARGS.ENCODER_TYPE == 'transformer':
        p_encoder = transformer_seq2seq(
            input_dim=p_input_size,
            model_dim=ARGS.HIDDEN_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=1024
        )
        q_encoder = transformer_seq2seq(
            input_dim=q_input_size,
            model_dim=ARGS.HIDDEN_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=512
        )
        a_encoder = transformer_seq2seq(
            input_dim=a_input_size,
            model_dim=ARGS.HIDDEN_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=512
        )

    # Instantiate modele with our embedding, encoder and vocabulary
    model = models.SimpleTrian(
        word_embeddings=word_embeddings,
        rel_embeddings=rel_embeddings,
        p_encoder=p_encoder,
        q_encoder=q_encoder,
        a_encoder=a_encoder,
        vocab=vocab,
        embedding_dropout=0,
        encoder_dropout=ARGS.RNN_DROPOUT
    )

    return model


def build_advanced_bert(vocab: Vocabulary) -> Model:
    """
    Builds the AdvancedBertClassifier.

    Parameters
    ---------
    vocab : Vocabulary built from the problem dataset.

    Returns
    -------
    A `AdvancedBertClassifier` model ready to be trained.
    """

    if ARGS.ENCODER_TYPE == 'lstm':
        encoder_fn = lstm_encoder
    elif ARGS.ENCODER_TYPE == 'gru':
        encoder_fn = gru_encoder
    else:
        raise ValueError('Invalid RNN type')

    hidden_dim = 100

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = ARGS.RNN_DROPOUT if ARGS.RNN_LAYERS > 1 else 0

    if ARGS.ENCODER_TYPE in ['lstm', 'gru']:
        encoder = encoder_fn(input_dim=hidden_dim,
                             output_dim=ARGS.HIDDEN_DIM,
                             num_layers=ARGS.RNN_LAYERS,
                             bidirectional=ARGS.BIDIRECTIONAL,
                             dropout=dropout)

    # Instantiate modele with our embedding, encoder and vocabulary
    model = models.AdvancedBertClassifier(
        bert_path=ARGS.BERT_PATH,
        encoder=encoder,
        vocab=vocab,
        encoder_dropout=0,
        hidden_dim=hidden_dim
    )

    return model


def build_simple_bert(vocab: Vocabulary) -> Model:
    """
    Builds the simple BERT-Based classifier.

    Parameters
    ---------
    vocab : Vocabulary built from the problem dataset.

    Returns
    -------
    A `SimpleBertClassifier` model ready to be trained.
    """
    model = models.SimpleBertClassifier(bert_path=ARGS.BERT_PATH, vocab=vocab)
    return model


def build_baseline(vocab: Vocabulary) -> Model:
    """
    Builds the Baseline classifier using Glove embeddings and RNN encoders.

    Parameters
    ---------
    vocab : Vocabulary built from the problem dataset.

    Returns
    -------
    A `BaselineClassifier` model ready to be trained.
    """
    embeddings = get_word_embeddings(vocab)

    if ARGS.ENCODER_TYPE == 'gru':
        encoder_fn = gru_encoder
    else:
        encoder_fn = lstm_encoder

    embedding_dim = embeddings.get_output_dim()
    encoder = encoder_fn(embedding_dim, ARGS.HIDDEN_DIM,
                         num_layers=ARGS.RNN_LAYERS,
                         bidirectional=ARGS.BIDIRECTIONAL)

    model = models.BaselineClassifier(embeddings, encoder, vocab)
    return model


def build_attentive_reader(vocab: Vocabulary) -> Model:
    """
    Builds the Attentive Reader using Glove embeddings and GRU encoders.

    Parameters
    ---------
    vocab : Vocabulary built from the problem dataset.

    Returns
    -------
    A `AttentiveReader` model ready to be trained.
    """
    embeddings = get_word_embeddings(vocab)

    embedding_dim = embeddings.get_output_dim()
    p_encoder = gru_seq2seq(embedding_dim, ARGS.HIDDEN_DIM,
                            num_layers=ARGS.RNN_LAYERS,
                            bidirectional=ARGS.BIDIRECTIONAL)
    q_encoder = gru_encoder(embedding_dim, ARGS.HIDDEN_DIM, num_layers=1,
                            bidirectional=ARGS.BIDIRECTIONAL)
    a_encoder = gru_encoder(embedding_dim, ARGS.HIDDEN_DIM, num_layers=1,
                            bidirectional=ARGS.BIDIRECTIONAL)

    model = models.AttentiveReader(
        embeddings, p_encoder, q_encoder, a_encoder, vocab
    )
    return model


def build_trian(vocab: Vocabulary) -> Model:
    """
    Builds the TriAN classifier using Glove embeddings and RNN encoders.

    Parameters
    ---------
    vocab : Vocabulary built from the problem dataset.

    Returns
    -------
    A `Trian` model ready to be trained.
    """
    word_embeddings = get_word_embeddings(vocab)
    pos_embeddings = learned_embeddings(vocab, ARGS.POS_EMBEDDING_DIM,
                                        'pos_tokens')
    ner_embeddings = learned_embeddings(vocab, ARGS.NER_EMBEDDING_DIM,
                                        'ner_tokens')
    rel_embeddings = learned_embeddings(vocab, ARGS.REL_EMBEDDING_DIM,
                                        'rel_tokens')

    if ARGS.ENCODER_TYPE == 'lstm':
        encoder_fn = lstm_seq2seq
    elif ARGS.ENCODER_TYPE == 'gru':
        encoder_fn = gru_seq2seq
    elif ARGS.ENCODER_TYPE == 'transformer':
        # Transformer has to be handled differently, but the two RNNs can share
        pass
    else:
        raise ValueError('Invalid RNN type')

    embedding_dim = word_embeddings.get_output_dim()

    # p_emb + p_q_weighted + p_pos_emb + p_ner_emb + p_q_rel + 2*p_a_rel
    #       + hc_feat
    p_input_size = (2*embedding_dim + ARGS.POS_EMBEDDING_DIM
                    + ARGS.NER_EMBEDDING_DIM + 3*ARGS.REL_EMBEDDING_DIM
                    + ARGS.HANDCRAFTED_DIM)
    # q_emb + q_pos_emb
    q_input_size = embedding_dim + ARGS.POS_EMBEDDING_DIM
    # a_emb + a_q_match + a_p_match
    a_input_size = 3 * embedding_dim

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = ARGS.RNN_DROPOUT if ARGS.RNN_LAYERS > 1 else 0

    if ARGS.ENCODER_TYPE in ['lstm', 'gru']:
        p_encoder = encoder_fn(input_dim=p_input_size,
                               output_dim=ARGS.HIDDEN_DIM,
                               num_layers=ARGS.RNN_LAYERS,
                               bidirectional=ARGS.BIDIRECTIONAL,
                               dropout=dropout)
        q_encoder = encoder_fn(input_dim=q_input_size,
                               output_dim=ARGS.HIDDEN_DIM,
                               num_layers=1,
                               bidirectional=ARGS.BIDIRECTIONAL)
        a_encoder = encoder_fn(input_dim=a_input_size,
                               output_dim=ARGS.HIDDEN_DIM,
                               num_layers=1,
                               bidirectional=ARGS.BIDIRECTIONAL)
    elif ARGS.ENCODER_TYPE == 'transformer':
        p_encoder = transformer_seq2seq(
            input_dim=p_input_size,
            model_dim=ARGS.HIDDEN_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=1024
        )
        q_encoder = transformer_seq2seq(
            input_dim=q_input_size,
            model_dim=ARGS.HIDDEN_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=512
        )
        a_encoder = transformer_seq2seq(
            input_dim=a_input_size,
            model_dim=ARGS.HIDDEN_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=512
        )

    # Instantiate modele with our embedding, encoder and vocabulary
    model = models.Trian(
        word_embeddings=word_embeddings,
        rel_embeddings=rel_embeddings,
        pos_embeddings=pos_embeddings,
        ner_embeddings=ner_embeddings,
        p_encoder=p_encoder,
        q_encoder=q_encoder,
        a_encoder=a_encoder,
        vocab=vocab,
        embedding_dropout=ARGS.EMBEDDDING_DROPOUT,
        encoder_dropout=ARGS.RNN_DROPOUT
    )

    return model


def create_reader(reader_type: str) -> readers.BaseReader:
    "Returns the appropriate Reder instance from the type and configuration."
    is_bert = ARGS.EMBEDDING_TYPE == 'bert'
    if reader_type == 'simple':
        return readers.SimpleMcScriptReader(
            embedding_type=ARGS.EMBEDDING_TYPE,
            xlnet_vocab_file=ARGS.xlnet_vocab_path
        )
    if reader_type == 'full-trian':
        return readers.FullTrianReader(is_bert=is_bert,
                                       conceptnet_path=ARGS.CONCEPTNET_PATH)
    if reader_type == 'simple-bert':
        return readers.SimpleBertReader()
    if reader_type == 'simple-trian':
        return readers.SimpleTrianReader(
            embedding_type=ARGS.EMBEDDING_TYPE,
            xlnet_vocab_file=ARGS.xlnet_vocab_path,
            conceptnet_path=ARGS.CONCEPTNET_PATH
        )
    if reader_type == 'relation-bert':
        return readers.RelationBertReader(is_bert=is_bert,
                                          conceptnet_path=ARGS.CONCEPTNET_PATH)
    if reader_type == 'simple-xl':
        return readers.SimpleXLNetReader(vocab_file=ARGS.xlnet_vocab_path)
    if reader_type == 'relation-xl':
        return readers.RelationXLNetReader(
            vocab_file=ARGS.xlnet_vocab_path,
            conceptnet_path=ARGS.CONCEPTNET_PATH
        )
    if reader_type == 'extended-xl':
        return readers.ExtendedXLNetReader(
            vocab_file=ARGS.xlnet_vocab_path,
            conceptnet_path=ARGS.CONCEPTNET_PATH
        )
    raise ValueError(f'Reader type {reader_type} is invalid')


def get_modelfn_reader() -> Tuple[Callable[[Vocabulary], Model], str]:
    "Gets the build function and reader for the model"
    # Model -> Build function, reader type
    models = {
        'baseline': (build_baseline, 'simple'),
        'trian': (build_trian, 'full-trian'),
        'reader': (build_attentive_reader, 'simple'),
        'simple-bert': (build_simple_bert, 'simple-bert'),
        'advanced-bert': (build_advanced_bert, 'simple-bert'),
        'hierarchical-bert': (build_hierarchical_bert, 'simple-bert'),
        'simple-trian': (build_simple_trian, 'simple-trian'),
        'advanced-attn-bert': (build_advanced_attn_bert, 'simple-bert'),
        'han': (build_hierarchical_attn_net, 'simple-bert'),
        'rtm': (build_relational_transformer, 'relation-bert'),
        'relhan': (build_rel_han, 'relation-bert'),
        'dcmn': (build_dcmn, 'simple'),
        'zero-trian': (build_zero_trian, 'simple'),
        'simple-xl': (build_simple_xlnet, 'simple-xl'),
        'advanced-xl': (build_advanced_xlnet, 'simple-xl'),
        'relation-xl': (build_relational_xl, 'relation-xl'),
        'extended-xl': (build_advanced_xlnet, 'extended-xl'),
    }

    if ARGS.MODEL in models:
        build_fn, reader_type = models[ARGS.MODEL]
        return build_fn, reader_type
    raise ValueError(f'Invalid model name: {ARGS.MODEL}')
