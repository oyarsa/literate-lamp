"""
A ``TokenEmbedder`` which uses one of the XLNet models
(https://github.com/zihangdai/xlnet) to produce embeddings.
At its core it uses Hugging Face's PyTorch implementation
(https://github.com/huggingface/pytorch-transformers), so thanks to them!
"""
from typing import Dict, cast, Optional
from pathlib import Path

import torch
from pytorch_transformers import XLNetModel, XLNetConfig
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.nn import util


class PretrainedXLNetModel:
    """
    In some instances you may want to load the same XLNet model twice
    (e.g. to use as a token embedder and also as a pooling layer).
    This factory provides a cache so that you don't actually have to load the
    model twice.
    """
    _cache: Dict[str, XLNetModel] = {}

    @classmethod
    def load(cls,
             config_path: Path,
             model_path: Path,
             cache_model: bool = True
             ) -> XLNetModel:
        if model_path in cls._cache:
            return PretrainedXLNetModel._cache[str(model_path)]

        config = XLNetConfig.from_pretrained(str(config_path))
        model = XLNetModel.from_pretrained(str(model_path), config=config)
        if cache_model:
            cls._cache[str(model_path)] = model

        return model


class XLNetEmbedder(TokenEmbedder):
    """
    A ``TokenEmbedder`` that produces XLNet embeddings for your tokens.
    Should be paired with a ``XLNetIndexer``, which produces wordpiece ids.
    Most likely you probably want to use ``PretrainedXLNetEmbedder``
    for one of the named pretrained models, not this base class.
    Parameters
    ----------
    bert_model: ``XLNet``
        The BERT model being wrapped.
    top_layer_only: ``bool``, optional (default = ``False``)
        If ``True``, then only return the top layer instead of apply
        the scalar mix.
    """

    def __init__(self, xlnet_model: XLNetModel) -> None:
        super().__init__()
        self.xlnet_model = xlnet_model
        self.output_dim = cast(int, xlnet_model.config.hidden_size)

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self,
                input_ids: torch.LongTensor,
                cls_indexes: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None
                ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : ``torch.LongTensor``
            The (batch_size, ..., max_sequence_length) tensor of wordpiece ids.
        token_type_ids : ``torch.LongTensor``, optional
            If an input consists of two sentences (as in the XLNet paper),
            tokens from the first sentence should have type 0 and tokens from
            the second sentence should have type 1.  If you don't provide this
            then it's assumed to be all 0s.
        """
        # pylint: disable=arguments-differ
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        input_mask = (input_ids != 0).long()

        # input_ids may have extra dimensions, so we reshape down to 2-d
        # before calling the XLBet model and then reshape back at the end.
        output, _ = self.xlnet_model(
            input_ids=util.combine_initial_dims(input_ids),
            token_type_ids=util.combine_initial_dims(token_type_ids),
            attention_mask=util.combine_initial_dims(input_mask)
        )
        return output


class PretrainedXLNetEmbedder(XLNetEmbedder):
    # pylint: disable=line-too-long
    """
    Parameters
    ----------
    pretrained_model: ``str``
        Either the name of the pretrained model to use (e.g.
        'xlnet-base-cased'), or the path to the .tar.gz file with the model
        weights.
    requires_grad : ``bool``, optional (default = False)
        If True, compute gradient of XLNet parameters for fine tuning.
    top_layer_only: ``bool``, optional (default = ``False``)
        If ``True``, then only return the top layer instead of apply the
        scalar mix.
    """

    def __init__(self,
                 config_path: Path,
                 model_path: Path,
                 requires_grad: bool = False
                 ) -> None:
        model = PretrainedXLNetModel.load(config_path=config_path,
                                          model_path=model_path)

        for param in model.parameters():
            param.requires_grad = requires_grad

        super().__init__(xlnet_model=model)
