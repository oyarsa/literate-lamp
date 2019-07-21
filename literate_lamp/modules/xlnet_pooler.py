from typing import Optional

import torch
from overrides import overrides
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder


class XLNetPooler(Seq2VecEncoder):
    def __init__(self, input_dim: int) -> None:
        super(XLNetPooler, self).__init__()
        self.dense = torch.nn.Linear(input_dim, input_dim)
        self.activation = torch.nn.Tanh()
        self.input_dim = input_dim

    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.input_dim

    def forward(self,
                hidden_states: torch.Tensor,
                cls_index: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        hidden_size = hidden_states.shape[-1]

        if cls_index is not None:
            # shape (bsz, 1, hsz)
            cls_index = cls_index[:, None].expand(-1, -1, hidden_size)
            # shape (bsz, hsz)
            cls_token_state = hidden_states.gather(-2, cls_index).squeeze(-2)
        else:
            cls_token_state = hidden_states[:, -1, :]  # shape (bsz, hsz)

        x = self.dense(cls_token_state)
        x = self.activation(x)

        return x
