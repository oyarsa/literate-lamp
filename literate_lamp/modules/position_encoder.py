from typing import Optional

import torch
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from overrides import overrides


class PositionEncoder(Seq2VecEncoder):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(PositionEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projection = torch.nn.Linear(input_dim, output_dim)

    def position_matrix(self, seq_len: int) -> torch.Tensor:
        D = self.input_dim
        M = seq_len

        pos_list = [[(1 - j/M) - (d/D)*(1 - 2*j/M) for d in range(1, D+1)]
                    for j in range(1, M+1)]
        pos = torch.tensor(pos_list)
        return pos

    def forward(self,
                inputs: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Arguments:
            inputs : batch_size * seq_len * dim
            mask   : batch_size * seq_len
                (1 for padding, 0 for true input)
        Output:
            position_encoded : batch * dim
        """
        seq_len = inputs.size(1)

        positions = self.position_matrix(seq_len).to(inputs.device)
        positions = positions.expand_as(inputs)

        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            inputs = inputs * mask

        output = (inputs * positions).sum(dim=-2)

        output = self.projection(output)
        return output

    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim


def position_encoder(input_dim: int, output_dim: int, num_layers: int = 1,
                     bidirectional: bool = False, dropout: float = 0.0
                     ) -> Seq2VecEncoder:
    return PositionEncoder(input_dim, output_dim)
