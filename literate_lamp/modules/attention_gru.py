import torch
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from overrides import overrides


class AttentionGRU(Seq2VecEncoder):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(AttentionGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = output_dim
        self.gru_cell = torch.nn.GRUCell(input_dim, output_dim)

    def forward(self,
                inputs: torch.Tensor,
                gate: torch.Tensor
                ) -> torch.Tensor:
        """
        Arguments:
            inputs : batch_size * seq_len * dim
            mask   : batch_size * seq_len
                (1 for padding, 0 for true input)
        Output:
            position_encoded : batch * dim
        """
        batch_size, seq_len, _ = inputs.shape

        hx = torch.zeros(batch_size, self.hidden_size,
                         dtype=inputs.dtype, device=inputs.device)

        for i in range(seq_len):
            input_i = inputs[:, i, :]
            gate_i = gate[:, i].unsqueeze(1).expand_as(input_i)

            next_h = self.gru_cell(input_i, hx)
            tmp = gate_i * next_h
            hx = tmp + (torch.tensor(1) - gate_i) * hx

        return hx

    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.hidden_size


def attention_gru(input_dim: int, output_dim: int, num_layers: int = 1,
                  bidirectional: bool = False, dropout: float = 0.0
                  ) -> Seq2VecEncoder:
    return attention_gru(input_dim, output_dim)
