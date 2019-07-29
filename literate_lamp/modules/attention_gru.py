from typing import cast

import torch
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from overrides import overrides


class AttentionGRUCell(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super(AttentionGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size

        self.update_gate_input = torch.nn.Linear(input_dim, hidden_size)
        self.update_gate_state = torch.nn.Linear(hidden_size, hidden_size)
        self.update_activation = torch.nn.Sigmoid()
        self.output_gate_input = torch.nn.Linear(input_dim, hidden_size)
        self.output_gate_state = torch.nn.Linear(hidden_size, hidden_size)
        self.output_activation = torch.nn.Tanh()

    @overrides
    def forward(self,
                inputs: torch.Tensor,
                previous_state: torch.Tensor,
                gate: torch.Tensor
                ) -> torch.Tensor:
        """
        Arguments:
            inputs : batch_size * seq_len * dim
            previous_state : batch_size * dim
            gate   : batch_size * 1
        Output:
            output : batch * dim
        """
        update = self.update_activation(self.update_gate_input(inputs) +
                                        self.update_gate_state(previous_state))
        next_state = self.output_activation(
            self.output_gate_input(inputs) +
            update * self.output_gate_state(previous_state)
        )
        output = gate * next_state + \
            (1 - gate) * previous_state  # type: ignore
        return cast(torch.Tensor, output)


class AttentionGRU(Seq2VecEncoder):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(AttentionGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = output_dim
        self.gru_cell = AttentionGRUCell(input_dim, output_dim)

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
            final_state : batch * dim
        """
        batch_size, seq_len, _ = inputs.shape

        hidden_state = torch.zeros(batch_size, self.hidden_size,
                                   dtype=inputs.dtype, device=inputs.device)

        for i in range(seq_len):
            input_i = inputs[:, i, :]
            gate_i = gate[:, i].unsqueeze(1).expand_as(input_i)

            hidden_state = self.gru_cell(input_i, hidden_state, gate_i)

        return hidden_state

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
