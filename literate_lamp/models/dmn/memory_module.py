from typing import cast

import torch
from overrides import overrides

from modules import AttentionGRU


class MemoryModule(torch.nn.Module):
    def __init__(self, hidden_dim: int):
        super(MemoryModule, self).__init__()

        self.hidden_dim = hidden_dim
        self.attention_gru = AttentionGRU(hidden_dim, hidden_dim)

        self.gate_nn = torch.nn.Sequential(
            torch.nn.Linear(4 * hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1, bias=False),
        )

        self.memory_output = torch.nn.Sequential(
            torch.nn.Linear(3 * hidden_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
        )

    def get_output_dim(self) -> int:
        return self.hidden_dim

    def get_gate(self,
                 facts: torch.Tensor,
                 question: torch.Tensor,
                 prev_mem: torch.Tensor
                 ) -> torch.Tensor:
        """
        Inputs:
            facts : tensor of shape batch_size * seq_len * hdim
            question : tensor of shape batch_size * hdim
            answer : tensor of shape batch_size * hdim
            prev_mem : tensor of shape batch_size * hdim
        Output:
            next_mem : tensor of shape batch_size * hdim
        """
        # bsz * num_sents * hdim
        f = facts
        # bsz * 1 * hdim
        q = question.unsqueeze(1).expand_as(facts)
        # bsz * 1 * hdim
        m = prev_mem.unsqueeze(1).expand_as(facts)

        # bsz * num_sents * hdim
        z = torch.cat([f*q, f*m, torch.abs(f-q), torch.abs(f-m)], dim=-1)
        # bsz * num_sents * 1
        scores = self.gate_nn(z).squeeze(-1)

        gate = torch.softmax(scores, dim=-1)
        return gate

    @overrides
    def forward(self,
                facts: torch.Tensor,
                question: torch.Tensor,
                prev_mem: torch.Tensor
                ) -> torch.Tensor:
        gate = self.get_gate(facts, question, prev_mem)
        context = self.attention_gru(facts, gate)

        mem_input = torch.cat([prev_mem, context, question], dim=-1)
        next_mem = self.memory_output(mem_input)
        return cast(torch.Tensor, next_mem)
