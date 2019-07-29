from typing import cast

import torch
from torch import abs as tabs
from torch.nn.init import xavier_normal_
from overrides import overrides

from modules import AttentionGRU
from models.util import initalise_weights
from util import clone_module


class MemoryModule(torch.nn.Module):
    def __init__(self, hidden_dim: int, num_hops: int, dropout: float = 0.5):
        super(MemoryModule, self).__init__()

        self.hidden_dim = hidden_dim
        self.attention_gru = AttentionGRU(hidden_dim, hidden_dim)

        self.gate_nn = torch.nn.Sequential(
            torch.nn.Linear(8 * hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1, bias=False),
        )

        memory = torch.nn.Sequential(
            torch.nn.Linear(4 * hidden_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
        )
        self.memories = clone_module(memory, num_hops)

        self.dropout = torch.nn.Dropout(dropout)

        initalise_weights(xavier_normal_, self.attention_gru)
        initalise_weights(xavier_normal_, self.gate_nn)
        for memory in self.memories:
            initalise_weights(xavier_normal_, memory)

    def get_output_dim(self) -> int:
        return self.hidden_dim

    def get_gate(self,
                 facts: torch.Tensor,
                 question: torch.Tensor,
                 answer: torch.Tensor,
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
        a = answer.unsqueeze(1).expand_as(facts)
        # bsz * 1 * hdim
        m = prev_mem.unsqueeze(1).expand_as(facts)

        # bsz * num_sents * hdim
        z = torch.cat([f*q, f*m, f*a, q*a,
                       tabs(f-q), tabs(f-m), tabs(f-a), tabs(q-a)],
                      dim=-1)
        # bsz * num_sents * 1
        scores = self.gate_nn(z).squeeze(-1)

        gate = torch.softmax(scores, dim=-1)
        return gate

    @overrides
    def forward(self,
                facts: torch.Tensor,
                question: torch.Tensor,
                answer: torch.Tensor,
                prev_mem: torch.Tensor,
                hop: int
                ) -> torch.Tensor:
        gate = self.get_gate(facts, question, answer, prev_mem)
        context = self.attention_gru(facts, gate)
        context = self.dropout(context)

        mem_input = torch.cat([prev_mem, context, question, answer], dim=-1)
        next_mem = self.memories[hop](mem_input)
        next_mem = self.dropout(next_mem)
        return cast(torch.Tensor, next_mem)
