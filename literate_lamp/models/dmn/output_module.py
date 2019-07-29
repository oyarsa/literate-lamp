import torch
from torch.nn.init import xavier_normal_
from overrides import overrides

from models.util import initalise_weights


class OutputModule(torch.nn.Module):
    """
    Calculates output from memory and answer vectors. Uses this operation:
        y = mUa'
    """

    def __init__(self, memory_size: int, answer_size: int, num_labels: int):
        super(OutputModule, self).__init__()
        self.memory_size = memory_size
        self.answer_size = answer_size

        self.combination = torch.nn.Linear(answer_size, memory_size)
        initalise_weights(xavier_normal_, self.combination)

    def get_input_dim(self) -> int:
        return self.memory_size

    def get_output_dim(self) -> int:
        return 1

    @overrides
    def forward(self,
                memory: torch.Tensor,
                answer: torch.Tensor
                ) -> torch.Tensor:
        """
        Inputs:
            memory: tensor of shape bsz * hdim
            answer: tensor of shape bsz * hdim
        Outputs:
            prob: tensor of shape bsz
        """
        Ua = self.combination(answer)
        memory = memory.unsqueeze(1)
        Ua = Ua.unsqueeze(1)
        mUa = memory.bmm(Ua.transpose(1, 2))
        return mUa.view(-1)
