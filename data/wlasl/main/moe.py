import dataclasses
from typing import List

import torch
import torch.nn.functional as F
from simple_parsing.helpers import Serializable
from torch import nn


@dataclasses.dataclass
class MoeArgs(Serializable):
    num_experts: int
    num_experts_per_tok: int


class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, moe_args: 'MoeArgs'):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.args = moe_args

    def forward(self, inputs: torch.Tensor):
        # Compute gate logits and select top-k experts per token
        gate_logits = self.gate(inputs)

        # Top-k selection
        weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)

        # Apply softmax to the selected weights
        weights = F.softmax(weights, dim=-1, dtype=torch.float).to(inputs.dtype)

        # Prepare the results tensor
        results = torch.zeros_like(inputs)

        # Process each expert index and apply its logic
        for i, expert in enumerate(self.experts):
            # Find all indices where this expert is selected
            mask = selected_experts == i
            batch_idx, seq_idx, keypoint_idx, expert_pos = torch.where(mask)

            # Use the mask to retrieve the corresponding weights
            selected_weights = weights[batch_idx, seq_idx, keypoint_idx, expert_pos]

            # Apply the expert only to the selected tokens
            results[batch_idx, seq_idx, keypoint_idx] += selected_weights.unsqueeze(-1) * expert(
                inputs[batch_idx, seq_idx, keypoint_idx]
            )

        return results
