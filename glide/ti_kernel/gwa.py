import torch
import Triton

class CasualGlidingWindowAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
