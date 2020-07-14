import torch


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        supported_consensus_types = ['avg', 'max', 'identity']
        super().__init__()
        self.consensus_type = consensus_type.lower()
        if self.consensus_type not in supported_consensus_types:
            raise ValueError("Unknown consensus type '{}', expected one of {}".format(consensus_type, supported_consensus_types))
        self.dim = dim


    def forward(self, input):
        if self.consensus_type == "avg":
            return input.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == "max":
            return input.max(dim=self.dim, keepdim=True)[0]
        elif self.consensus_type == "identity":
            return input
        else:
            raise ValueError("Unknown consensus_type '{}'".format(self.consensus_type))
