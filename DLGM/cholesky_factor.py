
import torch

class CholeskyFactor:
    def __init__(self, size, delta = 1e-4):
        self.size = size
        self._free_parameter_size = size + size*(size-1)//2

        self.delta = delta

        self.diag_ii = torch.arange(self.size)
        self.diag_jj = torch.arange(self.size)
        self.low_ii, self.low_jj = torch.tril_indices(size, size, -1)

    def free_parameter_size(self):
        return self._free_parameter_size

    def parameterize(self, free_parameter):
        batch_size = free_parameter.shape[0]

        assert free_parameter.shape[1] == self.free_parameter_size()
        R = torch.zeros(batch_size, self.size, self.size)
        R[:, self.diag_ii, self.diag_jj] = free_parameter[:, :self.size].exp() + self.delta
        R[:, self.low_ii, self.low_jj] = free_parameter[:, self.size:]
        return R

class DiagonalFactor:
    def __init__(self, size, delta = 1e-6):
        self.size = size
        self._free_parameter_size = size

        self.delta = delta

        self.diag_ii = torch.arange(self.size)
        self.diag_jj = torch.arange(self.size)

    def free_parameter_size(self):
        return self._free_parameter_size

    def parameterize(self, free_parameter):
        batch_size = free_parameter.shape[0]

        assert free_parameter.shape[1] == self.free_parameter_size()
        R = torch.zeros(batch_size, self.size, self.size)
        R[:, self.diag_ii, self.diag_jj] = free_parameter[:, :self.size].exp() + self.delta

        return R
