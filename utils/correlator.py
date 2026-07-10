import torch
from torch.utils.data import Dataset, DataLoader
import math
import torch
from torch import nn
from torch.nn import functional as F

from utils.DataLoader import generate_pss, PssDataset

# class Correlator:
#     def __init__(self, batch_size: int = 1):
#         self.fft_binsize = 2048
#         self.pss_td = {l: generate_pss(l) for l in range(3)}
#
#     def find_cell_id(self, freq_offset: torch.Tensor, td_samples: torch.Tensor) -> torch.Tensor:
#         """
#         find the cell id for a given frequency offset, using correlation.
#         cell_id with max correlation value is the corresponding cell id.
#         :param freq_offset: frequency offset of the PSS from start of fft grid.
#         :return: cell_id
#         """
#         batch_size = freq_offset.shape[0]
#         output = torch.zeros((batch_size, 3), dtype=torch.int)
#         corr = {i: None for i in range(3)}
#
#         # compute exponent of frequency offset
#         n = torch.arange(self.fft_binsize)
#         freq_offset = torch.exp(2j * math.pi * torch.outer(freq_offset, n / self.fft_binsize))
#
#         # compute correlation per option
#         for cell_id in self.pss_td.keys():
#             ref_signal = freq_offset * self.pss_td[cell_id]
#             corr[cell_id] = torch.abs(torch.sum(ref_signal * torch.conj(td_samples), dim=1))
#
#         # get the max one
#         for i in range(batch_size):
#             max_index = torch.argmax(torch.tensor([corr[cell_id][i] for cell_id in corr.keys()]))
#             output[i, max_index] = 1
#         return output




class SimplePssCorrelator(nn.Module):
    """Fully Connected Layer which is correlator."""
    def __init__(self):
        super(SimplePssCorrelator, self).__init__()
        weights = []
        for cell_id in range(3):
            weights.append(torch.conj(generate_pss(cell_id)))
        # self.weight = generate_pss(cell_id)[None, :]
        self.weights = torch.stack(weights)

    def forward(self, x):
        return F.linear(x, self.weights)


class PssCorrelator(nn.Module):
    """Fixed Convolutional and Fully Connected Network for Complex RF Signals."""

    def __init__(self):
        super(PssCorrelator, self).__init__()
        self.fft_binsize = 2048
        self.pss_td = {l: generate_pss(l) for l in range(3)}
        self.n = torch.arange(self.fft_binsize)

        # create correlator per cell id
        self.fc = SimplePssCorrelator()


    def forward(self, x: torch.Tensor, freq_offset: torch.Tensor) -> torch.Tensor:
        """Compute a correlatrion results between x and pss signal.
        :param x: Native complex tensor [Batch, 2048] with dtype=torch.complex64
        :param freq_offset: Frequency offset of pss signal (in subcarrier)
        """
        freq_offset_exp  = torch.exp(2j * math.pi * torch.outer(-freq_offset, self.n / self.fft_binsize))
        x = x * freq_offset_exp

        # Convolution sequence
        cell_id_score = torch.abs(self.fc(x))

        return cell_id_score




if __name__ == "__main__":
    corr = PssCorrelator().eval()
    dataset = DataLoader(PssDataset(snr_db=None), batch_size=32)
    for a in dataset:
        break
    b = corr(x=a[0], freq_offset=a[2])
    c = 1