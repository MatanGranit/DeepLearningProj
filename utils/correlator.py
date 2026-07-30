import torch
from torch.utils.data import Dataset, DataLoader
import math

from data_loader import generate_pss, PssDataset

class Correlator:
    def __init__(self, batch_size: int = 1):
        self.fft_binsize = 2048
        self.pss_td = {l: generate_pss(l) for l in range(3)}

    def find_cell_id(self, freq_offset: torch.Tensor, td_samples: torch.Tensor, threshold: float = 0.03) -> torch.Tensor:
        """
        find the cell id for a given frequency offset, using correlation.
        cell_id with max correlation value is the corresponding cell id.
        If max correlation is below the threshold, detects as class 3 (Noise only).
        :param freq_offset: frequency offset of the PSS from start of fft grid.
        :param td_samples: Input signal samples
        :param threshold: The correlation threshold to declare a signal is present
        :return: cell_id one-hot tensor of shape (batch_size, 4)
        """
        batch_size = freq_offset.shape[0]
        
        output = torch.zeros((batch_size, 4), dtype=torch.int)
        corr = {i: None for i in range(3)}

        # compute exponent of frequency offset
        n = torch.arange(self.fft_binsize)
        freq_offset_exp = torch.exp(2j * math.pi * torch.outer(freq_offset, n / self.fft_binsize))

        # compute correlation per option
        for cell_id in self.pss_td.keys():
            ref_signal = freq_offset_exp * self.pss_td[cell_id]
            corr[cell_id] = torch.abs(torch.sum(ref_signal * torch.conj(td_samples), dim=1))

        corr_tensor = torch.stack([corr[i] for i in range(3)], dim=1)
        
        max_vals, predicted_classes = torch.max(corr_tensor, dim=1)

        predicted_classes[max_vals < threshold] = 3

        output[torch.arange(batch_size), predicted_classes] = 1

        return output

if __name__ == "__main__":
    corr = Correlator()
    dataset = DataLoader(PssDataset(noise_db=None), batch_size=32)
    for a in dataset:
        break
    
    b = corr.find_cell_id(freq_offset=a[2], td_samples=a[0], threshold=200.0)
    print(f"Predictions shape: {b.shape}")