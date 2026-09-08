import torch
from torch.utils.data import Dataset, DataLoader
import math

from data_loader import generate_pss, PssDataset

DEFAULT_THRESHOLD = 0.053


class Correlator:
    def __init__(self, batch_size: int = 1):
        self.fft_binsize = 2048
        self.pss_td = {l: generate_pss(l) for l in range(3)}

    def correlation_scores(
        self,
        freq_offset: torch.Tensor,
        td_samples: torch.Tensor,
    ) -> torch.Tensor:
        """Return normalized correlation scores for the three PSS references."""
        if td_samples.ndim != 2 or td_samples.shape[1] != self.fft_binsize:
            raise ValueError(
                f"td_samples must have shape (batch, {self.fft_binsize})"
            )
        if freq_offset.shape[0] != td_samples.shape[0]:
            raise ValueError("freq_offset and td_samples must have the same batch size")

        n = torch.arange(self.fft_binsize, dtype=torch.float32)
        freq_offset_exp = torch.exp(
            2j
            * math.pi
            * torch.outer(freq_offset.float(), n / self.fft_binsize)
        )

        correlations = []
        for cell_id in self.pss_td:
            ref_signal = freq_offset_exp * self.pss_td[cell_id]
            numerator = torch.abs(
                torch.sum(torch.conj(ref_signal) * td_samples, dim=1)
            )
            denominator = (
                torch.linalg.vector_norm(ref_signal, dim=1)
                * torch.linalg.vector_norm(td_samples, dim=1)
            )
            correlations.append(numerator / denominator.clamp_min(1e-12))

        return torch.stack(correlations, dim=1)

    def find_cell_id(
        self,
        freq_offset: torch.Tensor,
        td_samples: torch.Tensor,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> torch.Tensor:
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
        
        output = torch.zeros((batch_size, 4), dtype=torch.int64)
        corr_tensor = self.correlation_scores(freq_offset, td_samples)
        
        max_vals, predicted_classes = torch.max(corr_tensor, dim=1)

        predicted_classes[max_vals < threshold] = 3

        output[torch.arange(batch_size), predicted_classes] = 1

        return output

if __name__ == "__main__":
    corr = Correlator()
    dataset = DataLoader(PssDataset(snr_db=None), batch_size=32)
    for a in dataset:
        break
    
    b = corr.find_cell_id(freq_offset=a[2], td_samples=a[0])
    print(f"Predictions shape: {b.shape}")
