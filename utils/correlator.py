import torch
from torch.utils.data import Dataset, DataLoader
import math

if __package__:
    from .data_loader import generate_pss, PssDataset
else:
    from data_loader import generate_pss, PssDataset

DEFAULT_THRESHOLD = 0.053


class Correlator(torch.nn.Module):
    def __init__(self, batch_size: int = 1, coherent=False):
        super().__init__()
        self.fft_binsize = 2048
        self.coherent = coherent
        templates = torch.stack([generate_pss(i) for i in range(3)])
        templates = templates / templates.norm(dim=1, keepdim=True)
        # Real rows compute real and imaginary parts of <template, sample>.
        real_rows = torch.view_as_real(templates).reshape(3, -1)
        imag_rows = torch.stack((-templates.imag, templates.real), dim=-1).reshape(3, -1)
        self.register_buffer("projection", real_rows if coherent else torch.cat((real_rows, imag_rows)))

    def forward(self, corrected_samples):
        """Scores on already corrected samples; vectorized and device-safe."""
        real = torch.view_as_real(corrected_samples).reshape(corrected_samples.shape[0], -1)
        scores = torch.nn.functional.linear(real, self.projection)
        if not self.coherent:
            scores = (scores[:, :3].square() + scores[:, 3:].square()).sqrt()
        norm = real.square().sum(dim=1, keepdim=True).clamp_min(1e-20).sqrt()
        return scores / norm

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

        n = torch.arange(self.fft_binsize, dtype=torch.float32, device=td_samples.device)
        freq_offset_exp = torch.exp(
            2j
            * math.pi
            * torch.outer(freq_offset.to(td_samples.device).float(), n / self.fft_binsize)
        )

        return self(td_samples * freq_offset_exp.conj())

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
        
        output = torch.zeros((batch_size, 4), dtype=torch.int64, device=td_samples.device)
        corr_tensor = self.correlation_scores(freq_offset, td_samples)
        
        max_vals, predicted_classes = torch.max(corr_tensor, dim=1)

        predicted_classes[max_vals <= threshold] = 3

        output[torch.arange(batch_size, device=td_samples.device), predicted_classes] = 1

        return output

if __name__ == "__main__":
    corr = Correlator()
    dataset = DataLoader(PssDataset(snr_db=None), batch_size=32)
    for a in dataset:
        break
    
    b = corr.find_cell_id(freq_offset=a[2], td_samples=a[0])
    print(f"Predictions shape: {b.shape}")
