from typing import List, Sequence, Union

import math
import torch
from torch.utils.data import Dataset



def generate_pss(n_id) -> torch.Tensor:
    """
    generate pss sequence given n_id, as described in 3GPP 138.211 - 7.4.2.
    pss is located at the beginning of the FFT grid, offset will be added later in time domain.
    :param n_id: Cell ID (Modulo 3)
    :return: pss sequence in time domain
    """
    fft_binsize = 2048

    # initialize a boolean tensor of zeros
    x = torch.zeros(127, dtype=torch.bool)

    # set the initial 7-element m-sequence initialization values
    x[:7] = torch.tensor([0, 1, 1, 0, 1, 1, 1], dtype=torch.bool)

    # generate the rest of the m-sequence using XOR (^)
    for i in range(127 - 7):
        x[i + 7] = x[i + 4] ^ x[i]

    # create the index array 0 to 126
    n = torch.arange(127, dtype=torch.long)

    # calculate the cyclic shift based on N_ID
    m_pss = (n + 43 * n_id) % 127

    # cast the boolean tensor to long integers to perform 1 - 2*x
    d_pss = 1 - 2 * x[m_pss].long()

    pss_freq_domain = torch.zeros(fft_binsize, dtype=torch.complex64)
    pss_freq_domain[:d_pss.shape[0]] = d_pss

    pss_time_domain = torch.fft.ifft(torch.fft.ifftshift(pss_freq_domain))
    return pss_time_domain


def generate_noise_vector(
    num_samples: int,
    snr_db: Union[float, Sequence[float]],
    pss_sequence_td: torch.Tensor,
) -> torch.Tensor:
    """
    Generate circularly symmetric complex AWGN at the requested SNR.

    SNR is defined using average time-domain powers:
        SNR = E[|s|^2] / E[|n|^2].
    :param num_samples:
    :param snr_db: scalar SNR in dB, or [minimum, maximum] for per-sample SNRs
    :param pss_sequence_td: sample of pss sequence in time domain to have sample of PSS power
    :return: noise vector in time domain
    """
    fft_binsize = pss_sequence_td.shape[0]
    signal_power = torch.mean(torch.abs(pss_sequence_td) ** 2)

    if isinstance(snr_db, (list, tuple)):
        if len(snr_db) != 2:
            raise ValueError("snr_db must be a scalar or a [minimum, maximum] pair")
        snr_min, snr_max = sorted(float(value) for value in snr_db)
        snr_db_tensor = torch.empty(num_samples).uniform_(snr_min, snr_max)
    else:
        snr_db_tensor = torch.full((num_samples,), float(snr_db))

    snr_linear = torch.pow(10.0, snr_db_tensor / 10.0)
    noise_variance = signal_power / snr_linear
    component_std = torch.sqrt(noise_variance / 2.0)

    noise_real = torch.randn((num_samples, fft_binsize), dtype=torch.float32)
    noise_imag = torch.randn((num_samples, fft_binsize), dtype=torch.float32)
    return component_std[:, None] * (noise_real + 1j * noise_imag)


class PssDataset(Dataset):
    def __init__(
        self,
        num_samples: int = int(1e4),
        snr_db: Union[float, List[float], None] = None,
        random_phase: bool = False,
    ):
        """
        initialize vector of labels (cell ids), pss sequences (1 per cell_id) and frequency shifts.
        :param num_samples: number of samples to create
        :param snr_db: SNR in dB. Labels 0--2 are PSS IDs and label 3 is noise only.
        :param random_phase: multiply every signal window by an independent
            phase exp(j*phi), phi uniform on [0, 2*pi). Noise-only windows are
            unchanged. The default is False to preserve the original results.
        """
        fft_binsize = 2048
        # initialize data
        self.labels = torch.randint(low=0, high=4, size=(num_samples,), dtype=torch.long)
        
        pss_sequences_td = {l: generate_pss(l) for l in range(3)}
        pss_sequences_td[3] = torch.zeros_like(pss_sequences_td[0])
        samples = torch.stack(
            [pss_sequences_td[label.item()] for label in self.labels]
        )

        # sample random frequency offsets (within fft grid) and add them to create real samples
        self.freq_offset = torch.randint(
            low=0, high=fft_binsize - 128, size=(num_samples,)
        )
        n = torch.arange(fft_binsize, dtype=torch.float32)
        freq_offset_exponent = torch.exp(
            2j
            * math.pi
            * torch.outer(self.freq_offset.float(), n / fft_binsize)
        )
        
        # For PSS (0,1,2) this shifts the frequency. For No Signal (3), 0 * shift = 0.
        samples = samples * freq_offset_exponent

        if random_phase:
            phase = 2 * math.pi * torch.rand(num_samples, dtype=torch.float32)
            samples = samples * torch.exp(1j * phase[:, None])
        
        if snr_db is not None:
            noise = generate_noise_vector(
                num_samples=num_samples,
                snr_db=snr_db,
                pss_sequence_td=pss_sequences_td[0],
            )
            samples = samples + noise

        self.samples = samples

    def __len__(self):
        # return the total number of samples
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx], self.freq_offset[idx]
