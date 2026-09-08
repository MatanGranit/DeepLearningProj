from typing import List, Union

import torch
from torch.utils.data import Dataset, DataLoader
import math



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

def generate_noise_vector(num_samples: int, snr_db: Union[float, List[float]], pss_sequence_td: torch.Tensor) -> torch.Tensor:
    """
    generate noise vector given the DB
    :param num_samples:
    :param snr_db: noise level in dB
    :param pss_sequence_td: sample of pss sequence in time domain to have sample of PSS power
    :return: noise vector in time domain
    """
    if isinstance(snr_db, list):
        snr_max = max(snr_db)
        snr_min = min(snr_db)
        snr_db = torch.rand(num_samples, dtype=torch.float32) * (snr_max - snr_min) + snr_min
    else:
        snr_db = torch.full((num_samples,), float(snr_db), dtype=torch.float32)
    # compute the noise's sigma
    fft_binsize = pss_sequence_td.shape[0]

    # signal power - E[s^2]
    signal_power = torch.sum(torch.abs(pss_sequence_td) ** 2) / fft_binsize

    # SNR = 10*log10(E[s^2] / E[n^2])  ->   E[n^2] = E[s^2] / (10^(SNR/10))
    noise_power = signal_power / torch.pow(10.0, snr_db / 10.0)
    noise_std = torch.sqrt(noise_power / 2.0)

    # compute noise - E[n^2]=sigma
    noise_real = torch.randn((num_samples, fft_binsize), dtype=torch.float32)
    noise_imag = torch.randn((num_samples, fft_binsize), dtype=torch.float32)
    noise_matrix = noise_std.unsqueeze(1) * (noise_real + 1j * noise_imag)
    return noise_matrix


class PssDataset(Dataset):
    def __init__(self, num_samples: int=int(1e4), snr_db: Union[float | List]=None):
        """
        initialize vector of labels (cell ids), pss sequences (1 per cell_id) and frequency shifts.
        :param num_samples: number of samples to create
        :param snr_db: noise level in dB
        """
        fft_binsize = 2048
        # initialize data
        # sample random cell ids and create basic pss sequence per optional cell_id
        labels = torch.randint(low=0, high=3, size=(num_samples,), dtype=torch.long)
        pss_sequences_td = {l: generate_pss(l) for l in range(3)}
        samples = torch.stack([pss_sequences_td[labels[i].item()] for i in range(num_samples)])

        # sample random frequency offsets (within fft grid) and add them to create real samples
        # self.freq_offset = torch.randint(low=0, high=fft_binsize-128, size=(num_samples,))
        self.freq_offset = torch.randint(low=0, high=int(fft_binsize/80)-1, size=(num_samples,)) * 80
        n = torch.arange(fft_binsize)
        freq_offset_exponent = torch.exp(2j * math.pi * torch.outer(self.freq_offset, n / fft_binsize))
        samples = samples * freq_offset_exponent
        if snr_db is not None:
            noise = generate_noise_vector(num_samples=num_samples, snr_db=snr_db, pss_sequence_td=pss_sequences_td[0])
            samples = samples + noise
        self.samples = samples
        self.labels = labels

    def __len__(self):
        # return the total number of samples
        return self.labels.shape[0]

    def __getitem__(self, idx):
        # retrieve a single sample and its corresponding label
        label = torch.zeros(3, dtype=torch.int)

        return self.samples[idx], self.labels[idx], self.freq_offset[idx]


if __name__ == '__main__':
    dataset = DataLoader(PssDataset(snr_db=None), shuffle=True, batch_size=32)
    for a in dataset:
        break

    b = 1
