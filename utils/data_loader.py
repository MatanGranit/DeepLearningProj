import torch
from torch._C import dtype
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


def generate_noise_vector(num_samples: int, noise_db: float, pss_sequence_td: torch.Tensor) -> torch.Tensor:
    """
    generate noise vector given the DB
    :param num_samples:
    :param noise_db: noise level in dB
    :param pss_sequence_td: sample of pss sequence in time domain to have sample of PSS power
    :return: noise vector in time domain
    """
    # compute the noise's sigma
    fft_binsize = pss_sequence_td.shape[0]

    # signal power - E[s^2]
    signal_power = torch.sum(torch.abs(pss_sequence_td) ** 2) / fft_binsize

    # 1. Calculate total noise power (Variance)
    noise_power = signal_power / (10.0 ** (noise_db / 10.0))

    # 2. Calculate the standard deviation for complex noise
    # Since power splits equally across real and imaginary parts, variance per part is (noise_power / 2)
    # Standard deviation is the square root of the variance
    noise_std = math.sqrt(noise_power / 2.0)

    # 3. Generate standard normal noise
    noise_real = torch.randn((num_samples, fft_binsize), dtype=torch.float32)
    noise_imag = torch.randn((num_samples, fft_binsize), dtype=torch.float32)
    
    # 4. Scale by the STANDARD DEVIATION, not the power!
    noise_matrix = noise_std * (noise_real + 1j * noise_imag)
    
    return noise_matrix


class PssDataset(Dataset):
    def __init__(self, num_samples: int=int(1e4), noise_db: float=None):
        """
        initialize vector of labels (cell ids), pss sequences (1 per cell_id) and frequency shifts.
        :param num_samples: number of samples to create
        :param noise_db: noise level in dB
        """
        fft_binsize = 2048
        # initialize data
        # UPDATE 1: sample random cell ids from 0 to 3 (4 classes total)
        labels = torch.randint(low=0, high=4, size=(num_samples,), dtype=torch.long)
        
        pss_sequences_td = {l: generate_pss(l) for l in range(3)}
        # UPDATE 2: Add label 3 (No Signal) as an array of pure zeros
        pss_sequences_td[3] = torch.zeros_like(pss_sequences_td[0])
        
        samples = torch.stack([pss_sequences_td[labels[i].item()] for i in range(num_samples)])

        # sample random frequency offsets (within fft grid) and add them to create real samples
        self.freq_offset = torch.randint(low=0, high=fft_binsize-128, size=(num_samples,))
        n = torch.arange(fft_binsize)
        freq_offset_exponent = torch.exp(2j * math.pi * torch.outer(self.freq_offset, n / fft_binsize))
        
        # For PSS (0,1,2) this shifts the frequency. For No Signal (3), 0 * shift = 0.
        samples = samples * freq_offset_exponent
        
        if noise_db is not None:
            # We use pss_sequences_td[0] to calculate noise power based on a real signal
            noise = generate_noise_vector(num_samples=num_samples, noise_db=noise_db, pss_sequence_td=pss_sequences_td[0])
            # For label 3, 'samples' is currently 0, so 0 + noise = pure noise!
            samples = samples + noise
            
        self.samples = samples
        
        # UPDATE 3: Update one-hot encoding size to 4 classes
        self.labels_hot_ones = torch.zeros((num_samples, 4), dtype=torch.float32) 
        self.labels_hot_ones[torch.arange(num_samples), labels] = 1.0

    def __len__(self):
        # return the total number of samples
        return self.labels_hot_ones.shape[0]

    def __getitem__(self, idx):
        # retrieve a single sample and its corresponding label
        # (I removed the unused 'label = torch.zeros...' line that was here)
        return self.samples[idx], self.labels_hot_ones[idx], self.freq_offset[idx]
    

if __name__ == '__main__':
    dataset = DataLoader(PssDataset(noise_db=None), shuffle=True, batch_size=32)
    for a in dataset:
        break

    b = 1