"""PSS signal generation: sequences, AWGN and the known frequency shift.

The four-class problem these signals build (see :class:`~utils.data.pss_dataset.PssDataset`):

* labels ``0``, ``1``, ``2`` -- a window containing PSS sequence ``N_ID^(2) = label``;
* label ``3``               -- a noise-only window (no signal at all).

Every window is exactly :data:`FFT_SIZE` complex time-domain samples.
"""
import math
from typing import Sequence, Union

import torch

FFT_SIZE = 2048            # complex samples per window, and the IFFT size of the PSS
PSS_LENGTH = 127           # BPSK symbols in one PSS sequence (38.211 §7.4.2.2)
NUM_PSS_IDS = 3            # N_ID^(2) in {0, 1, 2}
NOISE_LABEL = 3            # class index of a noise-only window
NUM_CLASSES = NUM_PSS_IDS + 1
# What every requested SNR means; recorded in each results file and shown on the figures.
SNR_DEFINITION = "per PSS subcarrier"
# generate_pss's ifftshift puts the first PSS symbol at bin FFT_SIZE / 2.
PSS_FIRST_BIN = FFT_SIZE // 2
# Samples of the narrowband view: the smallest power of two that holds the PSS's 127 bins.
NARROWBAND_SIZE = 128


class PssSignalGenerator:
    """Signal-level building blocks: PSS sequences, AWGN and frequency shifts.

    Stateless; every method is static. :class:`PssDataset` composes them into
    windows, and the receivers use :meth:`correct_frequency_offset` to undo the
    dataset's shift.
    """

    @staticmethod
    def generate_pss(n_id: int) -> torch.Tensor:
        """
        Generate one PSS sequence in the time domain (3GPP 38.211 §7.4.2.2).

        The PSS sits at the start of the FFT grid; the frequency offset is applied
        later, in the time domain, by :class:`PssDataset`.

        :param n_id: PSS identity N_ID^(2), 0..2 (the cell ID modulo 3)
        :return: complex64 tensor of shape (FFT_SIZE,)
        """
        # m-sequence x(i+7) = x(i+4) XOR x(i), seeded with [0, 1, 1, 0, 1, 1, 1].
        x = torch.zeros(PSS_LENGTH, dtype=torch.bool)
        x[:7] = torch.tensor([0, 1, 1, 0, 1, 1, 1], dtype=torch.bool)
        for i in range(PSS_LENGTH - 7):
            x[i + 7] = x[i + 4] ^ x[i]

        # Cyclic shift by 43 * N_ID, then BPSK: d = 1 - 2x.
        n = torch.arange(PSS_LENGTH, dtype=torch.long)
        m_pss = (n + 43 * n_id) % PSS_LENGTH
        d_pss = 1 - 2 * x[m_pss].long()

        # The 127 BPSK symbols occupy the lowest bins; ifftshift then moves bin 0 to
        # index N/2, so the sequence starts at the centre of the IFFT input and runs
        # upward from there (indices 1024..1150), not symmetrically about it.
        pss_freq_domain = torch.zeros(FFT_SIZE, dtype=torch.complex64)
        pss_freq_domain[:d_pss.shape[0]] = d_pss

        # torch's default (unnormalized) ifft divides by N. By Parseval the resulting
        # average time-domain power is therefore 127 / 2048**2 ~= 3.028e-5, not 1.
        return torch.fft.ifft(torch.fft.ifftshift(pss_freq_domain))

    @staticmethod
    def pss_templates(normalized: bool = False) -> torch.Tensor:
        """
        Stack the clean waveforms of every PSS identity.

        :param normalized: scale every row to unit norm (default: False)
        :return: complex64 tensor of shape (NUM_PSS_IDS, FFT_SIZE); row k is PSS k
        """
        templates = torch.stack([PssSignalGenerator.generate_pss(i) for i in range(NUM_PSS_IDS)])
        return templates / templates.norm(dim=1, keepdim=True) if normalized else templates

    @staticmethod
    def narrowband(samples: torch.Tensor) -> torch.Tensor:
        """
        Return the 128-sample view of frequency-corrected windows: the NARROWBAND_SIZE bins from PSS_FIRST_BIN.

        Keeping those FFT bins and inverse-transforming them is an ideal band-pass and
        decimation by FFT_SIZE / NARROWBAND_SIZE: the whole PSS stays, the noise of every
        other bin goes, and the SNR per subcarrier is unchanged.

        :param samples: complex tensor of shape (batch, FFT_SIZE), frequency corrected
        :return: complex tensor of shape (batch, NARROWBAND_SIZE)
        """
        return torch.fft.ifft(torch.fft.fft(samples)[..., PSS_FIRST_BIN:PSS_FIRST_BIN + NARROWBAND_SIZE])

    @staticmethod
    def generate_noise_vector(num_samples: int, snr_db: Union[float, Sequence[float]],
                              pss_sequence_td: torch.Tensor) -> torch.Tensor:
        """
        Generate circularly symmetric complex AWGN, one window per row.

        The SNR is per PSS subcarrier (3GPP's per-resource-element view): the power
        of one occupied FFT bin, ``|X[k]|^2 = 1`` for the BPSK PSS, over the noise power
        ``E|N[k]|^2`` in one bin of the unnormalized FFT. Averaged over the time samples
        instead, the same window has an SNR 10*log10(2048/127) = 12.08 dB lower,
        because the PSS fills 127 of the 2048 bins and the noise fills all of them.

        :param num_samples: number of independent noise windows to draw
        :param snr_db: scalar SNR in dB, or a [minimum, maximum] pair that draws one
            uniform SNR per window
        :param pss_sequence_td: a PSS window; only its length is used
        :return: complex64 tensor of shape (num_samples, len(pss_sequence_td)). Power
            is split evenly between real and imaginary parts.
        :raises ValueError: if ``snr_db`` is a sequence that is not a pair
        """
        fft_binsize = pss_sequence_td.shape[0]

        if isinstance(snr_db, (list, tuple)):
            if len(snr_db) != 2:
                raise ValueError("snr_db must be a scalar or a [minimum, maximum] pair")
            snr_min, snr_max = sorted(float(value) for value in snr_db)
            snr_db_tensor = torch.empty(num_samples).uniform_(snr_min, snr_max)
        else:
            snr_db_tensor = torch.full((num_samples,), float(snr_db))

        snr_linear = torch.pow(10.0, snr_db_tensor / 10.0)
        # Time-domain noise of variance s2 puts N * s2 into every bin of the
        # unnormalized FFT, and each PSS bin carries power 1: SNR = 1 / (N * s2).
        noise_variance = 1.0 / (fft_binsize * snr_linear)
        # Total power is split over two real components, hence the /2 before the sqrt.
        component_std = torch.sqrt(noise_variance / 2.0)

        noise_real = torch.randn((num_samples, fft_binsize), dtype=torch.float32)
        noise_imag = torch.randn((num_samples, fft_binsize), dtype=torch.float32)
        return component_std[:, None] * (noise_real + 1j * noise_imag)

    @staticmethod
    def frequency_shift(freq_offset: torch.Tensor, length: int,
                        device: Union[torch.device, None] = None) -> torch.Tensor:
        """
        Build the per-window rotation exp(2j*pi*k*n/N) that shifts a window by k bins.

        :param freq_offset: offsets k in FFT bins, shape (batch,)
        :param length: window length N in samples
        :param device: device to build the rotation on (default: the offsets' device)
        :return: complex64 tensor of shape (batch, length)
        """
        device = freq_offset.device if device is None else device
        n = torch.arange(length, dtype=torch.float32, device=device)
        offsets = freq_offset.to(device=device, dtype=torch.float32)
        return torch.exp(2j * math.pi * torch.outer(offsets, n / length))

    @staticmethod
    def receiver_input(samples: torch.Tensor, freq_offset: torch.Tensor, correct_offset: bool) -> torch.Tensor:
        """
        Return the window a receiver is handed: with its known offset removed, or as received.

        :param samples: complex tensor of shape (batch, FFT_SIZE)
        :param freq_offset: offsets in FFT bins, shape (batch,)
        :param correct_offset: remove the known offset (:meth:`correct_frequency_offset`);
            False hands the window over as received. Callers pass the receiver's
            ``RECEIVES_CORRECTED_WINDOW``.
        :return: complex tensor of shape (batch, FFT_SIZE)
        """
        return PssSignalGenerator.correct_frequency_offset(samples, freq_offset) if correct_offset else samples

    @staticmethod
    def correct_frequency_offset(samples: torch.Tensor, freq_offset: torch.Tensor) -> torch.Tensor:
        """
        Undo the known frequency offset applied by :class:`PssDataset`.

        This is the exact inverse of the dataset's shift, so it is side information,
        not an estimator. Every receiver gets the same correction, which is why it is
        excluded from all per-receiver operation counts.

        :param samples: complex tensor of shape (batch, FFT_SIZE)
        :param freq_offset: offsets in FFT bins, shape (batch,)
        :return: complex tensor of shape (batch, FFT_SIZE) with the offset removed
        """
        shift = PssSignalGenerator.frequency_shift(freq_offset, samples.shape[1], samples.device)
        return samples * torch.conj(shift)
