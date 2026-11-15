"""The PSS dataset: four-class windows and the pooled-SNR loaders built from them.

The four classes are drawn with equal probability; see :mod:`utils.data.pss_signal`
for the signals themselves.
"""
import math
from typing import List, Sequence, Tuple, Union

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .pss_signal import FFT_SIZE, NOISE_LABEL, NUM_CLASSES, NUM_PSS_IDS, PssSignalGenerator

# Dataset offsets are multiples of this many bins (0, 80, ... 1840), so the PSS
# lands on a coarse grid rather than at an arbitrary bin.
FREQ_OFFSET_STEP = 80
# One less than FFT_SIZE / FREQ_OFFSET_STEP, so the largest offset is 1840, not 1920.
NUM_FREQ_OFFSETS = int(FFT_SIZE / FREQ_OFFSET_STEP) - 1
# Every offset a window can carry, in FFT bins: the search space of a scanning receiver.
FREQ_OFFSETS = tuple(FREQ_OFFSET_STEP * index for index in range(NUM_FREQ_OFFSETS))


class PssDataset(Dataset):
    """Four-class PSS windows: three PSS identities plus a noise-only class.

    Each item is a ``(window, label, freq_offset)`` triple. ``freq_offset`` is the
    frequency shift applied when the window was built. Receivers with
    ``RECEIVES_CORRECTED_WINDOW`` get the window with it removed, as exact side
    information; the others get the window as built and must search the offset.
    """

    def __init__(self, num_samples: int = int(1e4), snr_db: Union[float, List[float], None] = None,
                 random_phase: bool = False) -> None:
        """
        Draw labels, frequency offsets, optional phases and noise for every window.

        The random draws happen in a fixed order (labels, offsets, phases, noise);
        keeping that order is what keeps seeded datasets reproducible.

        :param num_samples: number of windows to create
        :param snr_db: SNR in dB, or a [minimum, maximum] pair. ``None`` skips the noise
            stage and yields clean windows, which the unit tests use.
        :param random_phase: multiply every signal window by an independent phase
            exp(j*phi), phi uniform on [0, 2*pi). Noise-only windows are unchanged.
        """
        self.labels = torch.randint(low=0, high=NUM_CLASSES, size=(num_samples,), dtype=torch.long)

        # One waveform per class. The noise class is an all-zero window, so the noise
        # added below is the only thing it ever contains.
        pss_sequences_td = {label: PssSignalGenerator.generate_pss(label) for label in range(NUM_PSS_IDS)}
        pss_sequences_td[NOISE_LABEL] = torch.zeros_like(pss_sequences_td[0])
        samples = torch.stack([pss_sequences_td[label.item()] for label in self.labels])

        self.freq_offset = torch.randint(low=0, high=NUM_FREQ_OFFSETS, size=(num_samples,)) * FREQ_OFFSET_STEP
        # Multiplying by exp(2j*pi*k*n/N) in time is a k-bin rotation in frequency;
        # a noise-only window stays all zero.
        samples = samples * PssSignalGenerator.frequency_shift(self.freq_offset, FFT_SIZE)

        if random_phase:
            # One phase per window, constant across the window: this is exactly the
            # unknown common phase a coherent receiver would have to estimate.
            phase = 2 * math.pi * torch.rand(num_samples, dtype=torch.float32)
            samples = samples * torch.exp(1j * phase[:, None])

        if snr_db is not None:
            # All four classes get noise at the same absolute level, which is what
            # makes the noise-only class a fair false-alarm population.
            noise = PssSignalGenerator.generate_noise_vector(num_samples=num_samples, snr_db=snr_db,
                                                             pss_sequence_td=pss_sequences_td[0])
            samples = samples + noise

        self.samples = samples

    def __len__(self) -> int:
        """
        Return the number of windows in the dataset.

        :return: number of windows
        """
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return one dataset item.

        :param idx: index of the window to fetch
        :return: (complex window of shape (FFT_SIZE,), scalar label in 0..3, scalar
            frequency offset in FFT bins that was applied to that window)
        """
        return self.samples[idx], self.labels[idx], self.freq_offset[idx]

    @classmethod
    def pooled_loader(cls, snrs: Sequence[float], samples_per_snr: int, batch_size: int, seed: int,
                      shuffle: bool = True, random_phase: bool = False) -> DataLoader:
        """
        Build one loader over windows drawn at every SNR in ``snrs``.

        Training on pooled SNRs is what makes one set of weights usable across the
        whole sweep instead of one model per operating point.

        :param snrs: SNRs in dB, one sub-dataset each
        :param samples_per_snr: windows generated per SNR
        :param batch_size: windows per batch
        :param seed: seeds data generation; the loader's shuffling uses ``seed + 1``,
            keeping the two streams distinct
        :param shuffle: shuffle batches; training needs it, validation does not
        :param random_phase: passed to every :class:`PssDataset`
        :return: a DataLoader over ``len(snrs) * samples_per_snr`` windows
        """
        # fork_rng isolates this seed from the caller's global RNG state, so model
        # initialization cannot change the data and vice versa.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            dataset = ConcatDataset([cls(samples_per_snr, snr_db=snr, random_phase=random_phase)
                                     for snr in snrs])
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          generator=torch.Generator().manual_seed(seed + 1))
