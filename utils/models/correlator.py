"""Classical matched-filter baselines: magnitude and coherent PSS correlators.

Both are implemented as a single real matrix multiply so they run on the same
batched, device-agnostic path as the neural detectors and can be timed fairly
against them. Neither has trainable parameters.
"""
from typing import Tuple

import torch

from ..data.pss_signal import FFT_SIZE, NOISE_LABEL, NUM_CLASSES, NUM_PSS_IDS, PssSignalGenerator
from .receiver import NORM_FLOOR, Receiver

# Default for the standalone find_cell_id(). The experiments never use it: they
# calibrate a threshold at a measured false-alarm rate instead.
DEFAULT_THRESHOLD = 0.053


class Correlator(Receiver):
    """Normalized correlation against the three clean PSS templates.

    Two modes, selected at construction:

    * ``coherent=False`` (magnitude) -- scores are ``|<template, x>|``, so the
      result is unchanged by an arbitrary common phase. This is the practical
      classical baseline when the phase is unknown.
    * ``coherent=True`` -- scores are ``Re<template, x>``, the true matched filter.
      It is optimal when the phase matches the templates and degrades badly when it
      does not, which is why the random-phase experiment keeps it only as a
      deliberately mismatched reference.

    Scores are divided by the input norm, so they are scale invariant and a single
    threshold holds across noise powers.
    """

    def __init__(self, *, coherent: bool = False) -> None:
        """
        Build the correlator's fixed projection matrix.

        :param coherent: True selects the phase-sensitive coherent correlator,
            False the phase-invariant magnitude correlator
        """
        super().__init__()
        self.fft_binsize = FFT_SIZE
        self.coherent = coherent
        templates = self.reference_templates()
        # Real rows compute real and imaginary parts of <template, sample>.
        # Writing the complex inner product as a real matrix multiply keeps the
        # whole receiver in one torch.nn.functional.linear call.
        real_rows = torch.view_as_real(templates).reshape(NUM_PSS_IDS, -1)
        imag_rows = torch.stack((-templates.imag, templates.real), dim=-1).reshape(NUM_PSS_IDS, -1)
        # Coherent mode needs only the real part, so it stores half the rows and
        # costs half the multiply-accumulates.
        self.register_buffer("projection", real_rows if coherent else torch.cat((real_rows, imag_rows)))

    @staticmethod
    def reference_templates() -> torch.Tensor:
        """
        Return the unit-norm templates the correlator projects onto.

        :return: complex tensor of shape (NUM_PSS_IDS, samples correlated)
        """
        return PssSignalGenerator.pss_templates(normalized=True)

    def forward(self, corrected_samples: torch.Tensor) -> torch.Tensor:
        """
        Score already frequency-corrected samples; vectorized and device-safe.

        :param corrected_samples: complex tensor of shape (batch, samples correlated) whose
            frequency offset has already been removed: FFT_SIZE here, the 128-sample view
            for :class:`NarrowbandCorrelator`
        :return: real tensor of shape (batch, 3), one normalized correlation score
            per PSS identity
        """
        real = torch.view_as_real(corrected_samples).reshape(corrected_samples.shape[0], -1)
        scores = torch.nn.functional.linear(real, self.projection)
        if not self.coherent:
            # Six rows in, three out: sqrt(re^2 + im^2) discards the phase.
            scores = (scores[:, :NUM_PSS_IDS].square() + scores[:, NUM_PSS_IDS:].square()).sqrt()
        norm = real.square().sum(dim=1, keepdim=True).clamp_min(NORM_FLOOR).sqrt()
        return scores / norm

    def presence_and_id(self, output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the presence score and identity; see :meth:`Receiver.presence_and_id`.

        :param output: correlation scores of shape (batch, 3)
        :return: (largest correlation, the identity that produced it)
        """
        return output.max(dim=1)

    def operation_counts(self) -> Tuple[int, int]:
        """
        Count the real arithmetic per window; see :meth:`Receiver.operation_counts`.

        :return: (projection MACs, input norm plus per-score magnitude and division)
        """
        n = self.projection.shape[1] // 2  # complex samples correlated
        # Three template rows coherent, six for magnitude; 2N real values in.
        macs = (NUM_PSS_IDS if self.coherent else 2 * NUM_PSS_IDS) * 2 * n
        # 4N for the input norm, plus the per-score magnitude and division.
        return macs, 4 * n + (3 if self.coherent else 15)

    def correlation_scores(self, freq_offset: torch.Tensor, td_samples: torch.Tensor) -> torch.Tensor:
        """
        Correct the frequency offset, then return normalized correlation scores.

        :param freq_offset: offsets in FFT bins, shape (batch,)
        :param td_samples: raw complex windows of shape (batch, FFT_SIZE), still
            carrying their frequency offset
        :return: real tensor of shape (batch, 3)
        :raises ValueError: on a wrong window shape or mismatched batch sizes
        """
        if td_samples.ndim != 2 or td_samples.shape[1] != self.fft_binsize:
            raise ValueError(f"td_samples must have shape (batch, {self.fft_binsize})")
        if freq_offset.shape[0] != td_samples.shape[0]:
            raise ValueError("freq_offset and td_samples must have the same batch size")
        return self(PssSignalGenerator.correct_frequency_offset(td_samples, freq_offset))

    def find_cell_id(self, freq_offset: torch.Tensor, td_samples: torch.Tensor,
                     threshold: float = DEFAULT_THRESHOLD) -> torch.Tensor:
        """
        Classify windows by maximum correlation, as a one-hot four-class tensor.

        The identity with the largest correlation wins; if that correlation is not
        above ``threshold`` the window is declared noise only (class 3).

        :param freq_offset: offsets in FFT bins, shape (batch,)
        :param td_samples: raw complex windows of shape (batch, FFT_SIZE)
        :param threshold: correlation needed to declare a signal present
        :return: int64 one-hot tensor of shape (batch, 4)
        """
        batch_size = freq_offset.shape[0]
        output = torch.zeros((batch_size, NUM_CLASSES), dtype=torch.int64, device=td_samples.device)
        max_vals, predicted_classes = torch.max(self.correlation_scores(freq_offset, td_samples), dim=1)
        predicted_classes[max_vals <= threshold] = NOISE_LABEL
        output[torch.arange(batch_size, device=td_samples.device), predicted_classes] = 1
        return output


class NarrowbandCorrelator(Correlator):
    """Magnitude correlator on the 128-sample view of the window (:meth:`PssSignalGenerator.narrowband`).

    It is handed the frequency-corrected 2048-sample window like every receiver and
    keeps the 128 bins that hold the PSS; that band selection, the receiver's front
    end, is not counted in its operations. It is counted as the direct correlation,
    without the conjugate-symmetry pairing :class:`NarrowbandPssCnn` uses.
    """

    def __init__(self) -> None:
        """Build the magnitude correlator over the 128-sample templates."""
        super().__init__()

    @staticmethod
    def reference_templates() -> torch.Tensor:
        """
        Return the unit-norm 128-sample templates.

        :return: complex tensor of shape (NUM_PSS_IDS, NARROWBAND_SIZE)
        """
        templates = PssSignalGenerator.narrowband(PssSignalGenerator.pss_templates())
        return templates / templates.norm(dim=1, keepdim=True)

    def forward(self, corrected_samples: torch.Tensor) -> torch.Tensor:
        """
        Score frequency-corrected windows on their 128-sample view.

        :param corrected_samples: complex tensor of shape (batch, FFT_SIZE), frequency corrected
        :return: real tensor of shape (batch, 3), one normalized correlation score per PSS identity
        """
        return super().forward(PssSignalGenerator.narrowband(corrected_samples))
