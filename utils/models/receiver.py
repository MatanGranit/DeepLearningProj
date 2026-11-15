"""The interface every PSS receiver implements: neural detectors and correlators alike.

Evaluation, calibration and cost accounting score every receiver through this
interface. Trained detectors additionally derive from
:class:`~utils.models.networks.PssDetector`, which the evaluator checks to store a
calibrated threshold and report raw-argmax diagnostics.
"""
import itertools
from typing import Tuple, Union

import torch
import torch.nn as nn

from ..data.pss_signal import NOISE_LABEL

# Lower bound on a squared norm before dividing by it: keeps an all-zero window finite.
NORM_FLOOR = 1e-20


class Receiver(nn.Module):
    """Base class for a receiver scoring ``(batch, FFT_SIZE)`` windows."""

    # True: the receiver is handed each window with its known frequency offset removed.
    # False: it gets the window as received and must find the offset itself.
    RECEIVES_CORRECTED_WINDOW = True
    # True: the receiver also reports the frequency offset it found, through forward_with_offsets.
    REPORTS_OFFSET = False

    def presence_and_id(self, output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split this receiver's ``forward`` output into a presence score and an identity.

        :param output: what ``forward`` returned for a batch
        :return: (presence score, best PSS identity), both of shape (batch,)
        :raises NotImplementedError: in the base class
        """
        raise NotImplementedError

    def decide(self, output: torch.Tensor, threshold: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Turn a ``forward`` output into four-class labels at a presence threshold.

        :param output: what ``forward`` returned for a batch
        :param threshold: presence threshold, normally calibrated on noise-only data
            for a target false-alarm rate
        :return: long tensor of shape (batch,) with 0..2 for a detected identity and
            3 for "no signal"
        """
        score, predicted_id = self.presence_and_id(output)
        return torch.where(score > threshold, predicted_id, NOISE_LABEL)

    def stored_bytes(self) -> int:
        """
        Count the bytes a deployed receiver stores: its parameters and buffers.

        :return: bytes
        """
        return sum(t.numel() * t.element_size() for t in list(self.parameters()) + list(self.buffers()))

    def operation_counts(self) -> Tuple[int, int]:
        """
        Count the real arithmetic of one forward pass on one window.

        Multiply-accumulates are reported separately because they count as two
        operations; every other multiplication, addition, division or sqrt counts
        as one. Comparisons, memory movement and the shared frequency correction
        are excluded.

        :return: (multiply-accumulates, other real operations)
        :raises NotImplementedError: in the base class
        """
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        """The device holding this receiver's weights (or, for a correlator, its templates)."""
        return next(itertools.chain(self.parameters(), self.buffers())).device
