"""Detection metrics: reduce a 4x4 confusion matrix to accuracy, Pd and Pfa.

Pure arithmetic on counts, shared by the test stage (:mod:`utils.evaluation`) and the
reporting stage (:mod:`utils.reporting`), which pools stored matrices across seeds.
"""
import math
from typing import Dict, List

import torch

from .data.pss_signal import NOISE_LABEL, NUM_PSS_IDS


class DetectionMetrics:
    """Rates and confidence intervals from confusion counts. Stateless; every method is static."""

    @staticmethod
    def from_confusion(confusion: torch.Tensor) -> Dict:
        """
        Reduce a confusion matrix to accuracy, Pd and Pfa.

        Rows are the truth and columns the prediction. A signal reported with the
        wrong PSS identity still counts as a detection in Pd, but not in accuracy.

        :param confusion: 4x4 integer tensor; rows 0..2 are the PSS identities and row
            3 is noise-only
        :return: dict with ``accuracy``, ``pd``, ``pfa``, ``correct_id_given_signal``,
            the ``signal_count``/``noise_count`` the rates were computed from, 95%
            Wilson intervals ``pd_ci95``/``pfa_ci95``, and the matrix itself
        :raises ValueError: if the matrix is empty or lacks either signal or noise rows,
            which would make Pd or Pfa undefined
        """
        total = int(confusion.sum())
        signals = int(confusion[:NUM_PSS_IDS].sum())
        noise = int(confusion[NOISE_LABEL].sum())
        correct = int(confusion.diagonal().sum())
        # Any signal predicted as any signal counts as a detection: the top-left 3x3
        # block. Only its diagonal is a correct identification.
        detected = int(confusion[:NUM_PSS_IDS, :NUM_PSS_IDS].sum())
        false_alarms = int(confusion[NOISE_LABEL, :NUM_PSS_IDS].sum())
        correct_id = int(confusion[:NUM_PSS_IDS, :NUM_PSS_IDS].diagonal().sum())
        if not total or not signals or not noise:
            raise ValueError("Evaluation requires signal-present and noise-only examples")
        return {"accuracy": correct/total, "pd": detected/signals,
                "pfa": false_alarms/noise, "correct_id_given_signal": correct_id/signals,
                "signal_count": signals, "noise_count": noise,
                "pd_ci95": DetectionMetrics.wilson_interval(detected, signals),
                "pfa_ci95": DetectionMetrics.wilson_interval(false_alarms, noise),
                "confusion": confusion.tolist()}

    @staticmethod
    def offset_rates(signals: int, detected: int, right_offset: int) -> Dict:
        """
        Reduce frequency-offset counts to rates.

        :param signals: signal windows scored
        :param detected: of those, the ones the receiver detected
        :param right_offset: of those detected, the ones reported at the true offset
        :return: dict with the three counts, ``correct_given_detection`` (right / detected)
            and ``detected_with_offset`` (right / signals)
        """
        return {"signals": signals, "detected": detected, "right_offset": right_offset,
                "correct_given_detection": right_offset / detected if detected else float("nan"),
                "detected_with_offset": right_offset / signals}

    @staticmethod
    def wilson_interval(successes: int, total: int, z: float = 1.96) -> List[float]:
        """
        Compute the Wilson score confidence interval of a binomial proportion.

        Preferred over the normal approximation because Pfa sits near 1%, where the
        naive interval can run below zero.

        :param successes: number of successes observed
        :param total: number of trials; 0 yields ``[nan, nan]``
        :param z: normal quantile; 1.96 gives a 95% interval
        :return: ``[lower, upper]``, clipped to [0, 1]
        """
        if total == 0:
            return [float("nan"), float("nan")]
        p = successes / total
        denominator = 1 + z*z / total
        centre = (p + z*z/(2*total)) / denominator
        radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
        return [max(0., centre-radius), min(1., centre+radius)]
