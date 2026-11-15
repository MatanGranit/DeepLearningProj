"""Test stage: threshold calibration, detection metrics and arithmetic cost.

The measurement protocol :class:`Evaluator` implements:

1. :meth:`~Evaluator.calibrate_thresholds` picks one presence threshold per receiver
   from *noise-only* windows, so no test data and no signal influence the operating
   point.
2. :meth:`~Evaluator.evaluate_receivers` freezes those thresholds and scores every
   receiver on the same test windows, accumulating a 4x4 confusion matrix.
3. :meth:`DetectionMetrics.from_confusion` reduces that matrix to accuracy, Pd and Pfa.
4. :meth:`~Evaluator.arithmetic_cost` estimates each receiver's operations per window.
"""
from typing import Dict

import torch
from torch.utils.data import DataLoader

from .data.pss_signal import NOISE_LABEL, NUM_CLASSES, PssSignalGenerator
from .metrics import DetectionMetrics
from .models.networks import PssDetector
from .models.receiver import Receiver

# Calibration draws offsets on every bin below this bound, unlike the dataset's
# coarse grid; the exact correction removes any offset.
OFFSET_BOUND = 1920
# Noise power for calibrations that need one reference value: every score in this
# project is scale invariant, so the threshold does not depend on it.
REFERENCE_SNR_DB = -30.
DEFAULT_CALIBRATION_SAMPLES = 10000
DEFAULT_TARGET_PFA = 0.01


class Evaluator:
    """Calibrate, test and cost receivers. Stateless; every method is static."""

    @staticmethod
    def calibrate_thresholds(models: Dict[str, Receiver], snr_db: float = REFERENCE_SNR_DB,
                             num_samples: int = DEFAULT_CALIBRATION_SAMPLES, target_pfa: float = DEFAULT_TARGET_PFA,
                             batch_size: int = 256) -> Dict[str, float]:
        """
        Pick one presence threshold per receiver from identical, independent noise windows.

        No signal and no test data are involved. Every score in this project is
        scale invariant, so a threshold calibrated at one noise power holds at all.

        :param models: name -> receiver; every entry is calibrated on the same windows
        :param snr_db: SNR argument handed to the noise generator. It only sets the
            absolute noise power here (no signal is generated), so the default is
            arbitrary.
        :param num_samples: noise-only windows to draw
        :param target_pfa: desired false-alarm probability, strictly between 0 and 1
        :param batch_size: windows scored per iteration
        :return: name -> threshold. Neural detectors also get it written into their
            ``detection_threshold`` buffer, so it survives into their checkpoint.
        :raises ValueError: unless 0 < target_pfa < 1 and num_samples > 0
        """
        if not 0 < target_pfa < 1 or num_samples < 1:
            raise ValueError("Require 0 < target_pfa < 1 and num_samples > 0")
        scores = {name: [] for name in models}
        # Only used for its length; no signal is transmitted.
        reference = PssSignalGenerator.generate_pss(0)
        for model in models.values():
            model.eval()
        with torch.inference_mode():
            for start in range(0, num_samples, batch_size):
                size = min(batch_size, num_samples - start)
                noise = PssSignalGenerator.generate_noise_vector(size, snr_db, reference)
                offsets = torch.randint(0, OFFSET_BOUND, (size,))
                # Include the same frequency-correction path used in testing.
                corrected = PssSignalGenerator.correct_frequency_offset(noise, offsets)
                for name, model in models.items():
                    window = corrected if model.RECEIVES_CORRECTED_WINDOW else noise
                    score, _ = model.presence_and_id(model(window.to(model.device)))
                    scores[name].append(score.cpu())
        thresholds = {}
        for name, parts in scores.items():
            threshold = Evaluator.threshold_for_pfa(torch.cat(parts), target_pfa)
            thresholds[name] = threshold
            if isinstance(models[name], PssDetector):
                models[name].detection_threshold.fill_(threshold)
        return thresholds

    @staticmethod
    def threshold_for_pfa(noise_scores: torch.Tensor, target_pfa: float) -> float:
        """
        Return the presence threshold that lets through ``target_pfa`` of these noise-only scores.

        :param noise_scores: presence scores of noise-only windows, shape (count,)
        :param target_pfa: desired false-alarm probability, strictly between 0 and 1
        :return: the threshold
        """
        # The (1 - target_pfa) quantile is by definition that threshold. "higher"
        # interpolation rounds conservatively, never under-shooting the threshold.
        return torch.quantile(noise_scores, 1 - target_pfa, interpolation="higher").item()

    @staticmethod
    def evaluate_receivers(models: Dict[str, Receiver], loader: DataLoader, thresholds: Dict[str, float],
                           raw_diagnostics: bool = False) -> Dict[str, Dict]:
        """
        Score every receiver on the same test windows at its frozen threshold.

        Every receiver sees identical IQ, frequency corrected exactly for every receiver
        that is handed a corrected window, so differences between them are receiver
        differences only.

        :param models: name -> receiver, keyed as in ``thresholds``
        :param loader: yields ``(samples, labels, freq_offset)`` batches
        :param thresholds: name -> presence threshold from :meth:`calibrate_thresholds`
        :param raw_diagnostics: also report what a plain 4-way ``argmax`` would have
            given for the neural detectors, under the extra key ``raw_argmax``
        :return: name -> the dict :meth:`DetectionMetrics.from_confusion` produces. A
            receiver with ``REPORTS_OFFSET`` also gets ``offset``: of the signal windows,
            how many were detected and how many of those got the right frequency offset
            (:meth:`DetectionMetrics.offset_rates`)
        """
        confusion = {name: torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long) for name in models}
        # name -> [signal windows, detected, detected with the right offset]
        offset_counts = {name: [0, 0, 0] for name, model in models.items() if model.REPORTS_OFFSET}
        raw = {name: torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
               for name, model in models.items() if isinstance(model, PssDetector)}
        for model in models.values():
            model.eval()
        with torch.inference_mode():
            for samples, labels, offsets in loader:
                # Corrected once per batch, then shared by every receiver that is handed it.
                corrected = PssSignalGenerator.correct_frequency_offset(samples, offsets)
                for name, model in models.items():
                    window = corrected if model.RECEIVES_CORRECTED_WINDOW else samples
                    if model.REPORTS_OFFSET:
                        output, best_offsets = model.forward_with_offsets(window.to(model.device))
                    else:
                        output = model(window.to(model.device))
                    predicted = model.decide(output, thresholds[name])
                    if model.REPORTS_OFFSET:
                        found = model.reported_offsets(predicted, best_offsets).cpu()
                        signal = labels != NOISE_LABEL
                        detected = signal & (predicted.cpu() != NOISE_LABEL)
                        counts = offset_counts[name]
                        counts[0] += int(signal.sum())
                        counts[1] += int(detected.sum())
                        counts[2] += int((detected & (found == offsets)).sum())
                    if raw_diagnostics and name in raw:
                        raw[name] += Evaluator.confusion_counts(labels, output.argmax(1))
                    confusion[name] += Evaluator.confusion_counts(labels, predicted)
        result = {name: DetectionMetrics.from_confusion(matrix) for name, matrix in confusion.items()}
        for name, counts in offset_counts.items():
            result[name]["offset"] = DetectionMetrics.offset_rates(*counts)
        if raw_diagnostics:
            for name in raw:
                result[name]["raw_argmax"] = DetectionMetrics.from_confusion(raw[name])
        return result

    @staticmethod
    def confusion_counts(labels: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
        """
        Build a 4x4 confusion matrix (rows: truth, columns: prediction) for one batch.

        :param labels: true classes, shape (batch,)
        :param predicted: predicted classes, shape (batch,), on any device
        :return: int64 tensor of shape (4, 4)
        """
        # bincount over 4*truth + prediction is a flattened 4x4 histogram.
        return torch.bincount(NUM_CLASSES * labels.cpu() + predicted.cpu(),
                              minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)

    @staticmethod
    def arithmetic_cost(model: Receiver) -> Dict[str, int]:
        """
        Estimate the real arithmetic of one receive, after the shared frequency correction.

        A multiply-accumulate counts as two operations and everything else as one
        (see :meth:`Receiver.operation_counts`). Fixed template initialization is
        offline and not counted.

        :param model: the receiver to cost
        :return: dict with ``conv_linear_macs``, ``other_real_ops``, ``total_real_ops``,
            ``parameters`` and ``stored_bytes``
        """
        macs, overhead = model.operation_counts()
        return {"conv_linear_macs": int(macs), "other_real_ops": int(overhead),
                "total_real_ops": int(2*macs+overhead),
                "parameters": sum(p.numel() for p in model.parameters()),
                "stored_bytes": model.stored_bytes()}
