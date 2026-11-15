"""Unknown-bin checks: the convolutional scanning CNN computes the offset-searched correlation, within budget.

``ConvScanningPssCnn`` gets the window with its frequency offset still in it. Its global
depthwise convolution and dilated convolutions must equal the correlation with every
template shifted to every allowed offset, so that it finds the PSS wherever the dataset
put it, and it must cost no more arithmetic and store no more than the magnitude correlator.

Run with ``python -m unittest discover -s tests -v``.
"""
import sys
import unittest
from pathlib import Path

import torch

# Make the utils package importable however the tests are launched.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.data.pss_dataset import FREQ_OFFSETS  # noqa: E402 (needs the sys.path entry above)
from utils.data.pss_signal import FFT_SIZE, PssSignalGenerator  # noqa: E402 (needs the sys.path entry above)
from utils.evaluation import Evaluator  # noqa: E402 (needs the sys.path entry above)
from utils.models.correlator import Correlator  # noqa: E402 (needs the sys.path entry above)
from utils.models.networks import ConvScanningPssCnn  # noqa: E402 (needs the sys.path entry above)

# Single-threaded, for reproducibility.
torch.set_num_threads(1)


class UnknownBinTests(unittest.TestCase):
    """Checks on the scanning CNN."""

    def setUp(self) -> None:
        """Fix the global RNG so every test starts from the same state."""
        torch.manual_seed(23)

    @staticmethod
    def shifted(identity: int, offset: int, phase: float = 0.0) -> torch.Tensor:
        """
        Build one unit-norm PSS window at a frequency offset with a common phase.

        :param identity: PSS identity, 0..2
        :param offset: frequency offset in FFT bins
        :param phase: common phase in radians (default: 0)
        :return: complex tensor of shape (1, FFT_SIZE)
        """
        template = PssSignalGenerator.pss_templates(normalized=True)[identity]
        shift = PssSignalGenerator.frequency_shift(torch.tensor([offset]), FFT_SIZE)
        return template * shift * torch.exp(torch.tensor(1j * phase))

    def test_scan_equals_offset_searched_correlation(self) -> None:
        """Every identity score is the largest |<t shifted by k, x>|^2 over the offsets, times the gain over L1^2."""
        model = ConvScanningPssCnn().eval()
        x = self.shifted(2, FREQ_OFFSETS[9], phase=1.1) + 0.3 * torch.randn(1, FFT_SIZE, dtype=torch.complex64)
        templates = PssSignalGenerator.pss_templates(normalized=True)
        shifts = PssSignalGenerator.frequency_shift(torch.tensor(FREQ_OFFSETS), FFT_SIZE)
        direct = ((templates[:, None] * shifts[None]).conj() * x).sum(dim=2).abs().square().max(dim=1).values
        with torch.no_grad():
            scores = model(x)[0, :3] * model.l1_norm(x)[0].square() / model.gain
        torch.testing.assert_close(scores, direct, rtol=1e-3, atol=1e-5)

    def test_finds_the_pss_at_every_offset(self) -> None:
        """A clean PSS at any allowed offset and phase is identified, and the score does not depend on the offset."""
        model = ConvScanningPssCnn().eval()
        peaks = []
        with torch.no_grad():
            for index, offset in enumerate(FREQ_OFFSETS):
                identity = index % 3
                logits = model(self.shifted(identity, offset, phase=0.4 * index))
                self.assertEqual(int(logits[0, :3].argmax()), identity)
                peaks.append(float(logits[0, identity]))
        self.assertLess(max(peaks) / min(peaks) - 1, 0.05)

    def test_reports_the_offset_it_found(self) -> None:
        """predict_with_offset gives the identity and the frequency offset of a clear PSS at every offset."""
        model = ConvScanningPssCnn().eval()
        Evaluator.calibrate_thresholds({"cnn": model}, num_samples=3000)
        x = torch.cat([self.shifted(index % 3, offset, phase=0.5 * index) for index, offset in enumerate(FREQ_OFFSETS)])
        x = 10 * x + torch.randn_like(x) / 45  # about 0 dB per subcarrier: far above the detection knee
        labels, offsets = model.predict_with_offset(x)
        self.assertEqual(labels.tolist(), [index % 3 for index in range(len(FREQ_OFFSETS))])
        self.assertEqual(offsets.tolist(), list(FREQ_OFFSETS))
        _, noise_offsets = model.predict_with_offset(torch.zeros(2, FFT_SIZE, dtype=torch.complex64))
        self.assertEqual(noise_offsets.tolist(), [-1, -1])

    def test_dilated_layers_are_a_bit_reversed_dft(self) -> None:
        """The seven fixed dilated convolutions compute the 128-point DFT, in bit-reversed order."""
        model = ConvScanningPssCnn()
        u = torch.randn(2, 3, 128, dtype=torch.complex64)
        reverse = torch.tensor([int(format(k, "07b")[::-1], 2) for k in range(128)])
        torch.testing.assert_close(model.dilated_convolutions(u), torch.fft.fft(u)[..., reverse], rtol=1e-4, atol=1e-4)

    def test_budget_within_the_magnitude_correlator(self) -> None:
        """The CNN needs fewer operations and stores fewer bytes than the magnitude correlator it is compared with."""
        conv = Evaluator.arithmetic_cost(ConvScanningPssCnn())
        correlator = Evaluator.arithmetic_cost(Correlator())
        self.assertEqual(conv["total_real_ops"], 56837)
        self.assertLess(conv["total_real_ops"], correlator["total_real_ops"])
        self.assertLess(conv["stored_bytes"], correlator["stored_bytes"])
        self.assertFalse(ConvScanningPssCnn.RECEIVES_CORRECTED_WINDOW)
        self.assertTrue(Correlator.RECEIVES_CORRECTED_WINDOW)


if __name__ == "__main__":
    unittest.main()
