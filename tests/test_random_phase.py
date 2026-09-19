import math
import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "utils"))

from benchmark import arithmetic_cost
from correlator import Correlator
from data_loader import PssDataset
from net import PhaseInvariantPssCnn, correct_frequency_offset
from phase_benchmark import phase_cnn_cost

torch.set_num_threads(1)


class RandomPhaseTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_dataset_phase_preserves_signal_energy(self):
        torch.manual_seed(81)
        fixed = PssDataset(128, snr_db=None, random_phase=False)
        torch.manual_seed(81)
        rotated = PssDataset(128, snr_db=None, random_phase=True)
        self.assertTrue(torch.equal(fixed.labels, rotated.labels))
        self.assertTrue(torch.equal(fixed.freq_offset, rotated.freq_offset))
        torch.testing.assert_close(fixed.samples.abs(), rotated.samples.abs())
        self.assertFalse(torch.allclose(fixed.samples[fixed.labels < 3],
                                        rotated.samples[rotated.labels < 3]))

    def test_phase_cnn_is_invariant_by_construction(self):
        model = PhaseInvariantPssCnn().eval()
        x = torch.randn(23, 2048, dtype=torch.complex64)
        phase = torch.rand(23, 1)*2*math.pi
        with torch.no_grad():
            original = model(x)
            rotated = model(x*torch.exp(1j*phase))
        torch.testing.assert_close(original, rotated, rtol=2e-5, atol=2e-5)

    def test_phase_cnn_budget_shape_and_noise_free_ids(self):
        model = PhaseInvariantPssCnn().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 3085)
        # The design target is the original project's 44,039-op detector.
        # The ideal coherent correlator remains a useful lower-cost oracle,
        # but it is phase-sensitive and is not a practical random-phase receiver.
        self.assertLess(phase_cnn_cost(model)["total_real_ops"], 44039)
        data = PssDataset(512, snr_db=None, random_phase=True)
        corrected = correct_frequency_offset(data.samples, data.freq_offset,
                                             torch.arange(2048))
        with torch.no_grad():
            output = model(corrected)
        self.assertEqual(output.shape, (512, 4))
        signal = data.labels < 3
        self.assertTrue(torch.equal(output[signal, :3].argmax(1),
                                    data.labels[signal]))

    def test_magnitude_is_phase_invariant_but_coherent_is_not(self):
        x = torch.randn(16, 2048, dtype=torch.complex64)
        rotated = x*torch.exp(torch.tensor(1.1j))
        with torch.no_grad():
            torch.testing.assert_close(Correlator()(x), Correlator()(rotated),
                                       rtol=2e-5, atol=2e-5)
            difference = (Correlator(coherent=True)(x)
                          - Correlator(coherent=True)(rotated)).abs().max()
        self.assertGreater(float(difference), .005)


if __name__ == "__main__":
    unittest.main()
