import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "utils"))

from correlator import Correlator
from data_loader import PssDataset, generate_noise_vector, generate_pss
from net import EfficientPssCnn, correct_frequency_offset


class SignalPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_generated_noise_has_requested_snr(self):
        pss = generate_pss(0)
        signal_power = torch.mean(torch.abs(pss) ** 2)

        for requested_snr in (0.0, -20.0, -50.0):
            noise = generate_noise_vector(512, requested_snr, pss)
            noise_power = torch.mean(torch.abs(noise) ** 2)
            measured_snr = 10.0 * torch.log10(signal_power / noise_power)
            self.assertAlmostEqual(measured_snr.item(), requested_snr, delta=0.15)

    def test_dataset_has_scalar_four_class_labels(self):
        dataset = PssDataset(num_samples=512, snr_db=None)
        self.assertEqual(dataset.labels.ndim, 1)
        self.assertTrue(set(dataset.labels.tolist()).issubset({0, 1, 2, 3}))
        self.assertIn(3, dataset.labels.tolist())
        self.assertTrue(torch.all(dataset.samples[dataset.labels == 3] == 0))

    def test_cnn_output_shape(self):
        dataset = PssDataset(num_samples=8, snr_db=-20.0)
        corrected = correct_frequency_offset(
            dataset.samples,
            dataset.freq_offset,
            torch.arange(2048),
        )
        output = EfficientPssCnn()(corrected)
        self.assertEqual(output.shape, (8, 4))

    def test_correlator_is_exact_without_noise(self):
        dataset = PssDataset(num_samples=256, snr_db=None)
        correlator = Correlator()
        scores = correlator.correlation_scores(
            dataset.freq_offset,
            dataset.samples,
        )
        self.assertEqual(scores.shape, (256, 3))
        predicted = correlator.find_cell_id(
            dataset.freq_offset,
            dataset.samples,
        ).argmax(dim=1)
        self.assertTrue(torch.equal(predicted, dataset.labels))


if __name__ == "__main__":
    unittest.main()
