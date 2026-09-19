import sys
import unittest
import io
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "utils"))

from correlator import Correlator
from data_loader import PssDataset, generate_noise_vector, generate_pss
from net import EfficientPssCnn, correct_frequency_offset
from evaluation import calibrate_thresholds, metrics_from_confusion

torch.set_num_threads(1)


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

    def test_cnn_parameter_budget(self):
        trainable_parameters = sum(
            parameter.numel()
            for parameter in EfficientPssCnn().parameters()
            if parameter.requires_grad
        )
        self.assertLess(trainable_parameters, 2_000)

    def test_cnn_is_invariant_to_input_scale(self):
        dataset = PssDataset(num_samples=8, snr_db=-20.0)
        corrected = correct_frequency_offset(
            dataset.samples,
            dataset.freq_offset,
            torch.arange(2048),
        )
        model = EfficientPssCnn().eval()
        with torch.no_grad():
            original_output = model(corrected)
            scaled_output = model(10.0 * corrected)
        self.assertTrue(
            torch.allclose(original_output, scaled_output, atol=1e-5, rtol=1e-4)
        )

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

    def test_vectorized_correlator_matches_complex_formula(self):
        x = torch.randn(10, 2048, dtype=torch.complex64)
        expected = torch.stack([
            (x*generate_pss(i).conj()).sum(1).abs() /
            (x.norm(dim=1)*generate_pss(i).norm()) for i in range(3)], dim=1)
        self.assertTrue(torch.allclose(Correlator()(x), expected, atol=1e-6))

    def test_presence_detection_is_not_id_accuracy(self):
        # Two PSS examples: one wrong ID, one missed. Two noise examples: one FA.
        matrix = torch.tensor([[0, 1, 0, 0], [0, 0, 0, 1],
                               [0, 0, 0, 0], [1, 0, 0, 1]])
        metrics = metrics_from_confusion(matrix)
        self.assertEqual(metrics['pd'], .5)
        self.assertEqual(metrics['pfa'], .5)
        self.assertEqual(metrics['accuracy'], .25)

    def test_calibration_controls_independent_noise_and_checkpoint_preserves_it(self):
        model = EfficientPssCnn().eval()
        with self.assertRaises(RuntimeError):
            model.predict(torch.zeros(1, 2048, dtype=torch.complex64))
        calibrate_thresholds({'compact': model}, num_samples=20000)
        for seed, snr in [(310, -50.), (311, -20.), (312, 0.)]:
            torch.manual_seed(seed)
            noise = generate_noise_vector(10000, snr, generate_pss(0))
            with torch.no_grad():
                rate = (model.predict(noise) != 3).float().mean().item()
            self.assertGreater(rate, .004)
            self.assertLess(rate, .018)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = EfficientPssCnn().eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        with torch.no_grad():
            self.assertTrue(torch.equal(model.predict(noise[:16]), restored.predict(noise[:16])))

    def test_high_snr_and_zero_input(self):
        model = EfficientPssCnn().eval()
        calibrate_thresholds({'compact': model}, num_samples=5000)
        data = PssDataset(2048, snr_db=0.)
        corrected = correct_frequency_offset(data.samples, data.freq_offset, torch.arange(2048))
        with torch.no_grad():
            self.assertGreater((model.predict(corrected) == data.labels).float().mean().item(), .985)
            zeros = torch.zeros(2, 2048, dtype=torch.complex64)
            self.assertTrue(torch.isfinite(model(zeros)).all())
            self.assertTrue((model.predict(zeros) == 3).all())

    def test_backward_gradients_are_finite(self):
        model = EfficientPssCnn().train()
        data = PssDataset(32, snr_db=-26.)
        x = correct_frequency_offset(data.samples, data.freq_offset, torch.arange(2048))
        loss = torch.nn.functional.cross_entropy(model(x), data.labels)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
