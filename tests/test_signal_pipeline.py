"""Signal-generation and inference checks for the pipeline.

Covers the properties the experiments silently depend on: that the generated noise
really carries the requested SNR, that the dataset is well formed, that the
detector is numerically sound, and that threshold calibration actually delivers
the false-alarm rate it targets.

Run with ``python -m unittest discover -s tests -v``.
"""
import sys
import unittest
import io
from pathlib import Path

import torch


# Make the utils package importable however the tests are launched.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.models.correlator import Correlator  # noqa: E402 (needs the sys.path entry above)
from utils.data.pss_dataset import PssDataset  # noqa: E402 (needs the sys.path entry above)
from utils.data.pss_signal import PssSignalGenerator  # noqa: E402 (needs the sys.path entry above)
from utils.evaluation import Evaluator  # noqa: E402 (needs the sys.path entry above)
from utils.metrics import DetectionMetrics  # noqa: E402 (needs the sys.path entry above)
from utils.models.networks import NarrowbandPssCnn  # noqa: E402 (needs the sys.path entry above)
from utils.correlation_figure import PssCorrelationFigure  # noqa: E402 (needs the sys.path entry above)

# Single-threaded: keeps the RNG-sensitive tests reproducible.
torch.set_num_threads(1)


class SignalPipelineTests(unittest.TestCase):
    """End-to-end checks over generation, the detector and the metrics."""

    def setUp(self) -> None:
        """Fix the global RNG so every test starts from the same state."""
        torch.manual_seed(0)

    def test_generated_noise_has_requested_snr(self) -> None:
        """The measured SNR per PSS subcarrier matches the requested SNR to within 0.15 dB.

        This is the executable form of the contract in ``generate_noise_vector``'s
        docstring: the power of one occupied FFT bin of the PSS over the noise power
        in one FFT bin.
        """
        pss = PssSignalGenerator.generate_pss(0)
        spectrum = torch.fft.fft(pss).abs().square()
        occupied = spectrum > 0.5
        self.assertEqual(int(occupied.sum()), 127)
        signal_power = spectrum[occupied].mean()

        for requested_snr in (0.0, -20.0, -50.0):
            noise = PssSignalGenerator.generate_noise_vector(512, requested_snr, pss)
            noise_power = torch.fft.fft(noise, dim=1).abs().square().mean()
            measured_snr = 10.0 * torch.log10(signal_power / noise_power)
            self.assertAlmostEqual(measured_snr.item(), requested_snr, delta=0.15)

    def test_dataset_has_scalar_four_class_labels(self) -> None:
        """Labels are scalars within 0..3, the noise class occurs, and it is empty."""
        dataset = PssDataset(num_samples=512, snr_db=None)
        self.assertEqual(dataset.labels.ndim, 1)
        self.assertTrue(set(dataset.labels.tolist()).issubset({0, 1, 2, 3}))
        self.assertIn(3, dataset.labels.tolist())
        # Without noise, noise-only windows must be exactly zero -- this catches a
        # frequency shift or phase rotation accidentally injecting signal there.
        self.assertTrue(torch.all(dataset.samples[dataset.labels == 3] == 0))

    def test_cnn_output_shape(self) -> None:
        """The detector emits four logits per window."""
        dataset = PssDataset(num_samples=8, snr_db=-20.0)
        corrected = PssSignalGenerator.correct_frequency_offset(dataset.samples, dataset.freq_offset)
        output = NarrowbandPssCnn()(corrected)
        self.assertEqual(output.shape, (8, 4))

    def test_correlator_is_exact_without_noise(self) -> None:
        """On clean windows the correlator recovers every label exactly."""
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

    def test_pss_correlation_properties(self) -> None:
        """The correlations the detectors rely on: one sharp peak, lost on another subcarrier.

        The autocorrelation peaks at 1 at lag 0 only. Moving PSS 0 by one subcarrier,
        or correlating PSS 1 with PSS 0, never exceeds 0.2 at any lag; half a subcarrier
        already drops the peak below 0.7.
        """
        figure = PssCorrelationFigure()
        data = figure.correlations()
        auto = torch.as_tensor(data["auto"]).abs()
        at_zero = torch.as_tensor(data["lags"] == 0)
        self.assertAlmostEqual(auto[at_zero].item(), 1.0, places=5)
        self.assertLess(auto[~at_zero].max().item(), 0.995)
        self.assertLess(torch.as_tensor(data["shifted"]).abs().max().item(), 0.2)
        self.assertLess(torch.as_tensor(data["cross"]).abs().max().item(), 0.2)
        offsets = figure.offset_peaks()
        half_bin = abs(offsets["offsets"] - 0.5).argmin()
        self.assertLess(offsets["peak"][half_bin], 0.7)
        self.assertAlmostEqual(float(offsets["peak"][0]), 1.0, places=5)

    def test_vectorized_correlator_matches_complex_formula(self) -> None:
        """The real matrix multiply equals the textbook complex inner product.

        Guards the construction in ``Correlator.__init__`` that packs the complex
        correlation into real rows for speed.
        """
        x = torch.randn(10, 2048, dtype=torch.complex64)
        expected = torch.stack([
            (x*PssSignalGenerator.generate_pss(i).conj()).sum(1).abs() /
            (x.norm(dim=1)*PssSignalGenerator.generate_pss(i).norm()) for i in range(3)], dim=1)
        self.assertTrue(torch.allclose(Correlator()(x), expected, atol=1e-6))

    def test_presence_detection_is_not_id_accuracy(self) -> None:
        """Pd, Pfa and accuracy are three different things on one confusion matrix."""
        # Two PSS examples: one wrong ID, one missed. Two noise examples: one FA.
        matrix = torch.tensor([[0, 1, 0, 0], [0, 0, 0, 1],
                               [0, 0, 0, 0], [1, 0, 0, 1]])
        metrics = DetectionMetrics.from_confusion(matrix)
        # The wrong-ID example is still a detection, so Pd is 1/2 while only the
        # correct noise declaration counts toward accuracy, which is 1/4.
        self.assertEqual(metrics['pd'], .5)
        self.assertEqual(metrics['pfa'], .5)
        self.assertEqual(metrics['accuracy'], .25)

    def test_calibration_controls_independent_noise_and_checkpoint_preserves_it(self) -> None:
        """A 1% target really yields ~1% false alarms, at any noise power, after saving.

        Three claims at once: predict() refuses to run uncalibrated; one threshold
        holds across a 50 dB span of noise powers (scale invariance); and the
        threshold survives a state_dict round trip as a registered buffer.
        """
        model = NarrowbandPssCnn().eval()
        with self.assertRaises(RuntimeError):
            model.predict(torch.zeros(1, 2048, dtype=torch.complex64))
        Evaluator.calibrate_thresholds({'narrow_cnn': model}, num_samples=20000)
        for seed, snr in [(310, -50.), (311, -20.), (312, 0.)]:
            torch.manual_seed(seed)
            noise = PssSignalGenerator.generate_noise_vector(10000, snr, PssSignalGenerator.generate_pss(0))
            with torch.no_grad():
                rate = (model.predict(noise) != 3).float().mean().item()
            # Bounds around the 1% target, wide enough for binomial noise at n=10000.
            self.assertGreater(rate, .004)
            self.assertLess(rate, .018)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = NarrowbandPssCnn().eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        with torch.no_grad():
            self.assertTrue(torch.equal(model.predict(noise[:16]), restored.predict(noise[:16])))

    def test_high_snr_and_zero_input(self) -> None:
        """Near-perfect at high SNR, and an all-zero window is finite and rejected.

        The zero window is the degenerate case the ``clamp_min`` guards in
        ``forward`` exist for: without them the normalization divides by zero.
        """
        model = NarrowbandPssCnn().eval()
        model.set_template_weights()  # a trained-quality detector
        Evaluator.calibrate_thresholds({'narrow_cnn': model}, num_samples=5000)
        data = PssDataset(2048, snr_db=0.)
        corrected = PssSignalGenerator.correct_frequency_offset(data.samples, data.freq_offset)
        with torch.no_grad():
            self.assertGreater((model.predict(corrected) == data.labels).float().mean().item(), .985)
            zeros = torch.zeros(2, 2048, dtype=torch.complex64)
            self.assertTrue(torch.isfinite(model(zeros)).all())
            self.assertTrue((model.predict(zeros) == 3).all())

    def test_backward_gradients_are_finite(self) -> None:
        """Every parameter receives a finite gradient, so training cannot silently NaN."""
        model = NarrowbandPssCnn().train()
        data = PssDataset(32, snr_db=-26.)
        x = PssSignalGenerator.correct_frequency_offset(data.samples, data.freq_offset)
        loss = torch.nn.functional.cross_entropy(model(x), data.labels)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
