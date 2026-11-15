"""128-sample checks: the narrowband view keeps the PSS, and NarrowbandPssCnn equals the 128-sample correlator.

The PSS fills 127 of the 128 bins kept, so the view must lose none of it; its real
spectrum makes the templates conjugate symmetric, which the network's weight tying
relies on; and with its initial weights the network's score must be the squared
correlation of the 128-sample magnitude correlator, at fewer operations than the
direct correlation.

Run with ``python -m unittest discover -s tests -v``.
"""
import math
import sys
import unittest
from pathlib import Path

import torch

# Make the utils package importable however the tests are launched.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.data.pss_dataset import PssDataset  # noqa: E402 (needs the sys.path entry above)
from utils.data.pss_signal import NARROWBAND_SIZE, NOISE_LABEL, PssSignalGenerator  # noqa: E402 (sys.path above)
from utils.evaluation import Evaluator  # noqa: E402 (needs the sys.path entry above)
from utils.models.correlator import NarrowbandCorrelator  # noqa: E402 (needs the sys.path entry above)
from utils.models.networks import NarrowbandPssCnn  # noqa: E402 (needs the sys.path entry above)

# Single-threaded, for reproducibility.
torch.set_num_threads(1)


class NarrowbandTests(unittest.TestCase):
    """Checks on the 128-sample view, NarrowbandPssCnn and NarrowbandCorrelator."""

    def setUp(self) -> None:
        """Fix the global RNG so every test starts from the same state."""
        torch.manual_seed(31)

    def test_view_keeps_the_pss_and_is_conjugate_symmetric(self) -> None:
        """The 128 bins hold all the PSS energy, and each 128-sample template satisfies t[128-n] = conj(t[n])."""
        templates = PssSignalGenerator.pss_templates().to(torch.complex128)
        view = PssSignalGenerator.narrowband(templates)
        # Parseval: the 2048-point FFT energy of the kept bins is 128 times the view's energy.
        kept = NARROWBAND_SIZE * view.abs().square().sum(1) / (templates.abs().square().sum(1) * 2048)
        torch.testing.assert_close(kept, torch.ones(3, dtype=torch.float64), rtol=1e-6, atol=0.0)
        n = torch.arange(1, NARROWBAND_SIZE)
        # The templates are built in float32, so the symmetry holds to float32 precision.
        torch.testing.assert_close(view[:, NARROWBAND_SIZE - n], view[:, n].conj(), rtol=1e-5, atol=1e-7)

    def test_network_score_is_the_squared_correlation(self) -> None:
        """With the template weights, score * L1^2 / gain equals |<t, y>|^2 on the 128-sample view."""
        model, correlator = NarrowbandPssCnn().eval(), NarrowbandCorrelator()
        model.set_template_weights()
        x = torch.randn(8, 2048, dtype=torch.complex64)
        y = PssSignalGenerator.narrowband(x)
        with torch.no_grad():
            scores = model(x)[:, :3] * model.l1_norm(y).square() / model.gain
            correlation = correlator(x) * y.norm(dim=1, keepdim=True)
        torch.testing.assert_close(scores, correlation.square(), rtol=1e-4, atol=1e-8)

    def test_clean_windows_are_identified_at_any_phase(self) -> None:
        """With the template weights, the network names every PSS identity of noise-free windows."""
        data = PssDataset(256, snr_db=None, random_phase=True)
        corrected = PssSignalGenerator.correct_frequency_offset(data.samples, data.freq_offset)
        with torch.no_grad():
            model = NarrowbandPssCnn().eval()
            model.set_template_weights()
            identity = model(corrected)[:, :3].argmax(dim=1)
        signal = data.labels != NOISE_LABEL
        self.assertTrue(torch.equal(identity[signal], data.labels[signal]))

    def test_cheaper_than_the_128_sample_correlator(self) -> None:
        """The network needs fewer operations and stores fewer bytes than the direct 128-sample correlator."""
        network = Evaluator.arithmetic_cost(NarrowbandPssCnn())
        correlator = Evaluator.arithmetic_cost(NarrowbandCorrelator())
        self.assertEqual(network["total_real_ops"], 2058)
        self.assertEqual(correlator["total_real_ops"], 3599)
        self.assertLess(network["stored_bytes"], correlator["stored_bytes"])

    def test_weights_start_random(self) -> None:
        """The tied weights start random, at unit norm and conjugate symmetric, far from the templates."""
        model, template = NarrowbandPssCnn(), NarrowbandPssCnn()
        template.set_template_weights()
        with torch.no_grad():
            weights, templates = model.weights(), template.weights()
        torch.testing.assert_close(weights.norm(dim=1), torch.ones(3), rtol=1e-5, atol=1e-6)
        n = torch.arange(1, NARROWBAND_SIZE)
        torch.testing.assert_close(weights[:, NARROWBAND_SIZE - n], weights[:, n].conj())
        self.assertLess((templates.conj() * weights).sum(1).abs().max().item(), 0.5)

    def test_scores_are_scale_invariant(self) -> None:
        """Scaling the input does not move the scores, so one threshold serves every noise power."""
        model = NarrowbandPssCnn().eval()
        x = torch.randn(16, 2048, dtype=torch.complex64)
        with torch.no_grad():
            torch.testing.assert_close(model(x), model(1e3 * x), rtol=1e-5, atol=1e-7)
        self.assertTrue(math.isfinite(float(model.gain)))


if __name__ == "__main__":
    unittest.main()
