"""Random-phase checks: the dataset's phase rotation and the correlators' response to it.

The random phase must rotate each signal window and change nothing else, and the
magnitude correlator must be invariant to that rotation while the coherent one is
not.

Run with ``python -m unittest discover -s tests -v``.
"""
import sys
import unittest
from pathlib import Path

import torch

# Make the utils package importable however the tests are launched.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.models.correlator import Correlator  # noqa: E402 (needs the sys.path entry above)
from utils.data.pss_dataset import PssDataset  # noqa: E402 (needs the sys.path entry above)

# Single-threaded, for reproducibility.
torch.set_num_threads(1)


class RandomPhaseTests(unittest.TestCase):
    """Checks on the random-phase dataset and the correlators."""

    def setUp(self) -> None:
        """Fix the global RNG so every test starts from the same state."""
        torch.manual_seed(17)

    def test_dataset_phase_preserves_signal_energy(self) -> None:
        """random_phase rotates each window without changing anything else.

        Same seed, both flags: labels, offsets and per-sample magnitudes must be
        identical, while the complex values must differ. That is exactly what
        "multiply by exp(j*phi)" should do, and nothing more.
        """
        torch.manual_seed(81)
        fixed = PssDataset(128, snr_db=None, random_phase=False)
        torch.manual_seed(81)
        rotated = PssDataset(128, snr_db=None, random_phase=True)
        self.assertTrue(torch.equal(fixed.labels, rotated.labels))
        self.assertTrue(torch.equal(fixed.freq_offset, rotated.freq_offset))
        torch.testing.assert_close(fixed.samples.abs(), rotated.samples.abs())
        self.assertFalse(torch.allclose(fixed.samples[fixed.labels < 3],
                                        rotated.samples[rotated.labels < 3]))

    def test_magnitude_is_phase_invariant_but_coherent_is_not(self) -> None:
        """The two correlator modes differ exactly where the experiment claims.

        Magnitude scores are unchanged by a rotation; coherent scores move
        measurably.
        """
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
