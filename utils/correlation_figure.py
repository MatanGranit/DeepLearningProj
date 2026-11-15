"""Figure of the PSS correlation magnitudes the detectors rely on.

Panel (a): three normalized cyclic correlation magnitudes against PSS 0, over the time
lag:

* PSS 0 with itself (autocorrelation): one sharp peak of 1 at lag 0;
* PSS 0 moved to other subcarriers (shifted up by SUBCARRIER_SHIFT FFT bins, a phase
  that rotates along the window), with PSS 0: the subcarriers no longer line up, so
  no lag gives a peak;
* PSS 1 with PSS 0 (cross-correlation): small at every lag, so the identities are told
  apart.

Panel (b): the peak correlation magnitude (best lag) of PSS 0 with a frequency-shifted
copy of itself, against the residual frequency offset. Half a bin costs about 4 dB and
a full bin leaves no more correlation than another PSS, which is why every receiver
works on windows whose frequency offset has been corrected exactly.
"""
from pathlib import Path
from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data.pss_signal import PssSignalGenerator
from .reporting import FIGURE_EXTENSIONS, FIGURE_STYLE

FIGURE_STEM = "pss_correlation"
# FFT bins PSS 0 is moved by for the second curve of panel (a).
SUBCARRIER_SHIFT = 1.0
# Lags shown in panel (a), either side of 0.
SHOWN_LAGS = 128
# Residual frequency offsets of panel (b), in FFT bins.
MAX_OFFSET_BINS = 4.0
OFFSET_STEPS = 161
COLORS = {"auto": "#009E73", "shifted": "#CC79A7", "cross": "#0072B2"}


class PssCorrelationFigure:
    """Compute and draw the PSS auto-, phase-shifted, cross- and frequency-offset correlations."""

    @staticmethod
    def cyclic_correlation(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """
        Correlate a window with a reference at every cyclic time lag.

        Computed through the FFT, which gives exactly the time-domain sum.

        :param x: complex window, shape (N,), unit norm
        :param reference: complex reference window, shape (N,), unit norm
        :return: complex tensor of shape (N,), ``R[tau] = sum_n x[n + tau] conj(reference[n])``
        """
        # torch's ifft carries the 1/N, so this is the correlation sum itself.
        return torch.fft.ifft(torch.fft.fft(x) * torch.fft.fft(reference).conj())

    def correlations(self) -> Dict[str, np.ndarray]:
        """
        Compute the three correlations against PSS 0, centred on lag 0.

        :return: ``lags`` (samples) and the complex curves ``auto``, ``shifted`` (PSS 0
            moved up by SUBCARRIER_SHIFT bins before correlating) and ``cross``
        """
        templates = PssSignalGenerator.pss_templates(normalized=True)
        reference = templates[0]
        n = reference.shape[0]
        moved = reference * PssSignalGenerator.frequency_shift(torch.tensor([SUBCARRIER_SHIFT]), n)[0]
        curves = {"auto": self.cyclic_correlation(reference, reference),
                  "shifted": self.cyclic_correlation(moved, reference),
                  "cross": self.cyclic_correlation(templates[1], reference)}
        result = {name: torch.fft.fftshift(curve).numpy() for name, curve in curves.items()}
        result["lags"] = np.arange(n) - n // 2
        return result

    def offset_peaks(self) -> Dict[str, np.ndarray]:
        """
        Compute the peak correlation magnitude of PSS 0 with frequency-shifted copies of itself.

        :return: ``offsets`` (FFT bins, possibly fractional) and ``peak``, the largest
            ``|R|`` over all lags at each offset
        """
        reference = PssSignalGenerator.pss_templates(normalized=True)[0]
        offsets = torch.linspace(0, MAX_OFFSET_BINS, OFFSET_STEPS)
        # The frequency shift the dataset applies, here by any (also fractional) number of bins.
        shifted = reference * PssSignalGenerator.frequency_shift(offsets, reference.shape[0])
        peaks = torch.stack([self.cyclic_correlation(window, reference).abs().max() for window in shifted])
        return {"offsets": offsets.numpy(), "peak": peaks.numpy()}

    def render(self, output: Union[str, Path]) -> None:
        """
        Draw the lag and frequency-offset panels and save them.

        :param output: directory for ``pss_correlation.{png,pdf,svg}``; created if missing
        """
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        data = self.correlations()
        lags = data["lags"]
        bins = f"{SUBCARRIER_SHIFT:g} subcarrier" + ("" if SUBCARRIER_SHIFT == 1 else "s")
        plt.rcParams.update(FIGURE_STYLE)
        fig, (lag_ax, offset_ax) = plt.subplots(1, 2, figsize=(10, 3.8))

        shown = np.abs(lags) <= SHOWN_LAGS
        lag_ax.plot(lags[shown], np.abs(data["auto"][shown]), color=COLORS["auto"], linewidth=2.2,
                    label="PSS 0 with itself")
        lag_ax.plot(lags[shown], np.abs(data["shifted"][shown]), color=COLORS["shifted"],
                    linewidth=1.4, label=f"PSS 0 moved {bins} up, with PSS 0")
        lag_ax.plot(lags[shown], np.abs(data["cross"][shown]), color=COLORS["cross"], linewidth=1.2,
                    label="PSS 1 with PSS 0")
        lag_ax.set(title="(a) Correlation magnitude over the time lag", xlabel="Lag (samples)",
                   ylabel="|R|, normalized", ylim=(0, 1.05))
        lag_ax.legend(loc="upper left", frameon=False)

        offsets = self.offset_peaks()
        offset_ax.plot(offsets["offsets"], offsets["peak"], color=COLORS["auto"], linewidth=2.2,
                       label="PSS 0 shifted in frequency, with PSS 0")
        offset_ax.axhline(np.abs(data["cross"]).max(), color=COLORS["cross"], linestyle=":", linewidth=1.2,
                          label="Largest |R| of PSS 1 with PSS 0")
        offset_ax.set(title="(b) Peak correlation magnitude vs frequency offset",
                      xlabel="Residual frequency offset (FFT bins)", ylabel="Peak |R| over all lags, normalized",
                      ylim=(0, 1.05))
        offset_ax.legend(loc="upper right", frameon=False)

        for ax in (lag_ax, offset_ax):
            ax.grid(alpha=.25, linestyle=":")
            ax.spines[["top", "right"]].set_visible(False)
        fig.text(.5, -.02, "Cyclic time-domain correlation over the 2048-sample window. Moving the PSS by "
                 "one subcarrier destroys the peak; half a subcarrier already costs about 4 dB.",
                 ha="center", fontsize=8)
        fig.subplots_adjust(left=.07, right=.98, top=.9, bottom=.15, wspace=.25)
        for extension in FIGURE_EXTENSIONS:
            fig.savefig(output / f"{FIGURE_STEM}.{extension}", dpi=200, bbox_inches="tight")
        plt.close(fig)
