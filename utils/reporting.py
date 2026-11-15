"""Reporting only: figures and RESULTS.md tables rendered from a results JSON.

A reporter reads measurements and draws them; it never trains, tests or loads a model.

Nothing here re-derives a metric from raw scores: rates come from the recorded
confusion matrices (via :meth:`~utils.metrics.DetectionMetrics.from_confusion`) and
costs from the recorded operation counts, so the figures cannot drift from the
measurements.

* :class:`KnownBinReporter` writes ``known_bin_128_results.{png,pdf,svg}`` (accuracy,
  Pd, Pfa, arithmetic cost) and ``RESULTS.md``.
* :class:`UnknownBinReporter` writes ``unknown_bin_results.{png,pdf,svg}`` and
  ``RESULTS.md``.
"""
import json
from pathlib import Path
from typing import Dict, Iterable, List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data.pss_signal import NUM_CLASSES
from .metrics import DetectionMetrics

# Type 42 fonts embed as text, so the vector output stays selectable.
FIGURE_STYLE = {"font.family": "DejaVu Serif", "font.size": 9,
                "axes.titlesize": 11, "legend.fontsize": 8, "pdf.fonttype": 42}
REFERENCE_LINE_COLOR = "#994444"
RATE_PANELS = (("accuracy", "(a) Four-class accuracy"),
               ("pd", r"(b) Detection probability $P_d$"),
               ("pfa", r"(c) False-alarm probability $P_{fa}$"))
FIGURE_EXTENSIONS = ("png", "pdf", "svg")
# The classical reference every report compares the neural detector with. The results
# JSON also holds the other receivers an experiment ran; the reports leave them out.
MAGNITUDE_CORRELATOR = "correlator_magnitude"
# Colour of the frequency-offset (bin found) curves, apart from the receivers' colours.
OFFSET_COLOR = "#D55E00"


class Reporter:
    """Render one experiment's report: a 2x2 figure and a Markdown summary.

    Every report shows two receivers: the experiment's neural detector (``SUBJECT``)
    and the magnitude correlator. Panels (a)-(c) plot accuracy, Pd and Pfa against
    SNR, panel (d) their arithmetic; the figure layout and the summary text are set
    by the subclass.
    """

    LABELS: Dict[str, str]           # receiver key -> legend and table name (see label())
    SHORT_LABELS: Dict[str, str]     # receiver key -> bar-chart tick label
    COLORS: Dict[str, str]           # colour-blind-safe palette, by receiver key
    SUBJECT: str                     # the neural detector's results key, drawn solid and heavier
    FIGURE_STEM: str
    DPI: int
    PFA_BAND_ALPHA: float
    CHANCE_LINE_STYLE: dict
    LEGEND_Y: float
    LEGEND_COLUMNS: int
    CAPTION: str
    CAPTION_Y: float
    LAYOUT: dict
    COST_AXIS_LABEL = "Real arithmetic operations\n(common exact CFO correction excluded)"
    # The summary quotes Pd at the swept SNR where the magnitude correlator's Pd is
    # closest to this value: the middle of the detection knee, in any SNR sweep.
    COMPARISON_PD = 0.5

    def __init__(self, report: dict) -> None:
        """
        Bind the renderer to a report and pick the receivers it shows and the SNR grid.

        :param report: the report dict an experiment wrote
        """
        self.report = report
        self.names = [name for name in report["runs"][0]["cost"] if name in (self.SUBJECT, MAGNITUDE_CORRELATOR)]
        self.snrs = sorted(p["snr_db"] for p in report["runs"][0]["points"])

    @classmethod
    def from_file(cls, path: Path) -> "Reporter":
        """
        Load a report from its JSON file.

        :param path: a results JSON written by an experiment
        :return: the renderer for it
        """
        return cls(json.loads(path.read_text()))

    def pooled_metrics(self, snr: float, name: str) -> Dict:
        """
        Pool one receiver's confusion matrices across every seed at one SNR.

        Summing the matrices before deriving rates -- rather than averaging per-seed
        rates -- keeps the counts intact, so the Wilson intervals reflect the full
        pooled sample size.

        :param snr: the SNR in dB to pool at
        :param name: receiver key, as used in the report
        :return: metrics dict from :meth:`~utils.metrics.DetectionMetrics.from_confusion`
        """
        matrices = [point["metrics"][name]["confusion"] for run_data in self.report["runs"]
                    for point in run_data["points"] if point["snr_db"] == snr]
        return DetectionMetrics.from_confusion(torch.tensor(matrices).sum(dim=0))

    def snr_axis_label(self) -> str:
        """
        Name the SNR axis by the definition the results record.

        :return: e.g. "SNR per PSS subcarrier (dB)"; results without a recorded
            definition get "Input SNR (dB)"
        """
        definition = self.report.get("snr_definition")
        return f"SNR {definition} (dB)" if definition else "Input SNR (dB)"

    def label(self, name: str) -> str:
        """
        Name a receiver in legends and tables, with its network class when the results record it.

        :param name: receiver key
        :return: e.g. "Neural detector, 128 samples (NarrowbandPssCnn)"
        """
        network = self.report.get("models", {}).get(name)
        return f"{self.LABELS[name]} ({network})" if network else self.LABELS[name]

    def cost(self, name: str) -> Dict[str, int]:
        """
        Return one receiver's arithmetic cost; costs are identical across seeds.

        :param name: receiver key
        :return: the cost dict of the first run
        """
        return self.report["runs"][0]["cost"][name]

    def render(self, output: Union[str, Path]) -> None:
        """
        Write every figure and the Markdown summary.

        :param output: directory to write into; created if missing
        """
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self.render_main_figure(output)
        self.write_summary(output)

    def render_main_figure(self, output: Path) -> None:
        """
        Draw the 2x2 figure: three rate panels plus the arithmetic-cost panel (d).

        :param output: directory to write ``<FIGURE_STEM>.{png,pdf,svg}`` into
        """
        plt.rcParams.update(FIGURE_STYLE)
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        self.plot_rate_panels(axes.flat[:3])
        self.plot_panel_d(axes[1, 1])
        for ax in axes.flat:
            ax.spines[["top", "right"]].set_visible(False)
        # One shared legend: every panel shows the same receivers.
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, self.LEGEND_Y),
                   ncol=self.LEGEND_COLUMNS, frameon=False)
        fig.text(.5, self.CAPTION_Y, self.CAPTION, ha="center", fontsize=8)
        fig.subplots_adjust(**self.LAYOUT)
        for extension in FIGURE_EXTENSIONS:
            fig.savefig(output / f"{self.FIGURE_STEM}.{extension}", dpi=self.DPI, bbox_inches="tight")
        plt.close(fig)

    def plot_rate_panels(self, axes: Iterable[plt.Axes]) -> None:
        """
        Plot accuracy, Pd and Pfa against SNR for every plotted receiver.

        :param axes: the three axes for panels (a), (b) and (c)
        """
        for ax, (key, title) in zip(axes, RATE_PANELS):
            for name in self.names:
                points = [self.pooled_metrics(snr, name) for snr in self.snrs]
                subject = name == self.SUBJECT
                ax.plot(self.snrs, 100*np.array([p[key] for p in points]), label=self.label(name),
                        color=self.COLORS[name], marker="s" if subject else "o", markersize=3,
                        linestyle="-" if subject else "--", linewidth=2 if subject else 1.2)
                if key == "pfa":
                    # Pfa sits near 1%, where binomial noise is visible; show it.
                    intervals = 100*np.array([p["pfa_ci95"] for p in points])
                    ax.fill_between(self.snrs, intervals[:, 0], intervals[:, 1],
                                    color=self.COLORS[name], alpha=self.PFA_BAND_ALPHA)
            if key == "accuracy":
                ax.axhline(100 / NUM_CLASSES, **self.CHANCE_LINE_STYLE)  # uniform guessing
            if key == "pfa":
                ax.axhline(100*self.report["config"]["target_pfa"], color=REFERENCE_LINE_COLOR, linestyle=":")
            ax.set(title=title, xlabel=self.snr_axis_label(),
                   ylabel="Percent (%)")
            if key == "pfa":
                self.set_pfa_ylim(ax)
            else:
                ax.set_ylim(0, 103)
            ax.grid(alpha=.25, linestyle=":")

    def rate_table_lines(self) -> List[str]:
        """
        Build the Markdown table of pooled accuracy / Pd / Pfa per SNR, for every receiver.

        :return: lines of Markdown, starting with the section heading
        """
        lines = ["## Accuracy / Pd / Pfa (%)", "",
                 f"| {self.snr_axis_label()} | " + " | ".join(self.label(n) for n in self.names) + " |",
                 "|---|" + "---|"*len(self.names)]
        for snr in self.snrs:
            metrics = [self.pooled_metrics(snr, name) for name in self.names]
            lines.append(f"| {snr:g} | " + " | ".join(
                f"{100*m['accuracy']:.2f} / {100*m['pd']:.2f} / {100*m['pfa']:.2f}" for m in metrics) + " |")
        return lines

    def set_pfa_ylim(self, ax: plt.Axes) -> None:
        """
        Set the vertical range of the Pfa panel.

        :param ax: the Pfa axes
        :raises NotImplementedError: in the base class
        """
        raise NotImplementedError

    def plot_panel_d(self, ax: plt.Axes) -> None:
        """
        Plot arithmetic cost as horizontal bars in panel (d).

        :param ax: the bottom-right axes
        """
        names = self.names
        costs = [self.cost(name)["total_real_ops"] for name in names]
        # Horizontal so the multi-line receiver names stay readable, and reversed so
        # the first receiver ends up on top rather than at the bottom.
        bars = ax.barh([self.SHORT_LABELS[name] for name in names][::-1], costs[::-1],
                       color=[self.COLORS[name] for name in names][::-1], height=.62)
        ax.bar_label(bars, labels=[f"{value:,}" for value in costs[::-1]], padding=3, fontsize=8)
        ax.set(title="(d) Approximate detector arithmetic",
               xlabel=self.COST_AXIS_LABEL,
               # Headroom so the value labels do not run off the axis.
               xlim=(0, max(costs) * 1.22))
        ax.grid(axis="x", alpha=.25, linestyle=":")

    def comparison_lines(self) -> List[str]:
        """
        Compare the neural detector with the magnitude correlator in arithmetic and in Pd.

        Pd is quoted at the swept SNR where the magnitude correlator's Pd is closest to
        COMPARISON_PD.

        :return: lines of Markdown, starting with a blank line
        """
        detector_ops = self.cost(self.SUBJECT)["total_real_ops"]
        magnitude_ops = self.cost(MAGNITUDE_CORRELATOR)["total_real_ops"]
        snr = min(self.snrs, key=lambda s: abs(self.pooled_metrics(s, MAGNITUDE_CORRELATOR)["pd"]
                                               - self.COMPARISON_PD))
        target = 100 * self.report["config"]["target_pfa"]
        return ["", f"The neural detector needs {detector_ops:,} real operations per window, "
                f"{100*(1-detector_ops/magnitude_ops):.1f}% fewer than the magnitude correlator's "
                f"{magnitude_ops:,}. At {snr:g} dB, the swept SNR where the magnitude correlator's Pd is "
                f"closest to {100*self.COMPARISON_PD:g}%, the neural detector's pooled Pd is "
                f"{100*self.pooled_metrics(snr, self.SUBJECT)['pd']:.2f}% versus "
                f"{100*self.pooled_metrics(snr, MAGNITUDE_CORRELATOR)['pd']:.2f}% for the magnitude correlator, "
                f"both calibrated to the same {target:g}% Pfa target.", ""]

    def write_summary(self, output: Path) -> None:
        """
        Write ``RESULTS.md``.

        :param output: directory to write into
        :raises NotImplementedError: in the base class
        """
        raise NotImplementedError


class KnownBinReporter(Reporter):
    """Figures and tables for :class:`~utils.experiments.KnownBinExperiment`."""

    LABELS = {"narrow_cnn": "Neural detector, 128 samples", MAGNITUDE_CORRELATOR: "Magnitude correlator, 128 samples"}
    SHORT_LABELS = {"narrow_cnn": "Neural detector\n(128 samples)",
                    MAGNITUDE_CORRELATOR: "Magnitude correlator\n(128 samples)"}
    COLORS = {"narrow_cnn": "#009E73", MAGNITUDE_CORRELATOR: "#0072B2"}
    SUBJECT = "narrow_cnn"
    FIGURE_STEM = "known_bin_128_results"
    COST_AXIS_LABEL = "Real arithmetic operations\n(common CFO correction and 128-bin band selection excluded)"
    DPI = 220
    PFA_BAND_ALPHA = .07
    CHANCE_LINE_STYLE = {"color": REFERENCE_LINE_COLOR, "linestyle": ":"}
    LEGEND_Y = .04
    LEGEND_COLUMNS = 2
    CAPTION = ("Exact CFO, independent uniform phase, AWGN; both receivers on the 128 bins that hold the PSS. "
               "Separate 1% false-alarm calibration; frozen test thresholds.")
    SUMMARY_TITLE = "Known-bin, 128-sample measured results"
    SETUP_NOTE = ("Every signal window has an independent uniform phase; timing and CFO are known exactly. Both "
                  "receivers keep the 128 FFT bins that hold the PSS (an ideal band-pass and decimation by 16) "
                  "and score those 128 samples.")
    CAPTION_Y = .01
    LAYOUT = {"left": .08, "right": .98, "top": .95, "bottom": .2, "hspace": .4, "wspace": .28}

    def set_pfa_ylim(self, ax: plt.Axes) -> None:
        """
        Give the Pfa axis, which hugs 1%, at least a 2% ceiling.

        :param ax: the Pfa axes
        """
        ax.set_ylim(0, max(2., ax.get_ylim()[1]))

    def write_summary(self, output: Path) -> None:
        """
        Write ``RESULTS.md``: pooled tables plus the comparison with the magnitude correlator.

        The closing paragraph compares the neural detector with the magnitude
        correlator in arithmetic and in Pd at the middle of the detection knee.

        :param output: directory to write into
        """
        config = self.report["config"]
        # Older results also record the block_size and rank their network was built with.
        selected = (f"Selected CNN: `{config['selected_architecture']}`. "
                    if "selected_architecture" in config else "")
        lines = [f"# {self.SUMMARY_TITLE}", "",
                 f"{selected}Seeds: {config['seeds']}; epochs: {config['epochs']}; "
                 f"training samples/SNR: {config['train_per_snr']}.", "",
                 self.SETUP_NOTE, ""]
        lines += self.rate_table_lines()
        lines += ["", "## Model and arithmetic cost", "",
                  "| Receiver | Parameters | MACs | Approx. real operations | Stored bytes |",
                  "|---|---:|---:|---:|---:|"]
        for name in self.names:
            cost = self.cost(name)
            lines.append(f"| {self.label(name)} | {cost['parameters']:,} | "
                         f"{cost['conv_linear_macs']:,} | {cost['total_real_ops']:,} | "
                         f"{cost['stored_bytes']:,} |")
        lines += self.comparison_lines()
        (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


class UnknownBinReporter(KnownBinReporter):
    """Figures and tables for :class:`~utils.experiments.UnknownBinExperiment`."""

    LABELS = {"scan_cnn": "Neural detector, offset unknown", MAGNITUDE_CORRELATOR: "Magnitude correlator, offset known"}
    SHORT_LABELS = {"scan_cnn": "Neural detector\n(offset unknown)",
                    MAGNITUDE_CORRELATOR: "Magnitude correlator\n(offset known)"}
    COLORS = {"scan_cnn": "#009E73", MAGNITUDE_CORRELATOR: "#0072B2"}
    SUBJECT = "scan_cnn"
    FIGURE_STEM = "unknown_bin_results"
    COST_AXIS_LABEL = "Real arithmetic operations\n(the known-offset correlator's exact CFO correction excluded)"
    CAPTION = ("2048 samples, independent uniform phase, AWGN. The network gets the window with its frequency offset; "
               "the offset-known correlator gets it corrected. Separate 1% false-alarm calibration.")
    SUMMARY_TITLE = "Unknown-bin measured results"
    SETUP_NOTE = ("Every signal window has an independent uniform phase and a random frequency offset. The neural "
                  "detector gets the window as received and searches every offset; the magnitude correlator gets "
                  "it with the exact offset removed.")
    OFFSET_FIGURE_STEM = "offset_accuracy"

    def pooled_offset(self, snr: float) -> Dict:
        """
        Pool the neural detector's offset counts across every seed at one SNR.

        :param snr: the SNR in dB
        :return: dict from :meth:`~utils.metrics.DetectionMetrics.offset_rates`
        """
        counts = [0, 0, 0]
        for run_data in self.report["runs"]:
            for point in run_data["points"]:
                if point["snr_db"] == snr:
                    offset = point["metrics"][self.SUBJECT]["offset"]
                    counts = [counts[0] + offset["signals"], counts[1] + offset["detected"],
                              counts[2] + offset["right_offset"]]
        return DetectionMetrics.offset_rates(*counts)

    def has_offsets(self) -> bool:
        """Return whether the results record the frequency offsets the network found."""
        return "offset" in self.report["runs"][0]["points"][0]["metrics"][self.SUBJECT]

    def plot_rate_panels(self, axes: Iterable[plt.Axes]) -> None:
        """
        Plot the rate panels; when the results hold offsets, panel (b) also shows how often the bin was found.

        :param axes: the three axes for panels (a), (b) and (c)
        """
        axes = list(axes)
        super().plot_rate_panels(axes)
        if not self.has_offsets():
            return
        pd_axes = axes[[key for key, _ in RATE_PANELS].index("pd")]
        rates = [self.pooled_offset(snr) for snr in self.snrs]
        found = pd_axes.plot(self.snrs, [100 * r["correct_given_detection"] for r in rates], color=OFFSET_COLOR,
                             lw=1.6, marker="o", markersize=2, label="Right bin, of the network's detections")
        found += pd_axes.plot(self.snrs, [100 * r["detected_with_offset"] for r in rates], color=OFFSET_COLOR,
                              lw=1.2, linestyle=":", label="Network: detected at the right bin")
        pd_axes.set_title("(b) Detection probability $P_d$ and bin found")
        # Panel-local legend for the bin curves only; the shared figure legend names the receivers.
        pd_axes.legend(handles=found, loc="center right", bbox_to_anchor=(1, .45), fontsize=7, frameon=False)

    def render(self, output: Union[str, Path]) -> None:
        """
        Write every figure, including the offset-accuracy figure when the results hold offsets, and the summary.

        :param output: directory to write into; created if missing
        """
        super().render(output)
        if self.has_offsets():
            self.render_offset_figure(Path(output))

    def render_offset_figure(self, output: Path) -> None:
        """
        Plot, per SNR, Pd and how often the reported frequency offset is the true one.

        :param output: directory to write ``<OFFSET_FIGURE_STEM>.{png,pdf,svg}`` into
        """
        plt.rcParams.update(FIGURE_STYLE)
        fig, ax = plt.subplots(figsize=(8, 4.6))
        rates = [self.pooled_offset(snr) for snr in self.snrs]
        detected = [self.pooled_metrics(snr, self.SUBJECT)["pd"] for snr in self.snrs]
        ax.plot(self.snrs, [100 * r["correct_given_detection"] for r in rates], "o-", color=OFFSET_COLOR, lw=2,
                markersize=3, label="Right offset, of the detected windows")
        ax.plot(self.snrs, [100 * p for p in detected], "s-", color=self.COLORS[self.SUBJECT], lw=2, markersize=3,
                label="Pd")
        ax.plot(self.snrs, [100 * r["detected_with_offset"] for r in rates], "--", color="#0072B2", lw=1.4,
                label="Detected and right offset")
        ax.set(xlabel=self.snr_axis_label(), ylabel="Percent (%)", ylim=(0, 103),
               title="Frequency offset found by the max-pool's argmax")
        ax.grid(alpha=.25, linestyle=":")
        ax.legend(loc="center right", fontsize=9)
        for extension in FIGURE_EXTENSIONS:
            fig.savefig(output / f"{self.OFFSET_FIGURE_STEM}.{extension}", dpi=self.DPI, bbox_inches="tight")
        plt.close(fig)

    def comparison_lines(self) -> List[str]:
        """
        Compare with the magnitude correlator, then tabulate how often the reported offset is right.

        :return: lines of Markdown, starting with a blank line
        """
        lines = super().comparison_lines()
        if not self.has_offsets():
            return lines
        lines += ["## Frequency offset found", "",
                  "The network also reports where it found the PSS: the argmax of its max-pool over the "
                  "searched offsets.", "",
                  f"| {self.snr_axis_label()} | Pd (%) | Right offset, of detected (%) "
                  "| Detected and right offset (%) |",
                  "|---|---:|---:|---:|"]
        for snr in self.snrs:
            rates = self.pooled_offset(snr)
            lines.append(f"| {snr:g} | {100 * self.pooled_metrics(snr, self.SUBJECT)['pd']:.2f} | "
                         f"{100 * rates['correct_given_detection']:.2f} | {100 * rates['detected_with_offset']:.2f} |")
        return lines + [""]
