"""Create random-phase figures and tables from measured JSON."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__:
    from .evaluation import metrics_from_confusion
    from .phase_benchmark import parse_args, run
else:
    from evaluation import metrics_from_confusion
    from phase_benchmark import parse_args, run

LABELS = {"phase_cnn": "Phase-invariant complex CNN",
          "legacy_linear": "Legacy linear network (retrained)",
          "correlator_magnitude": "Magnitude correlator",
          "correlator_coherent": "Coherent correlator"}
COLORS = {"phase_cnn": "#009E73", "legacy_linear": "#CC79A7",
          "correlator_magnitude": "#0072B2", "correlator_coherent": "#333333"}


def pooled(report, snr, model):
    matrices = [p["metrics"][model]["confusion"] for r in report["runs"]
                for p in r["points"] if p["snr_db"] == snr]
    return metrics_from_confusion(torch.tensor(matrices).sum(0))


def render(report, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    names = list(report["runs"][0]["cost"])
    plot_names = [name for name in names if name != "legacy_linear"]
    snrs = sorted(p["snr_db"] for p in report["runs"][0]["points"])
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
                         "axes.titlesize": 11, "legend.fontsize": 8,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, key, title in zip(axes.flat[:3], ("accuracy", "pd", "pfa"),
                              ("(a) Four-class accuracy", r"(b) Detection probability $P_d$",
                               r"(c) False-alarm probability $P_{fa}$")):
        for name in plot_names:
            values = [pooled(report, snr, name) for snr in snrs]
            ax.plot(snrs, [100*v[key] for v in values], label=LABELS[name],
                    color=COLORS[name], marker="s" if name == "phase_cnn" else "o",
                    markersize=3, linewidth=2 if name == "phase_cnn" else 1.2,
                    linestyle="-" if name == "phase_cnn" else "--")
            if key == "pfa":
                ci = 100*np.array([v["pfa_ci95"] for v in values])
                ax.fill_between(snrs, ci[:, 0], ci[:, 1], color=COLORS[name], alpha=.07)
        if key == "accuracy": ax.axhline(25, color="#994444", linestyle=":")
        if key == "pfa": ax.axhline(100*report["config"]["target_pfa"], color="#994444", linestyle=":")
        ax.set(title=title, xlabel="Input SNR (dB)", ylabel="Percent (%)")
        ax.set_ylim(0, 103 if key != "pfa" else max(2., ax.get_ylim()[1]))
        ax.grid(alpha=.25, linestyle=":")
    short_labels = {
        "phase_cnn": "Phase-invariant\ncomplex CNN",
        "correlator_magnitude": "Magnitude\ncorrelator",
        "correlator_coherent": "Coherent\ncorrelator",
    }

    costs = [
        report["runs"][0]["cost"][name]["total_real_ops"]
        for name in plot_names
    ]

    bar_labels = [short_labels[name] for name in plot_names]

    bars = axes[1, 1].barh(
        bar_labels[::-1],
        costs[::-1],
        color=[COLORS[name] for name in plot_names][::-1],
        height=.62,
    )

    axes[1, 1].bar_label(
        bars,
        labels=[f"{value:,}" for value in costs[::-1]],
        padding=3,
        fontsize=8,
    )

    axes[1, 1].set(
        title="(d) Approximate detector arithmetic",
        xlabel=(
            "Real arithmetic operations\n"
            "(common exact CFO correction excluded)"
        ),
        xlim=(0, max(costs) * 1.22),
    )
    axes[1, 1].grid(axis="x", alpha=.25, linestyle=":")
    for ax in axes.flat: ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .04),
               ncol=2, frameon=False)
    fig.text(.5, .01, "Aligned 2048 samples, exact CFO, independent uniform phase, AWGN. "
             "Separate 1% false-alarm calibration; frozen test thresholds.", ha="center", fontsize=8)
    fig.subplots_adjust(left=.08, right=.98, top=.95, bottom=.2, hspace=.4, wspace=.28)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output/f"random_phase_results.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    write_report(report, output, names, snrs)


def write_report(report, output, names, snrs):
    lines = ["# Random-phase measured results", "",
             "The original fixed-phase files in `results/` are retained unchanged.", "",
             f"Selected CNN: `{report['config']['selected_architecture']}`. "
             f"Seeds: {report['config']['seeds']}; epochs: {report['config']['epochs']}; "
             f"training samples/SNR: {report['config']['train_per_snr']}.", "",
             "Every signal window has an independent uniform phase. Timing and CFO are still known exactly; "
             "this isolates phase robustness and is not blind acquisition.", "",
             "## Accuracy / Pd / Pfa (%)", "",
             "| SNR | " + " | ".join(LABELS[n] for n in names) + " |",
             "|---|" + "---|"*len(names)]
    for snr in snrs:
        values = [pooled(report, snr, name) for name in names]
        lines.append(f"| {snr:g} | " + " | ".join(
            f"{100*v['accuracy']:.2f} / {100*v['pd']:.2f} / {100*v['pfa']:.2f}"
            for v in values) + " |")
    lines += ["", "## Model and arithmetic cost", "",
              "| Receiver | Real trainable parameters | MACs | Approx. real operations | Stored bytes |",
              "|---|---:|---:|---:|---:|"]
    for name in names:
        cost = report["runs"][0]["cost"][name]
        lines.append(f"| {LABELS[name]} | {cost['parameters']:,} | "
                     f"{cost['conv_linear_macs']:,} | {cost['total_real_ops']:,} | "
                     f"{cost['stored_bytes']:,} |")
    block_size = report["config"]["selected_architecture"]["block_size"]
    cnn_ops = report["runs"][0]["cost"]["phase_cnn"]["total_real_ops"]
    old_ops = report["runs"][0]["cost"]["legacy_linear"]["total_real_ops"]
    mag_ops = report["runs"][0]["cost"]["correlator_magnitude"]["total_real_ops"]
    lines += ["", f"The complex CNN uses a tied complex convolution with kernel=stride={block_size}, one complex "
              "channel, a complex global projection, and a modulus/energy nonlinearity. Its output is "
              "invariant to an arbitrary common phase rotation by construction.", "",
              f"Its arithmetic estimate is {100*(1-cnn_ops/old_ops):.1f}% below the legacy network and "
              f"{100*(1-cnn_ops/mag_ops):.1f}% below the magnitude correlator. At -24 dB its pooled "
              f"Pd is {100*pooled(report, -24.0, 'phase_cnn')['pd']:.2f}% versus "
              f"{100*pooled(report, -24.0, 'correlator_magnitude')['pd']:.2f}% for the magnitude "
              "correlator, at approximately the same 1% Pfa. This is a compute/accuracy trade-off, "
              "not an accuracy win over the classical phase-invariant baseline.", "",
              "The coherent correlator is shown as a deliberately mismatched reference: it does not know "
              "the random phase and is not an oracle. The magnitude correlator is the practical phase-invariant "
              "classical baseline.", ""]
    (output/"RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-results", type=Path)
    known, remaining = parser.parse_known_args()
    if known.from_results:
        report = json.loads(known.from_results.read_text())
        render(report, known.from_results.parent)
    else:
        args = parse_args()
        report = run(args)
        render(report, args.output)


if __name__ == "__main__":
    main()
