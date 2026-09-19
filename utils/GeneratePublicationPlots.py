"""Render measured benchmark results; train once only if no results are present."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__:
    from .benchmark import ROOT, parse_args, run
    from .evaluation import metrics_from_confusion
else:
    from benchmark import ROOT, parse_args, run
    from evaluation import metrics_from_confusion

LABELS = {"compact": "Compact neural detector",
          "correlator_magnitude": "Magnitude correlator",
          "correlator_coherent": "Coherent correlator"}
COLORS = {"compact": "#009E73",
          "correlator_magnitude": "#0072B2", "correlator_coherent": "#333333"}


def pooled_metrics(report, snr, model):
    matrices = [point["metrics"][model]["confusion"] for run_data in report["runs"]
                for point in run_data["points"] if point["snr_db"] == snr]
    return metrics_from_confusion(torch.tensor(matrices).sum(dim=0))


def render_results(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = list(report["runs"][0]["cost"])
    snrs = sorted(p["snr_db"] for p in report["runs"][0]["points"])
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
                         "axes.titlesize": 11, "legend.fontsize": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    metric_keys = ["accuracy", "pd", "pfa"]
    titles = ["(a) Four-class accuracy", r"(b) Detection probability $P_d$",
              r"(c) False-alarm probability $P_{fa}$"]
    for ax, key, title in zip(axes.flat, metric_keys, titles):
        for name in names:
            points = [pooled_metrics(report, snr, name) for snr in snrs]
            values = 100*np.array([p[key] for p in points])
            ax.plot(snrs, values, label=LABELS[name], color=COLORS[name],
                    marker="s" if name == "compact" else "o", markersize=3,
                    linestyle="-" if name == "compact" else "--",
                    linewidth=2 if name == "compact" else 1.2)
            if key == "pfa":
                intervals = 100*np.array([p["pfa_ci95"] for p in points])
                ax.fill_between(snrs, intervals[:, 0], intervals[:, 1],
                                color=COLORS[name], alpha=.08)
        if key == "accuracy":
            ax.axhline(25, linestyle=":", color="#994444", linewidth=1)
        if key == "pfa":
            ax.axhline(100*report["config"]["target_pfa"], color="#994444", linestyle=":")
        ax.set(title=title, xlabel="Input SNR (dB)" if key != "pfa" else "Reference SNR / noise level (dB)",
               ylabel="Percent (%)")
        if key != "pfa":
            ax.set_ylim(0, 103)
        else:
            ax.set_ylim(bottom=0)
        ax.grid(alpha=.25, linestyle=":")
    latency = [np.median([r["latency"][name]["64"]["core_us_per_window"] for r in report["runs"]])
               for name in names]
    short = {"compact": "Compact detector",
             "correlator_magnitude": "Magnitude\ncorr.", "correlator_coherent": "Coherent\ncorr."}
    bars = axes[1, 1].bar([short[n] for n in names], latency, color=[COLORS[n] for n in names])
    axes[1, 1].bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    axes[1, 1].set(title="(d) Measured inference time", ylabel="Microseconds per window",
                   ylim=(0, max(latency)*1.18))
    env = report["environment"]
    axes[1, 1].set_xlabel(f"{env['device'].upper()}, {env['threads']} thread(s), batch 64; correction excluded")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .047), ncol=3, frameon=False)
    fig.text(.5, .012, "Aligned 2048 samples, exact frequency offset, fixed phase, AWGN. "
             "Each detector calibrated on separate noise; test thresholds frozen.", ha="center", fontsize=8)
    fig.subplots_adjust(left=.08, right=.98, top=.95, bottom=.18, hspace=.37, wspace=.28)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"academic_results.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # A separate arithmetic plot avoids presenting theoretical operations as speed.
    fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
    costs = [report["runs"][0]["cost"][name]["total_real_ops"] for name in names]
    bars = ax.bar([short[n] for n in names], costs, color=[COLORS[n] for n in names])
    ax.bar_label(bars, labels=[f"{c:,}" for c in costs], padding=3)
    ax.set(ylabel="Approximate real arithmetic operations", title="Detector arithmetic, including normalization",
           ylim=(0, max(costs)*1.18))
    ax.set_xlabel("Common frequency correction excluded from every bar")
    fig.savefig(output / "arithmetic_costs.png", dpi=200)
    plt.close(fig)
    write_summary(report, output, names, snrs)


def write_summary(report, output, names, snrs):
    lines = ["# Measured validation results", "",
             "Generated from the measured data in benchmark_results.json.", "",
             f"Training seeds: {report['config']['seeds']}. Epochs: {report['config']['epochs']}. "
             f"Training samples/SNR: {report['config']['train_per_snr']}.", "",
             f"Calibration noise windows per SNR/seed: {report['config']['calibration_samples']}; "
             f"test windows per SNR/seed: {report['config']['test_samples']}.", "",
             f"Every model is calibrated to the same {100*report['config']['target_pfa']:g}% target; measured test Pfa is reported below. "
             "A target is not an exact test guarantee.", "",
             "## Accuracy / Pd / Pfa (%)", "",
             "| SNR | " + " | ".join(LABELS[n] for n in names) + " |",
             "|---|" + "---|"*len(names)]
    for snr in snrs:
        ms = [pooled_metrics(report, snr, name) for name in names]
        lines.append(f"| {snr:g} | " + " | ".join(
            f"{100*m['accuracy']:.2f} / {100*m['pd']:.2f} / {100*m['pfa']:.2f}" for m in ms)+" |")
    lines += ["", "## Cost and measured latency", "",
              "Times are medians across runs, in microseconds per window. CPU/GPU data transfer "
              "is excluded; +CFO includes the common frequency correction. These are local measurements.", "",
              "| Receiver | Parameters | Conv/Linear MACs | Real ops incl. normalization | Stored bytes | Batch 1 core | Batch 1 +CFO | Batch 64 core | Batch 64 +CFO |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        c = report['runs'][0]['cost'][name]
        times = [np.median([r['latency'][name][b][key] for r in report['runs']])
                 for b in ('1', '64') for key in ('core_us_per_window', 'with_correction_us_per_window')]
        lines.append(f"| {LABELS[name]} | {c['parameters']:,} | {c['conv_linear_macs']:,} | "
                     f"{c['total_real_ops']:,} | {c['stored_bytes']:,} | " + " | ".join(f"{t:.2f}" for t in times)+" |")
    lines += ["", "The neural detector advantage over the magnitude correlator exploits the fixed-phase model. "
              "The coherent correlator is the stronger, lower-cost classical reference in this model; "
              "this experiment does not demonstrate a fundamental neural advantage over it.", "",
              "Shared correction is roughly 8N real arithmetic operations plus N complex exponential "
              "evaluations for N=2048. It is excluded from all detector operation counts and explicitly "
              "included in the +CFO latency columns. Operation estimates exclude comparisons and memory movement.", "",
              "Pfa bands in the plot are pooled 95% Wilson binomial intervals; they do not account "
              "for all training/calibration variability. Per-seed counts, thresholds, training times, "
              "raw-argmax diagnostics and intervals are retained in JSON.", "",
              "Environment: " + json.dumps(report['environment']), ""]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    if "--from-results" in sys.argv:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--from-results", type=Path, required=True)
        options = parser.parse_args()
        render_results(json.loads(options.from_results.read_text()), options.from_results.parent)
    else:
        args = parse_args()
        existing = args.output / "benchmark_results.json"
        if len(sys.argv) == 1 and existing.exists():
            report = json.loads(existing.read_text())
        else:
            report = run(args)
        render_results(report, args.output)


if __name__ == "__main__":
    main()
