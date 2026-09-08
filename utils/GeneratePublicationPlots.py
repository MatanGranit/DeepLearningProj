"""
Generates publication figures with four-class accuracy, probability of
detection (Pd), probability of false alarm (Pfa), and computational complexity
for both the neural network and the classic correlator.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter, MultipleLocator, MaxNLocator
from thop import profile
from torch.utils.data import ConcatDataset, DataLoader

from net import EfficientPssCnn, train_model, correct_frequency_offset
from data_loader import PssDataset, generate_noise_vector, generate_pss
from correlator import Correlator

SEED = 0
FFT_BINSIZE = 2048
NUM_CLASSES = 4
NUM_PSS_SEQUENCES = 3
TARGET_CORRELATOR_PFA = 0.01
CALIBRATION_SAMPLES = 10_000
CALIBRATION_SNR_DB = -30.0

TRAIN_SAMPLES_PER_SNR = 3000
TEST_SAMPLES_PER_SNR = 2000
TRAIN_EPOCHS = 8
BATCH_SIZE = 64

SNR_RANGE = np.array([0.0, -10.0, -20.0, -22.0, -24.0, -26.0, -28.0, -30.0, -32.0, -34.0, -36.0, -38.0, -40.0, -45.0, -50.0])


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_pooled_training_loader(snr_range, samples_per_snr, batch_size):
    """
    Train on data pooled across every SNR in the sweep, so the NN is
    evaluated at conditions it has actually seen -- not extrapolating.
    """
    datasets = [
        PssDataset(num_samples=samples_per_snr, snr_db=float(snr))
        for snr in snr_range
    ]
    combined = ConcatDataset(datasets)
    return DataLoader(combined, batch_size=batch_size, shuffle=True)


def calibrate_correlator_threshold(
    correlator,
    target_pfa=TARGET_CORRELATOR_PFA,
    num_samples=CALIBRATION_SAMPLES,
    snr_db=CALIBRATION_SNR_DB,
    batch_size=BATCH_SIZE,
):
    """Calibrate on independent noise-only data for the requested Pfa."""
    if not 0.0 < target_pfa < 1.0:
        raise ValueError("target_pfa must be strictly between 0 and 1")

    pss_reference = generate_pss(0)
    maximum_scores = []

    for start in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - start)
        noise = generate_noise_vector(
            current_batch_size,
            snr_db,
            pss_reference,
        )
        frequency_offsets = torch.randint(
            low=0,
            high=FFT_BINSIZE - 128,
            size=(current_batch_size,),
        )
        scores = correlator.correlation_scores(frequency_offsets, noise)
        maximum_scores.append(scores.max(dim=1).values)

    maximum_scores = torch.cat(maximum_scores)
    threshold = torch.quantile(maximum_scores, 1.0 - target_pfa).item()
    empirical_pfa = (maximum_scores >= threshold).float().mean().item()
    return threshold, empirical_pfa


def evaluate_at_snr(
    model,
    correlator,
    snr_db,
    num_samples,
    batch_size,
    device,
    correlator_threshold,
):
    """
    Builds a fresh test set at a single SNR and returns
    accuracy, Pd and Pfa percentages for both methods.
    """
    test_loader = DataLoader(
        PssDataset(num_samples=num_samples, snr_db=float(snr_db)),
        batch_size=batch_size,
        shuffle=False,
    )

    model.eval()
    n = torch.arange(FFT_BINSIZE, device=device)

    total = 0
    nn_correct = 0
    corr_correct = 0
    signal_total = 0
    noise_total = 0
    nn_detections = 0
    corr_detections = 0
    nn_false_alarms = 0
    corr_false_alarms = 0

    with torch.no_grad():
        for samples, true_labels, freq_offset in test_loader:
            samples = samples.to(device)
            true_labels = true_labels.to(device)
            freq_offset = freq_offset.to(device)

            samples_corrected = correct_frequency_offset(samples, freq_offset, n)

            nn_outputs = model(samples_corrected)
            nn_predicted = torch.argmax(nn_outputs, dim=1)
            nn_correct += (nn_predicted == true_labels).sum().item()

            corr_outputs = correlator.find_cell_id(
                freq_offset.cpu(),
                samples.cpu(),
                threshold=correlator_threshold,
            )
            corr_predicted = torch.argmax(corr_outputs, dim=1).to(device)
            corr_correct += (corr_predicted == true_labels).sum().item()

            signal_mask = true_labels != 3
            noise_mask = true_labels == 3
            signal_total += signal_mask.sum().item()
            noise_total += noise_mask.sum().item()

            nn_detections += ((nn_predicted != 3) & signal_mask).sum().item()
            corr_detections += ((corr_predicted != 3) & signal_mask).sum().item()
            nn_false_alarms += ((nn_predicted != 3) & noise_mask).sum().item()
            corr_false_alarms += ((corr_predicted != 3) & noise_mask).sum().item()

            total += true_labels.size(0)

    return {
        "nn_accuracy": 100.0 * nn_correct / total,
        "corr_accuracy": 100.0 * corr_correct / total,
        "nn_pd": 100.0 * nn_detections / max(signal_total, 1),
        "corr_pd": 100.0 * corr_detections / max(signal_total, 1),
        "nn_pfa": 100.0 * nn_false_alarms / max(noise_total, 1),
        "corr_pfa": 100.0 * corr_false_alarms / max(noise_total, 1),
    }


def calculate_dynamic_operations(net):
    net.eval()
    dummy_input = torch.randn(1, FFT_BINSIZE, dtype=torch.complex64)

    with torch.no_grad():
        # THOP reports multiply-accumulates (MACs).
        nn_macs, _ = profile(net, inputs=(dummy_input,), verbose=False)

    # Match the implementation: for each of three PSS references there is one
    # complex frequency-shift multiplication and one correlation multiplication
    # per sample. A complex multiply costs 4 real multiplications + 2 additions.
    complex_multiply_flops = NUM_PSS_SEQUENCES * 2 * FFT_BINSIZE * 6
    accumulation_flops = NUM_PSS_SEQUENCES * (FFT_BINSIZE - 1) * 2
    correlator_flops = complex_multiply_flops + accumulation_flops

    # Use the same approximate unit on both sides: one MAC ~= two real FLOPs.
    nn_flops = 2 * nn_macs
    return int(correlator_flops), int(nn_flops)


def generate_publication_quality_plots():
    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    net = EfficientPssCnn().to(device)
    corr = Correlator()

    # ---- Train once, on data pooled across the whole SNR sweep ----
    train_loader = build_pooled_training_loader(
        SNR_RANGE, TRAIN_SAMPLES_PER_SNR, BATCH_SIZE
    )
    train_model(net, train_loader, epochs=TRAIN_EPOCHS)

    # Calibrate the correlator on data that is separate from every test set.
    correlator_threshold, calibration_pfa = calibrate_correlator_threshold(corr)
    print(
        f"Calibrated correlator threshold={correlator_threshold:.5f} | "
        f"validation Pfa={100.0 * calibration_pfa:.2f}%"
    )

    # ---- Evaluate both models at every SNR point ----
    nn_accuracies = []
    corr_accuracies = []
    nn_pd_values = []
    corr_pd_values = []
    nn_pfa_values = []
    corr_pfa_values = []

    for snr in SNR_RANGE:
        metrics = evaluate_at_snr(
            net,
            corr,
            snr,
            TEST_SAMPLES_PER_SNR,
            BATCH_SIZE,
            device,
            correlator_threshold,
        )
        nn_accuracies.append(metrics["nn_accuracy"])
        corr_accuracies.append(metrics["corr_accuracy"])
        nn_pd_values.append(metrics["nn_pd"])
        corr_pd_values.append(metrics["corr_pd"])
        nn_pfa_values.append(metrics["nn_pfa"])
        corr_pfa_values.append(metrics["corr_pfa"])
        print(
            f"SNR={snr:6.1f} dB | "
            f"NN: acc={metrics['nn_accuracy']:6.2f}% "
            f"Pd={metrics['nn_pd']:6.2f}% Pfa={metrics['nn_pfa']:6.2f}% | "
            f"Correlator: acc={metrics['corr_accuracy']:6.2f}% "
            f"Pd={metrics['corr_pd']:6.2f}% Pfa={metrics['corr_pfa']:6.2f}%"
        )

    nn_accuracies = np.array(nn_accuracies)
    corr_accuracies = np.array(corr_accuracies)
    nn_pd_values = np.array(nn_pd_values)
    corr_pd_values = np.array(corr_pd_values)
    nn_pfa_values = np.array(nn_pfa_values)
    corr_pfa_values = np.array(corr_pfa_values)

    correlator_flops, nn_flops = calculate_dynamic_operations(net)
    operation_counts = np.array([correlator_flops, nn_flops], dtype=float)

    print(
        "Calculated complexity -> "
        f"Correlator: {correlator_flops:,} approximate real FLOPs | "
        f"NN (CNN): {nn_flops:,} approximate real FLOPs"
    )

    # Publication-oriented typography
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    correlator_color = "#0072B2"
    nn_color = "#009E73"
    baseline_color = "#8B1A1A"
    grid_color = "#CFCFCF"

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.1, 5.7),
        constrained_layout=True,
    )
    ax1, ax2, ax3, ax4 = axes.ravel()

    # Panel (a): Accuracy versus SNR
    ax1.plot(
        SNR_RANGE, corr_accuracies, color=correlator_color, linewidth=1.8,
        marker="o", markersize=5.2, markerfacecolor="white", markeredgewidth=1.1,
        label="Classic correlator", zorder=3,
    )
    ax1.plot(
        SNR_RANGE, nn_accuracies, color=nn_color, linewidth=1.8, linestyle="--",
        marker="s", markersize=5.0, markerfacecolor="white", markeredgewidth=1.1,
        label="Efficient CNN", zorder=3, # Updated label here too!
    )
    ax1.axhline(100 / 4, color=baseline_color, linewidth=1.2, linestyle=":", label="Random guessing", zorder=2)

    ax1.set_title("(a) Four-class accuracy", loc="left", pad=6)
    ax1.set_xlabel("SNR (dB)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xlim(SNR_RANGE.min() - 3, SNR_RANGE.max() + 3)
    ax1.set_ylim(15, 103)
    ax1.xaxis.set_major_locator(MultipleLocator(10))
    ax1.yaxis.set_major_locator(MultipleLocator(10))
    ax1.grid(axis="both", linestyle=":", linewidth=0.6, color=grid_color, zorder=0)
    ax1.legend(loc="upper left", frameon=False, handlelength=2.3)

    # Panel (b): probability of detection.
    ax2.plot(
        SNR_RANGE, corr_pd_values, color=correlator_color, linewidth=1.8,
        marker="o", markersize=5.2, markerfacecolor="white", markeredgewidth=1.1,
        label="Classic correlator", zorder=3,
    )
    ax2.plot(
        SNR_RANGE, nn_pd_values, color=nn_color, linewidth=1.8, linestyle="--",
        marker="s", markersize=5.0, markerfacecolor="white", markeredgewidth=1.1,
        label="Efficient CNN", zorder=3,
    )
    ax2.set_title(r"(b) Detection probability $P_d$", loc="left", pad=6)
    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel(r"$P_d$ (%)")
    ax2.set_xlim(SNR_RANGE.min() - 3, SNR_RANGE.max() + 3)
    ax2.set_ylim(-3, 103)
    ax2.xaxis.set_major_locator(MultipleLocator(10))
    ax2.yaxis.set_major_locator(MultipleLocator(20))
    ax2.grid(axis="both", linestyle=":", linewidth=0.6, color=grid_color, zorder=0)
    ax2.legend(loc="lower right", frameon=False, handlelength=2.3)

    # Panel (c): probability of false alarm.
    ax3.plot(
        SNR_RANGE, corr_pfa_values, color=correlator_color, linewidth=1.8,
        marker="o", markersize=5.2, markerfacecolor="white", markeredgewidth=1.1,
        label="Classic correlator", zorder=3,
    )
    ax3.plot(
        SNR_RANGE, nn_pfa_values, color=nn_color, linewidth=1.8, linestyle="--",
        marker="s", markersize=5.0, markerfacecolor="white", markeredgewidth=1.1,
        label="Efficient CNN", zorder=3,
    )
    ax3.axhline(
        100.0 * TARGET_CORRELATOR_PFA,
        color=baseline_color,
        linewidth=1.2,
        linestyle=":",
        label="Correlator target",
        zorder=2,
    )
    pfa_upper_limit = max(
        5.0,
        1.2 * float(max(nn_pfa_values.max(), corr_pfa_values.max())),
    )
    ax3.set_title(r"(c) False-alarm probability $P_{fa}$", loc="left", pad=6)
    ax3.set_xlabel("SNR (dB)")
    ax3.set_ylabel(r"$P_{fa}$ (%)")
    ax3.set_xlim(SNR_RANGE.min() - 3, SNR_RANGE.max() + 3)
    ax3.set_ylim(0, min(100.0, pfa_upper_limit))
    ax3.xaxis.set_major_locator(MultipleLocator(10))
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax3.grid(axis="both", linestyle=":", linewidth=0.6, color=grid_color, zorder=0)
    ax3.legend(loc="upper right", frameon=False, handlelength=2.3)

    # Panel (d): computational complexity.
    model_names = ["Classic\ncorrelator", "Efficient CNN"]
    bars = ax4.bar(
        model_names, operation_counts, color=[correlator_color, nn_color],
        width=0.56, edgecolor="#333333", linewidth=0.5, zorder=3,
    )
    bars[1].set_hatch("///")

    ax4.set_title("(d) Computational complexity", loc="left", pad=6)
    ax4.set_ylabel("Approximate real FLOPs")
    ax4.set_ylim(0, operation_counts.max() * 1.24)
    ax4.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax4.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if v >= 1000 else f"{v:.0f}")
    )
    ax4.grid(axis="y", linestyle=":", linewidth=0.6, color=grid_color, zorder=0)

    for bar, value in zip(bars, operation_counts):
        ax4.annotate(
            f"{int(value):,}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )

    # Dynamic calculation of reduction (will automatically adapt if CNN has more/less ops)
    if operation_counts[1] < operation_counts[0]:
        reduction = 100 * (1 - operation_counts[1] / operation_counts[0])
        ratio = operation_counts[0] / operation_counts[1]
        text_str = f"{reduction:.1f}% fewer operations\n{ratio:.2f}\u00d7 reduction"
    else:
        increase = 100 * (operation_counts[1] / operation_counts[0] - 1)
        ratio = operation_counts[1] / operation_counts[0]
        text_str = f"{increase:.1f}% more operations\n{ratio:.2f}\u00d7 increase"

    ax4.text(
        0.97, 0.96,
        text_str,
        transform=ax4.transAxes, ha="right", va="top", fontsize=7.5,
    )

    for ax in (ax1, ax2, ax3, ax4):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", width=0.7, length=3)
        ax.set_axisbelow(True)

    fig.text(
        0.5,
        -0.015,
        "Assumptions: perfect timing, exact frequency-offset knowledge, "
        "AWGN-only channel, and one aligned 2048-sample observation window.",
        ha="center",
        va="top",
        fontsize=7.5,
    )

    fig.savefig("academic_results_figure_cnn.pdf", bbox_inches="tight")
    fig.savefig("academic_results_figure_cnn.svg", bbox_inches="tight")
    fig.savefig("academic_results_figure_cnn.png", dpi=600, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    generate_publication_quality_plots()
