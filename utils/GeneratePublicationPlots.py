"""
Generates the publication figure with computed accuracies
for both the neural network (CNN) and the classic
correlator, across the given SNR range.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter, MultipleLocator, MaxNLocator
from thop import profile
from torch.utils.data import ConcatDataset, DataLoader

# --- שינוי 1: ייבוא ה-CNN במקום הרשת הישנה ---
from net import EfficientPssCnn, train_model, correct_frequency_offset
from data_loader import PssDataset
from correlator import Correlator

SEED = 0
FFT_BINSIZE = 2048
NUM_CLASSES = 4

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
        PssDataset(num_samples=samples_per_snr, noise_db=float(snr))
        for snr in snr_range
    ]
    combined = ConcatDataset(datasets)
    return DataLoader(combined, batch_size=batch_size, shuffle=True)


def evaluate_at_snr(model, correlator, snr_db, num_samples, batch_size, device):
    """
    Builds a fresh test set at a single SNR and returns
    (nn_accuracy_pct, correlator_accuracy_pct).
    """
    test_loader = DataLoader(
        PssDataset(num_samples=num_samples, noise_db=float(snr_db)),
        batch_size=batch_size,
        shuffle=False,
    )

    model.eval()
    n = torch.arange(FFT_BINSIZE, device=device)

    total = 0
    nn_correct = 0
    corr_correct = 0

    with torch.no_grad():
        for samples, labels_hot_ones, freq_offset in test_loader:
            samples = samples.to(device)
            labels_hot_ones = labels_hot_ones.to(device)
            freq_offset = freq_offset.to(device)

            true_labels = torch.argmax(labels_hot_ones, dim=1)

            samples_corrected = correct_frequency_offset(samples, freq_offset, n)

            nn_outputs = model(samples_corrected)
            nn_predicted = torch.argmax(nn_outputs, dim=1)
            nn_correct += (nn_predicted == true_labels).sum().item()

            corr_outputs = correlator.find_cell_id(freq_offset.cpu(), samples.cpu())
            corr_predicted = torch.argmax(corr_outputs, dim=1).to(device)
            corr_correct += (corr_predicted == true_labels).sum().item()

            total += true_labels.size(0)

    nn_acc = 100.0 * nn_correct / total
    corr_acc = 100.0 * corr_correct / total
    return nn_acc, corr_acc


def calculate_dynamic_operations(net):
    net.eval()
    dummy_input = torch.randn(1, FFT_BINSIZE, dtype=torch.complex64)

    with torch.no_grad():
        # thop will dynamically calculate MACs for your Conv1d layers!
        nn_macs, _ = profile(net, inputs=(dummy_input,), verbose=False)

    complex_multiplications = (
        NUM_CLASSES * FFT_BINSIZE
        + NUM_CLASSES * FFT_BINSIZE
    )
    complex_additions = NUM_CLASSES * (FFT_BINSIZE - 1)

    real_multiplication_ops = 4 * complex_multiplications
    real_addition_ops = 2 * complex_additions

    correlator_ops = real_multiplication_ops + real_addition_ops

    return int(correlator_ops), int(nn_macs)


def generate_publication_quality_plots():
    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- שינוי 2: אתחול של ה-CNN ---
    net = EfficientPssCnn().to(device)
    corr = Correlator()

    # ---- Train once, on data pooled across the whole SNR sweep ----
    train_loader = build_pooled_training_loader(
        SNR_RANGE, TRAIN_SAMPLES_PER_SNR, BATCH_SIZE
    )
    train_model(net, train_loader, epochs=TRAIN_EPOCHS)

    # ---- Evaluate both models at every SNR point ----
    nn_accuracies = []
    corr_accuracies = []

    for snr in SNR_RANGE:
        nn_acc, corr_acc = evaluate_at_snr(
            net, corr, snr, TEST_SAMPLES_PER_SNR, BATCH_SIZE, device
        )
        nn_accuracies.append(nn_acc)
        corr_accuracies.append(corr_acc)
        print(f"SNR={snr:6.1f} dB | NN acc={nn_acc:6.2f}% | Correlator acc={corr_acc:6.2f}%")

    nn_accuracies = np.array(nn_accuracies)
    corr_accuracies = np.array(corr_accuracies)

    correlator_ops, nn_macs = calculate_dynamic_operations(net)
    operation_counts = np.array([correlator_ops, nn_macs], dtype=float)

    print(
        "Calculated complexity -> "
        f"Correlator: {correlator_ops:,} operation equivalents | "
        f"NN (CNN): {nn_macs:,} MACs"
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3.0), constrained_layout=True)

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

    ax1.set_title("(a) Detection accuracy", loc="left", pad=6)
    ax1.set_xlabel("SNR (dB)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xlim(SNR_RANGE.min() - 3, SNR_RANGE.max() + 3)
    ax1.set_ylim(15, 103)
    ax1.xaxis.set_major_locator(MultipleLocator(10))
    ax1.yaxis.set_major_locator(MultipleLocator(10))
    ax1.grid(axis="both", linestyle=":", linewidth=0.6, color=grid_color, zorder=0)
    ax1.legend(loc="upper left", frameon=False, handlelength=2.3)

    # Panel (b): Computational complexity
    model_names = ["Classic\ncorrelator", "Efficient CNN"]
    bars = ax2.bar(
        model_names, operation_counts, color=[correlator_color, nn_color],
        width=0.56, edgecolor="#333333", linewidth=0.5, zorder=3,
    )
    bars[1].set_hatch("///")

    ax2.set_title("(b) Computational complexity", loc="left", pad=6)
    ax2.set_ylabel("Operation count")
    ax2.set_ylim(0, operation_counts.max() * 1.24)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax2.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if v >= 1000 else f"{v:.0f}")
    )
    ax2.grid(axis="y", linestyle=":", linewidth=0.6, color=grid_color, zorder=0)

    for bar, value in zip(bars, operation_counts):
        ax2.annotate(
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

    ax2.text(
        0.97, 0.96,
        text_str,
        transform=ax2.transAxes, ha="right", va="top", fontsize=7.5,
    )

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", width=0.7, length=3)
        ax.set_axisbelow(True)

    fig.savefig("academic_results_figure_cnn.pdf", bbox_inches="tight")
    fig.savefig("academic_results_figure_cnn.svg", bbox_inches="tight")
    fig.savefig("academic_results_figure_cnn.png", dpi=600, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    generate_publication_quality_plots()