"""
Evaluate how the trained PSS classifier (and the classic correlator) behave
specifically on NOISE-ONLY input (label 3, "No Signal") -- i.e. no PSS present
at all, just noise on the FFT grid.

For each noise level this measures:
  1. How often the model correctly says "no signal" (class 3).
  2. When it's wrong, which cell ID (0/1/2) it hallucinates out of pure noise.

"""

import os
import math

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data_loader import generate_pss, PssDataset
from net import EfficientPssCnn, correct_frequency_offset, train_model
from correlator import Correlator

FIX_NOISE_BUG = True
FFT_BINSIZE = 2048
MODEL_PATH = "pss_cnn_model.pth"


def generate_noise_only_batch(num_samples: int, noise_db: float, fix_bug: bool = FIX_NOISE_BUG):
    """Pure noise, no PSS at all -- ground truth label is always 3."""
    pss0 = generate_pss(0)
    signal_power = torch.sum(torch.abs(pss0) ** 2) / FFT_BINSIZE
    noise_power = signal_power / (10.0 ** (noise_db / 10.0))
    scale = torch.sqrt(noise_power / 2.0) if fix_bug else noise_power

    noise_real = torch.randn(num_samples, FFT_BINSIZE)
    noise_imag = torch.randn(num_samples, FFT_BINSIZE)
    samples = scale * torch.complex(noise_real, noise_imag)

    # freq_offset is still sampled (it's part of the real inference path,
    # even though it does nothing meaningful to pure noise)
    freq_offset = torch.randint(low=0, high=FFT_BINSIZE - 128, size=(num_samples,))
    labels = torch.full((num_samples,), 3, dtype=torch.long)
    return samples, labels, freq_offset


def evaluate_noise_only(model, correlator, noise_db_values, num_samples=1000, device="cpu"):
    model.eval()
    n = torch.arange(FFT_BINSIZE, device=device)

    nn_preds_by_db = {}
    corr_preds_by_db = {}

    with torch.no_grad():
        for noise_db in noise_db_values:
            samples, _labels, freq_offset = generate_noise_only_batch(num_samples, noise_db)
            samples = samples.to(device)
            freq_offset = freq_offset.to(device)

            samples_corrected = correct_frequency_offset(samples, freq_offset, n)
            nn_outputs = model(samples_corrected)
            nn_pred = torch.argmax(nn_outputs, dim=1).cpu()

            corr_onehot = correlator.find_cell_id(freq_offset.cpu(), samples.cpu())
            corr_pred = torch.argmax(corr_onehot, dim=1)

            nn_preds_by_db[noise_db] = nn_pred
            corr_preds_by_db[noise_db] = corr_pred

    return nn_preds_by_db, corr_preds_by_db


def summarize(preds_by_db, noise_db_values):
    """Returns:
      correct_pct[db]  -> % correctly classified as 'no signal'
      breakdown[db]    -> [count_said_0, count_said_1, count_said_2] among the wrong ones
    """
    correct_pct = []
    breakdown = []
    for db in noise_db_values:
        pred = preds_by_db[db]
        correct_pct.append((pred == 3).float().mean().item() * 100)
        wrong = pred[pred != 3]
        breakdown.append([(wrong == c).sum().item() for c in range(3)])
    return correct_pct, np.array(breakdown)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = EfficientPssCnn().to(device)
    if os.path.exists(MODEL_PATH):
        net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded weights from {MODEL_PATH}")
    else:
        print("No saved weights found -- training a fresh model first...")
        train_loader = DataLoader(PssDataset(num_samples=5000, noise_db=10.0), batch_size=32, shuffle=True)
        train_model(net, train_loader, epochs=5)
        torch.save(net.state_dict(), MODEL_PATH)

    correlator = Correlator()

    noise_db_values = list(range(-40, 21, 5))
    nn_preds, corr_preds = evaluate_noise_only(net, correlator, noise_db_values, num_samples=1000, device=device)

    nn_correct, nn_breakdown = summarize(nn_preds, noise_db_values)
    corr_correct, corr_breakdown = summarize(corr_preds, noise_db_values)

    print(f"\n{'noise_db':>10} | {'NN: correct no-signal %':>26} | {'Correlator: correct no-signal %':>32}")
    for db, nc, cc in zip(noise_db_values, nn_correct, corr_correct):
        print(f"{db:>10} | {nc:>26.1f} | {cc:>32.1f}")

    # ---------------- plotting ----------------
    
    # Configure Matplotlib to use Times New Roman globally
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(noise_db_values, nn_correct, marker='o', linewidth=2, label="Neural net", color="#0072B2")
    ax.plot(noise_db_values, corr_correct, marker='s', linewidth=2, label="Correlator", color="#009E73")
    ax.set_xlabel("noise_db (higher = less noise)")
    ax.set_ylabel("Correctly classified as 'No Signal' (%)")
    ax.set_title("Pure noise input: does it correctly say 'no signal'?")
    ax.set_ylim(-2, 105)
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    x_labels = [str(d) for d in noise_db_values]
    bottom = np.zeros(len(noise_db_values))
    colors = ["#009E73", "#E30F0FD9", "#1D6893"]
    for c in range(3):
        ax.bar(x_labels, nn_breakdown[:, c], bottom=bottom, label=f"hallucinated cell {c}", color=colors[c])
        bottom += nn_breakdown[:, c]
    ax.set_xlabel("noise_db")
    ax.set_ylabel("# false detections (out of 1000 noise-only samples)")
    ax.set_title("When the NET is wrong on noise-only input,\nwhich cell ID does it hallucinate?")
    ax.legend()

    plt.tight_layout()
    plt.savefig("noise_only_behavior.png", dpi=300) # Bumped DPI slightly for clearer text resolution
    print("\nSaved plot to noise_only_behavior.png")
    plt.show()


if __name__ == "__main__":
    main()