# Measured validation results

Generated from the measured data in benchmark_results.json.

Training seeds: [0, 1, 2]. Epochs: 8. Training samples/SNR: 3000.

Calibration noise windows per SNR/seed: 10000; test windows per SNR/seed: 8000.

Every model is calibrated to the same 1% target; measured test Pfa is reported below. A target is not an exact test guarantee.

## Accuracy / Pd / Pfa (%)

| SNR | Compact neural detector | Magnitude correlator | Coherent correlator |
|---|---|---|---|
| -50 | 24.88 / 1.12 / 0.64 | 24.72 / 1.08 / 0.86 | 24.91 / 1.27 / 0.78 |
| -45 | 26.27 / 1.77 / 0.86 | 25.80 / 1.23 / 1.02 | 26.26 / 1.74 / 0.84 |
| -40 | 26.33 / 2.73 / 1.38 | 25.44 / 1.62 / 1.31 | 26.38 / 2.76 / 1.33 |
| -38 | 27.12 / 3.53 / 0.94 | 25.76 / 1.52 / 0.94 | 27.21 / 3.67 / 0.99 |
| -36 | 28.07 / 4.92 / 0.93 | 26.06 / 2.32 / 1.10 | 28.13 / 4.96 / 0.93 |
| -34 | 30.58 / 8.30 / 0.97 | 27.27 / 3.93 / 0.95 | 30.79 / 8.65 / 1.05 |
| -32 | 34.70 / 13.59 / 0.85 | 29.18 / 6.38 / 0.98 | 35.33 / 14.52 / 0.78 |
| -30 | 42.22 / 23.61 / 0.60 | 33.70 / 12.41 / 0.75 | 43.00 / 24.63 / 0.57 |
| -28 | 55.73 / 41.20 / 0.99 | 44.15 / 25.86 / 1.19 | 56.65 / 42.54 / 0.95 |
| -26 | 75.69 / 68.12 / 1.08 | 61.82 / 49.75 / 1.14 | 76.61 / 69.40 / 1.14 |
| -24 | 91.95 / 89.61 / 0.69 | 83.05 / 77.95 / 0.89 | 93.01 / 91.08 / 0.82 |
| -22 | 98.81 / 98.80 / 1.06 | 97.38 / 96.86 / 0.92 | 98.99 / 99.01 / 0.99 |
| -20 | 99.73 / 100.00 / 1.06 | 99.65 / 99.94 / 1.23 | 99.72 / 100.00 / 1.09 |
| -10 | 99.73 / 100.00 / 1.07 | 99.72 / 100.00 / 1.10 | 99.72 / 100.00 / 1.10 |
| 0 | 99.78 / 100.00 / 0.87 | 99.80 / 100.00 / 0.81 | 99.79 / 100.00 / 0.84 |

## Cost and measured latency

Times are medians across runs, in microseconds per window. CPU/GPU data transfer is excluded; +CFO includes the common frequency correction. These are local measurements.

| Receiver | Parameters | Conv/Linear MACs | Real ops incl. normalization | Stored bytes | Batch 1 core | Batch 1 +CFO | Batch 64 core | Batch 64 +CFO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Compact neural detector | 1,668 | 17,920 | 44,039 | 6,676 | 117.30 | 192.70 | 4.19 | 66.31 |
| Magnitude correlator | 0 | 24,576 | 57,359 | 98,304 | 67.60 | 152.80 | 3.15 | 63.53 |
| Coherent correlator | 0 | 12,288 | 32,771 | 49,152 | 53.10 | 138.00 | 2.85 | 59.55 |

The neural detector advantage over the magnitude correlator exploits the fixed-phase model. The coherent correlator is the stronger, lower-cost classical reference in this model; this experiment does not demonstrate a fundamental neural advantage over it.

Shared correction is roughly 8N real arithmetic operations plus N complex exponential evaluations for N=2048. It is excluded from all detector operation counts and explicitly included in the +CFO latency columns. Operation estimates exclude comparisons and memory movement.

Pfa bands in the plot are pooled 95% Wilson binomial intervals; they do not account for all training/calibration variability. Per-seed counts, thresholds, training times, raw-argmax diagnostics and intervals are retained in JSON.

Environment: {"python": "3.14.6", "torch": "2.12.1+cpu", "platform": "Windows-11-10.0.26200-SP0", "device": "cpu", "threads": 1}
