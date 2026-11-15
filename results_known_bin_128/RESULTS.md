# Known-bin, 128-sample measured results

Seeds: [91, 92, 93]; epochs: 20; training samples/SNR: 3000.

Every signal window has an independent uniform phase; timing and CFO are known exactly. Both receivers keep the 128 FFT bins that hold the PSS (an ideal band-pass and decimation by 16) and score those 128 samples.

## Accuracy / Pd / Pfa (%)

| SNR per PSS subcarrier (dB) | Neural detector, 128 samples (NarrowbandPssCnn) | Magnitude correlator, 128 samples |
|---|---|---|
| -26 | 25.54 / 1.75 / 1.02 | 25.55 / 1.79 / 1.07 |
| -24 | 26.17 / 2.14 / 0.97 | 26.19 / 2.25 / 1.12 |
| -22 | 26.40 / 3.31 / 0.88 | 26.55 / 3.65 / 1.02 |
| -21.5 | 27.57 / 4.42 / 1.19 | 27.57 / 4.36 / 1.26 |
| -21 | 27.99 / 4.81 / 1.16 | 28.00 / 4.90 / 1.21 |
| -20.5 | 27.78 / 5.72 / 1.32 | 27.85 / 5.83 / 1.28 |
| -20 | 28.73 / 6.34 / 1.06 | 28.71 / 6.36 / 1.06 |
| -19.5 | 29.21 / 6.62 / 0.97 | 29.42 / 6.83 / 0.94 |
| -19 | 30.45 / 8.27 / 1.08 | 30.50 / 8.24 / 1.00 |
| -18.5 | 31.57 / 9.58 / 0.90 | 32.00 / 10.21 / 1.03 |
| -18 | 32.95 / 11.80 / 0.92 | 33.25 / 12.23 / 1.01 |
| -17.5 | 34.59 / 13.72 / 0.67 | 34.30 / 13.24 / 0.60 |
| -17 | 35.73 / 15.24 / 0.74 | 36.12 / 15.77 / 0.82 |
| -16.5 | 39.48 / 20.22 / 1.13 | 40.14 / 21.09 / 1.13 |
| -16 | 42.46 / 23.29 / 0.73 | 43.82 / 25.24 / 0.88 |
| -15.5 | 44.70 / 27.69 / 1.29 | 45.52 / 28.82 / 1.34 |
| -15 | 48.97 / 32.96 / 1.24 | 49.80 / 34.04 / 1.21 |
| -14.5 | 54.62 / 39.77 / 0.93 | 55.41 / 40.82 / 0.91 |
| -14 | 57.82 / 44.47 / 0.92 | 58.43 / 45.30 / 0.87 |
| -13.5 | 64.84 / 53.48 / 0.77 | 65.73 / 54.71 / 0.92 |
| -13 | 70.35 / 61.30 / 1.29 | 71.12 / 62.31 / 1.22 |
| -12.5 | 74.50 / 66.56 / 0.87 | 75.59 / 67.97 / 0.84 |
| -12 | 80.40 / 74.40 / 0.84 | 81.90 / 76.39 / 0.87 |
| -11.5 | 86.51 / 82.34 / 0.85 | 87.47 / 83.63 / 0.90 |
| -11 | 90.50 / 87.79 / 0.94 | 91.32 / 88.89 / 1.04 |
| -10.5 | 93.74 / 92.20 / 1.39 | 94.42 / 93.01 / 1.15 |
| -10 | 96.11 / 95.18 / 0.96 | 96.56 / 95.74 / 0.87 |
| -9.5 | 97.88 / 97.45 / 0.77 | 98.10 / 97.80 / 0.97 |
| -9 | 98.94 / 98.89 / 0.88 | 99.02 / 98.97 / 0.81 |
| -8.5 | 99.45 / 99.59 / 0.96 | 99.48 / 99.68 / 1.11 |
| -8 | 99.65 / 99.87 / 1.01 | 99.69 / 99.90 / 0.92 |
| -7 | 99.74 / 99.97 / 0.98 | 99.78 / 99.97 / 0.83 |
| -5 | 99.75 / 100.00 / 1.02 | 99.75 / 100.00 / 1.00 |
| 5 | 99.80 / 100.00 / 0.80 | 99.81 / 100.00 / 0.76 |

## Model and arithmetic cost

| Receiver | Parameters | MACs | Approx. real operations | Stored bytes |
|---|---:|---:|---:|---:|
| Neural detector, 128 samples (NarrowbandPssCnn) | 386 | 768 | 2,058 | 1,552 |
| Magnitude correlator, 128 samples | 0 | 1,536 | 3,599 | 6,144 |

The neural detector needs 2,058 real operations per window, 42.8% fewer than the magnitude correlator's 3,599. At -14 dB, the swept SNR where the magnitude correlator's Pd is closest to 50%, the neural detector's pooled Pd is 44.47% versus 45.30% for the magnitude correlator, both calibrated to the same 1% Pfa target.
