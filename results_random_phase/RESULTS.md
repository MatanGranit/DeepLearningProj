# Random-phase measured results

The original fixed-phase files in `results/` are retained unchanged.

Selected CNN: `{'block_size': 4, 'rank': 1}`. Seeds: [91, 92, 93]; epochs: 8; training samples/SNR: 3000.

Every signal window has an independent uniform phase. Timing and CFO are still known exactly; this isolates phase robustness and is not blind acquisition.

## Accuracy / Pd / Pfa (%)

| SNR | Phase-invariant complex CNN | Legacy linear network (retrained) | Magnitude correlator | Coherent correlator |
|---|---|---|---|---|
| -50 | 25.10 / 1.15 / 0.85 | 25.01 / 1.12 / 1.15 | 24.99 / 1.14 / 1.22 | 25.05 / 1.16 / 1.18 |
| -45 | 25.42 / 1.27 / 0.97 | 25.17 / 0.99 / 1.17 | 25.38 / 1.10 / 1.02 | 25.33 / 1.20 / 1.04 |
| -40 | 24.80 / 1.42 / 0.94 | 24.60 / 1.13 / 0.92 | 24.79 / 1.26 / 0.87 | 24.73 / 1.32 / 0.95 |
| -38 | 25.58 / 1.55 / 0.85 | 25.15 / 1.14 / 1.28 | 25.75 / 1.92 / 0.98 | 25.60 / 1.76 / 1.21 |
| -36 | 26.08 / 2.30 / 1.13 | 25.23 / 1.06 / 0.80 | 26.42 / 2.88 / 0.98 | 25.73 / 1.74 / 1.00 |
| -34 | 26.06 / 3.28 / 1.04 | 24.46 / 1.29 / 0.84 | 26.56 / 3.98 / 1.03 | 25.52 / 2.65 / 0.92 |
| -32 | 28.10 / 5.57 / 0.91 | 24.91 / 1.15 / 0.71 | 29.19 / 6.92 / 0.88 | 26.84 / 3.83 / 0.81 |
| -30 | 31.66 / 9.97 / 1.10 | 25.15 / 1.54 / 1.39 | 33.36 / 12.05 / 0.99 | 28.92 / 6.21 / 1.15 |
| -28 | 40.12 / 21.12 / 1.10 | 25.48 / 1.98 / 1.33 | 43.62 / 25.61 / 0.83 | 32.15 / 10.37 / 0.70 |
| -26 | 55.06 / 40.88 / 1.00 | 25.93 / 2.43 / 0.82 | 61.14 / 48.80 / 0.75 | 37.23 / 17.21 / 1.00 |
| -24 | 77.08 / 70.06 / 0.91 | 26.62 / 3.76 / 1.01 | 83.44 / 78.44 / 0.92 | 43.49 / 25.75 / 1.06 |
| -22 | 94.57 / 93.18 / 0.92 | 27.78 / 5.37 / 0.82 | 97.13 / 96.51 / 0.84 | 48.27 / 32.00 / 0.96 |
| -20 | 99.52 / 99.66 / 0.91 | 30.96 / 10.10 / 1.08 | 99.67 / 99.89 / 1.01 | 51.30 / 36.10 / 1.03 |
| -10 | 99.77 / 100.00 / 0.93 | 52.26 / 38.22 / 1.15 | 99.75 / 100.00 / 1.01 | 58.88 / 45.79 / 1.15 |
| 0 | 99.78 / 100.00 / 0.88 | 58.11 / 45.28 / 0.94 | 99.80 / 100.00 / 0.80 | 61.10 / 48.01 / 0.96 |

## Model and arithmetic cost

| Receiver | Real trainable parameters | MACs | Approx. real operations | Stored bytes |
|---|---:|---:|---:|---:|
| Phase-invariant complex CNN | 3,085 | 14,336 | 36,887 | 12,344 |
| Legacy linear network (retrained) | 1,668 | 17,920 | 44,039 | 6,676 |
| Magnitude correlator | 0 | 24,576 | 57,359 | 98,304 |
| Coherent correlator | 0 | 12,288 | 32,771 | 49,152 |

The complex CNN uses a tied complex convolution with kernel=stride=4, one complex channel, a complex global projection, and a modulus/energy nonlinearity. Its output is invariant to an arbitrary common phase rotation by construction.

Its arithmetic estimate is 16.2% below the legacy network and 35.7% below the magnitude correlator. At -24 dB its pooled Pd is 70.06% versus 78.44% for the magnitude correlator, at approximately the same 1% Pfa. This is a compute/accuracy trade-off, not an accuracy win over the classical phase-invariant baseline.

The coherent correlator is shown as a deliberately mismatched reference: it does not know the random phase and is not an oracle. The magnitude correlator is the practical phase-invariant classical baseline.
