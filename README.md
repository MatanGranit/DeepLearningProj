# Compact PSS detection

This project detects one of three known Primary Synchronization Signal (PSS)
sequences, or declares that no PSS is present. Classes 0, 1 and 2 identify the
PSS sequence; class 3 is noise only. Every observation contains 2,048 complex
time samples, and the four classes are sampled with equal probability.

Two neural detectors are compared against the
magnitude correlator, `|<template, x>| / ||x||` (57,359 operations per window on 2,048
samples, 3,599 on 128):

| Experiment | Signal phase | Detector | Stored results |
|---|---|---|---|
| Known bin, 128 samples (`known-bin-128`) | independent, uniform on [0, 2π) | `NarrowbandPssCnn` | `results_known_bin_128/` |
| Unknown bin, 2,048 samples (`unknown-bin`, the default) | independent, uniform on [0, 2π) | `ConvScanningPssCnn` | `results_unknown_bin/` |

| Detector | Parameters | Ops per window | Of the magnitude correlator's | Pd at −14 dB, 1% Pfa (detector vs magnitude correlator) |
|---|---:|---:|---:|---|
| `NarrowbandPssCnn` (128 samples; against the 128-sample correlator) | 386 | 2,058 | 57% | 44.47% vs 45.30% |
| `ConvScanningPssCnn` (bin unknown; the 2,048-sample correlator is told it) | 6,146 | 56,837 | 99.1% | 18.58% vs 48.42% |

- With a **known bin** and **128 samples** (`--experiment known-bin-128`)
  both receivers get the window with its frequency offset removed and keep the 128 FFT
  bins that hold the PSS, then score those samples. With an unknown phase the magnitude
  correlator is the optimal detector. The network's weights start random (20 epochs,
  learning rate 1e-3); training starts at chance (validation Pd 1–3%) and learns the
  matched filter: the trained weights match the PSS templates with a cosine similarity
  of 0.987–0.993, and the detection ends within about 0.1 dB of the correlator's at
  every SNR. The PSS spectrum on those bins is real, so each template is conjugate symmetric,
  `t[128-n] = conj(t[n])`; `NarrowbandPssCnn` ties its weights to that symmetry, so
  samples `n` and `128-n` share their pre-additions and the multiplies halve. The
  correlator is counted as the direct correlation: using the same symmetry it would
  need about 2,315 operations, so on equal terms the network saves about 11%, from its
  L1 norm. The 128-bin band selection, the front end of both receivers, is not counted.
- With an **unknown bin** (`--experiment unknown-bin`) the network gets the window with
  its frequency offset still in it (one of 24 positions, 0–1840 bins in steps of 80) and
  searches every position, while the correlator is handed the window with the offset
  removed. `ConvScanningPssCnn` is a convolutional network: a global depthwise complex
  convolution (the window as 128 channels of 16 samples, one learned kernel per channel)
  weights the window by each identity's template and folds it to 128 samples; seven fixed
  dilated convolutions (kernel 2, taps `[1, ±1]`, dilations 64 to 1, plus twiddles) form
  a 128-point DFT pruned to the 24 needed bins, whose bin `k/16` is the correlation at
  offset `k`; the score is the largest `|z|^2` over the 24 offsets over L1². It costs no
  more than the correlator (56,837 against 57,359 operations), but detects about 2 dB
  later: testing 24 offsets × 3 identities instead of 3 raises the threshold for the same
  Pfa. Only the first layer learns; the dilated layers are fixed because they compute the
  DFT. Its kernels start at the conjugate templates (8 epochs, learning rate 1e-4).
  It also reports **where** it found the PSS: the max-pool's argmax is the frequency
  offset. Of the detected windows, the reported offset is the true one in 95.6% at
  −14 dB, 98.7% at −12 dB and 99.7–100% from −10 dB up, at no extra operations.

Full per-SNR tables are in each results folder's `RESULTS.md`.

Every SNR in this project is **per PSS subcarrier**; see [SNR definition](#snr-definition).

## Quick start

Python 3.10 or newer. From the project root:

```bash
python -m venv .venv
```

Activate it — Windows: `.venv\Scripts\activate`; Linux/macOS: `source .venv/bin/activate`. Then:

```bash
python -m pip install -r requirements.txt
```

Check the installation; all 22 tests should pass (`OK`):

```bash
python -m unittest discover -s tests -v
```

Train and test the unknown-bin CNN with every default, then print its test results
(the same as `python main.py train`; about 15–20 minutes on a CPU):

```bash
python main.py
```

Show the options:

```bash
python main.py --help
```

Redraw the stored results (figures and `RESULTS.md`, no training):

```bash
python main.py report
```

```bash
python main.py report --experiment known-bin-128
```

Draw the PSS correlation figure (`figures/pss_correlation.png`):

```bash
python main.py correlation
```

A short training run, about a minute on a CPU, into a new folder under `runs/` (one
seed, two epochs, smaller test sets, tested at three SNRs):

```bash
python main.py train --seeds 91 --epochs 2 --train-per-snr 300 --test-samples 2000 --calibration-samples 3000 --snrs -17 -14 -11
```

A full run with the default settings (3 seeds, 16 SNRs; about 15–20 minutes per
experiment on a CPU):

```bash
python main.py train
```

```bash
python main.py train --experiment known-bin-128
```

## Commands

`main.py` has three modes; `train` and `report` take `--experiment
{unknown-bin,known-bin-128}` (default `unknown-bin`):

- **`train`** — create the datasets, train, validate and test, save the checkpoints
  and the results JSON, then write the report (figures and `RESULTS.md`). Output goes
  to `--output`, by default a new folder `runs/<experiment>-<date>-<time>/`, so a run
  never touches the stored results. All other options are training options
  (`--seeds`, `--epochs`, `--train-per-snr`, `--test-samples`, `--snrs`,
  `--device`, ...); `python main.py train --help` lists them with their defaults.
  Training always uses every SNR of `SNR_RANGE`; `--snrs` chooses only the SNRs the
  thresholds are calibrated and the receivers tested at.
  - `--no-test` skips the test sweep. Checkpoints are still saved (with their
    calibrated thresholds), and the results — training histories only — go to
    `training_results.json`. There is then nothing to plot, so no report is written.
  - `--no-report` skips the figures and `RESULTS.md`.
  - Every experiment refuses an `--output` that already holds results.
- **`report`** — only write the report, next to an existing results JSON. Nothing is
  trained. Without `--results FILE` it uses the experiment's stored results
  (`results_known_bin_128/known_bin_128_results.json` or
  `results_unknown_bin/unknown_bin_results.json`).
  A results file named like another experiment's is always reported as that
  experiment's result, whatever `--experiment` says.

```bash
python main.py report --results runs/known-bin-128-20261012-101500/known_bin_128_results.json
```

- **`correlation`** — draw `pss_correlation.{png,pdf,svg}` into `--output` (default
  `figures/`), correlation magnitudes computed in the time domain. Panel (a) plots,
  over the time lag, PSS 0 with itself, PSS 0 moved up by one subcarrier with PSS 0,
  and PSS 1 with PSS 0: one sharp peak of 1 at lag 0 for the autocorrelation, and
  nothing above about 0.2 for the other two. Panel (b) plots the peak magnitude of
  PSS 0 against a frequency-shifted copy of itself, over the residual frequency
  offset: half a subcarrier costs about 4 dB and a whole one leaves no more
  correlation than another PSS — why every receiver works on windows whose frequency
  offset has been corrected exactly.

## Where everything is

```text
main.py                         Main: the entry point, modes `train`, `report` and `correlation`
utils/
  data/
    pss_signal.py               PssSignalGenerator: PSS sequences, AWGN, frequency shift + correction
    pss_dataset.py              PssDataset: the four-class torch Dataset and its pooled-SNR loaders
  models/
    receiver.py                 Receiver: the interface every detector implements
    networks.py                 PssDetector (shared base), NarrowbandPssCnn,
                                ConvScanningPssCnn
    correlator.py               Correlator (2,048 samples), NarrowbandCorrelator (128 samples)
  experiments.py                KnownBinExperiment, UnknownBinExperiment: what each
                                experiment trains, its seeds, options and results format
  trainer.py                    Trainer: create datasets -> train -> validate -> test -> save
  evaluation.py                 Evaluator: test-stage tools (thresholds, test metrics, arithmetic cost)
  metrics.py                    DetectionMetrics: confusion matrix -> accuracy, Pd, Pfa
  reporting.py                  KnownBinReporter, UnknownBinReporter: figures + RESULTS.md only
  correlation_figure.py         PssCorrelationFigure: the PSS correlation figure
tests/                          unit tests
results_known_bin_128/          known bin, 128 samples: results of NarrowbandPssCnn
results_unknown_bin/            unknown bin: results of ConvScanningPssCnn
figures/                        pss_correlation; known_bin_128_network and unknown_bin_network (the CNN diagrams)
runs/                           new training runs (created on first use)
```

Each results folder holds the results JSON, `RESULTS.md`, the figure (`.png`,
`.pdf`, `.svg`) and the trained checkpoints (`.pth`).

Who calls whom:

```text
Main.train  ──> Trainer(experiment).run()
                  │   for every seed:
                  ├─ create_datasets      PssDataset.pooled_loader (train + validation)
                  ├─ train_models         for each detector: fit = train_epoch + validate per epoch
                  ├─ test (optional)      + Correlators; Evaluator: arithmetic cost,
                  │                       per SNR calibrate_thresholds -> evaluate_receivers
                  ├─ save_checkpoints     deployment threshold + .pth per trained detector
                  └─ results JSON         written after every seed and at the end
            ──> Reporter(results).render()   figures + RESULTS.md

Main.report ──> Reporter.from_file(results JSON).render()   (no training)
```

The experiment classes only *describe* an experiment; the `Trainer` runs it. The
reporters only read a results JSON and draw it; they never train or test.

## The 128-sample detector

`NarrowbandPssCnn` (`utils/models/networks.py`) scores the 128-sample view of a
frequency-corrected window. The PSS occupies only 127 of the 2,048 FFT bins
(1024..1150, next to Nyquist), so most of the window is noise; keeping those 128 bins
and inverse-transforming them is an ideal band-pass and decimation by 16
(`PssSignalGenerator.narrowband`), after which the matched filter is applied:

```text
corrected window, 2048 complex samples
  -> keep the 128 FFT bins that hold the PSS            -> 128 samples y
  -> complex layer, 3 x 128 tied weights: z_k = sum conj(w_k) y, one per PSS identity
  -> score:  |z_k|^2 / L1^2
  -> 4 logits: three PSS scores and a noise logit
```

- The weights are conjugate symmetric, `w[128-n] = conj(w[n])`: 65 real and 63 imaginary
  trained values per identity. Samples `n` and `128-n` share their pre-additions, which
  halves the multiplies in the operation count.
- L1 is `sum |Re| + |Im|` over the 128 samples, a noise-level estimate that makes the
  score scale invariant, so one calibrated threshold holds at every noise power.
  `|z_k|` ignores the common phase.
- **Training updates every layer** by backpropagation (cross-entropy, AdamW): the tied
  complex weights, an output scale and the noise logit. `NarrowbandPssCnn` starts at random
  unit-norm weights; `set_template_weights()` sets the unit-norm templates instead, where
  `z_k` is the magnitude correlator's correlation. Training recovers the templates. The epoch kept is the
  one with the highest validation Pd at the target Pfa (threshold set on the validation
  noise-only windows), not the lowest loss: cross-entropy can fall while Pd at a fixed
  Pfa gets worse.

The operation counts treat `|.|` in the L1 norm as a sign flip folded into the
addition.

## The unknown-bin detector

`ConvScanningPssCnn` (`utils/models/networks.py`) scores the 2,048-sample window as
received, frequency offset included, for each PSS identity `t`:

```text
window, 2048 complex samples = 128 channels x 16 samples (channel r: samples r, r+128, ...)
  -> global depthwise complex convolution, one 16-tap kernel per channel   -> u[r], 128 samples
  -> 7 fixed dilated convolutions, kernel 2 [1, +-1], dilations 64 ... 1, + twiddles
     (the radix-2 butterflies of a 128-point DFT, pruned to the 24 needed bins)
  -> z at each of the 24 offsets; score:  max over offsets of |z|^2 / L1^2
  -> 4 logits: three PSS scores and a noise logit
```

- The depthwise kernels are the 2,048 weights `w` per identity, learned and tied to the
  template's conjugate symmetry, `w[2048-n] = conj(w[n])`: 1,025 real and 1,023 imaginary
  trained values per identity, 6,144 in all (6,146 parameters with the output scale and
  the noise logit). They start at the conjugate templates, where `z` at offset `k` is the
  magnitude correlator's correlation with the template shifted by `k` bins.
- **Only the first layer learns.** The dilated layers have fixed taps and twiddles because
  they compute the DFT; they carry no parameters.
- **It reports the offset it found.** `forward_with_offsets(x)` returns the logits and, per
  identity, the offset (in FFT bins) where the max-pool found the maximum; `forward(x)`
  returns only the logits, so training ignores the offset. `predict_with_offset(x)` gives
  the decision and the reported identity's offset (-1 when no PSS is reported). The test
  stage counts, per SNR, how many detections also got the true offset: the
  "Frequency offset found" table in `results_unknown_bin/RESULTS.md` and the figure
  `results_unknown_bin/offset_accuracy.png`.

## Training and evaluation configuration

| Setting | `known-bin-128` | `unknown-bin` |
|---|---:|---:|
| Loss | Cross-entropy | Cross-entropy |
| Optimizer | AdamW | AdamW |
| Learning rate | 1e-3 | 1e-4 |
| Weight decay | 1e-4 | 1e-4 |
| Epochs | 20 | 8 |
| Batch size | 128 | 128 |
| Learning-rate schedule | Cosine annealing | Cosine annealing |
| Gradient clipping | 5.0 | 5.0 |
| Training seeds | 91, 92, 93 | 91, 92, 93 |
| Training samples per SNR | 3,000 | 3,000 |
| Validation samples per SNR | 400 | 400 |
| Calibration noise windows per SNR | 10,000 | 10,000 |
| Test windows per SNR | 8,000 | 8,000 |
| False-alarm target | 1% | 1% |

Those are the command-line defaults, and the stored runs used them. Training and validation always pool every
SNR of `SNR_RANGE`; the `config` block of each results JSON records the other
settings, and its `models` block the network class of each trained detector.

The SNR sweep, `SNR_RANGE` in `utils/experiments.py`, is dense around the detection
knee (about 50% Pd near −14 dB per PSS subcarrier):

```text
-26, -23, -21, -19, -18, -17, -16, -15, -14, -13, -12, -11, -9, -7, -5, 5 dB
```

The stored results test on denser grids: `results_known_bin_128/` every 0.5 dB from −22 to −8 dB plus −26, −24, −7,
−5 and 5 dB; `results_unknown_bin/` every 0.5 dB from −20
to −5 dB plus −26, −24, −22, 0 and 5 dB.

Training, validation, calibration and test streams are seeded from widely
separated bases (`SeedPlan` in `utils/experiments.py`).

### How thresholds are set

Detection is never a plain `argmax`. For each receiver,
`Evaluator.calibrate_thresholds` draws noise-only windows — no signal, no test
data — and takes the `1 - target_pfa` quantile of their presence scores. That
threshold is then frozen for testing. Because every score in this project is
normalized by the input magnitude, one threshold is valid at every noise power.

## Classical receivers

The **magnitude correlator** computes the magnitude of the normalized complex
correlation with each PSS template, `|<template, x>| / ||x||`, and picks the largest.
It ignores the phase, which makes it the optimal detector when the phase is unknown;
it is the classical reference in every report.

`Correlator` works on 2,048 samples; `NarrowbandCorrelator` applies it to the 128-sample
view. `Correlator(coherent=True)` keeps only the real part of the correlation instead;
no experiment runs it.

### Arithmetic cost

Counted by `Evaluator.arithmetic_cost` from each receiver's `operation_counts()`:
multiply-accumulate as two operations, everything else as one, with comparisons
and memory movement excluded. The exact frequency correction is common to all
receivers and is excluded from every count.

| Receiver | Parameters | MACs | Approximate real operations | Stored bytes |
|---|---:|---:|---:|---:|
| `NarrowbandPssCnn` | 386 | 768 | 2,058 | 1,552 |
| Magnitude correlator, 128 samples | 0 | 1,536 | 3,599 | 6,144 |
| `ConvScanningPssCnn` | 6,146 | 0 | 56,837 | 37,828 |
| Magnitude correlator, 2,048 samples | 0 | 24,576 | 57,359 | 98,304 |

The 128-sample correlator is counted as the direct correlation; with the same
conjugate-symmetry pairing as the network it would need about 2,315 operations.

## How to understand the figures

Panels (a)–(c) plot four-class accuracy, Pd and Pfa against SNR; panel (d) is the
arithmetic cost of each receiver per window.

The horizontal axis is the SNR per PSS subcarrier, the channel condition presented
to every receiver (see [SNR definition](#snr-definition)). Results that record no
SNR definition label it "Input SNR".

- **Four-class accuracy:** correct PSS identity or correct noise declaration.
  Uniform guessing gives 25% expected accuracy.
- **Pd:** probability of declaring that a signal is present when a signal is
  present. A wrong PSS identity still counts as detection, so Pd must be read
  together with four-class accuracy.
- **Pfa:** probability of declaring a signal on a noise-only window. Thresholds
  target 1% Pfa and are frozen before testing; the shaded bands are 95% intervals.
- **Arithmetic:** detector operations only; the exact CFO correction common to every
  receiver is excluded.
- **Offset accuracy** (`results_unknown_bin/offset_accuracy.png`): Pd, the share of
  detected windows reported at the true frequency offset, and the share of signal
  windows both detected and placed at the true offset.

## Use a trained checkpoint

Checkpoints written by `python main.py train` (including the stored
`results_known_bin_128/` and `results_unknown_bin/`
ones) record their network class under `model_class`:

```python
import torch
from utils.data.pss_signal import PssSignalGenerator
from utils.models import networks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("results_known_bin_128/narrow_cnn_seed91.pth", map_location=device, weights_only=True)
model = getattr(networks, checkpoint["model_class"])(**checkpoint.get("model_kwargs", {})).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

# samples: complex tensor [batch, 2048]
# offsets: exact frequency offsets [batch]
# corrected = PssSignalGenerator.correct_frequency_offset(samples.to(device), offsets.to(device))
# with torch.inference_mode():
#     labels = model.predict(corrected)  # 0, 1, 2 or 3
```

An unknown-bin checkpoint (`results_unknown_bin/scan_cnn_seed91.pth`) loads the same
way, but `predict()` takes the window as received, without the frequency correction, and
`predict_with_offset()` also returns the frequency offset (in FFT bins) where it found the PSS.

`forward()` returns training logits. `predict()` applies the independently
calibrated detection threshold, which travels inside `state_dict()` as a
registered buffer. Applying `argmax` directly is not equivalent to the reported
1% false-alarm operating point, and `predict()` raises `RuntimeError` if the
threshold was never calibrated.

## SNR definition

The SNR is defined **per PSS subcarrier**, as in 3GPP's per-resource-element view:

```text
SNR = |X[k]|^2 / E|N[k]|^2
```

the power of one occupied FFT bin of the PSS — `|X[k]|^2 = 1`, since the PSS is
BPSK — over the noise power in one FFT bin. `generate_noise_vector` in
`utils/data/pss_signal.py` sets the time-domain noise variance to
`s^2 = 1 / (N * SNR)` with `N = 2048`, because time-domain noise of variance `s^2`
puts `N * s^2` into every bin of the unnormalized FFT.
`tests.test_signal_pipeline.SignalPipelineTests.test_generated_noise_has_requested_snr`
measures it back from the FFT to within 0.15 dB.

Averaged over the time samples instead, the same window has an SNR

```text
SNR_time = SNR_subcarrier - 10*log10(2048 / 127) = SNR_subcarrier - 12.08 dB
```

because the PSS fills 127 of the 2,048 bins while the noise fills all of them.

## Assumptions and limitations

All experiments assume:

- aligned 2,048-sample observation windows;
- exact frequency-offset knowledge and correction for the correlator, and for the
  network in the known-bin experiments;
- AWGN only;
- one constant phase across each PSS window;
- no multipath, fading, timing search or blind CFO search;
- representative noise-only data for threshold calibration.

Timing remains exact. Frequency offsets are drawn on a coarse grid — multiples of 80
FFT bins, 0 to 1,840 — rather than uniformly over all bins; the unknown-bin network
searches exactly that grid.
