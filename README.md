# Compact PSS detection

This project identifies three known Primary Synchronization Signal (PSS) patterns
in noise. Classes 0, 1 and 2 identify the PSS sequence; class 3 means noise only.
Each example contains 2048 complex time samples. The four classes are sampled
with equal probabilities.

Two experiments are included. The original fixed-phase experiment and every
file in `results/` are retained unchanged. Its compact, template-initialized
linear detector is evaluated against magnitude and coherent correlators; the
measured report is in [results](results/RESULTS.md). Across three training seeds, the
network reached 75.69% accuracy at -26 dB and 99.73% at -20 dB, with a 1%
false-alarm calibration target. Measured batch-64 detector time was 3.85
microseconds per window on one CPU thread, or 82.22 microseconds including
frequency correction. The coherent correlator remains faster and slightly more
accurate through the transition region under the stated simulation assumptions.

The added experiment gives every signal window an independent phase sampled
uniformly from 0 to 2π. Its new `PhaseInvariantPssCnn` is compared on identical
test IQ against the old network retrained for random phase, a magnitude
correlator, and a deliberately phase-mismatched coherent correlator. The full
three-seed measurements, checkpoints and graphs are in
[results_random_phase](results_random_phase/RESULTS.md).

## Why the old detector was not really a CNN

`EfficientPssCnn` used two factorized linear projections. Although its first
shared projection is mathematically equivalent to a strided convolution, the
whole signal path had no intermediate nonlinear operation. Therefore the
composition was still one linear mapping followed by class logits. Calling it
a CNN was technically possible in a narrow implementation sense, but misleading
in the usual machine-learning sense.

`PhaseInvariantPssCnn` is a small, physics-guided complex CNN. It applies a
shared complex convolution to non-overlapping four-sample blocks, combines the
features with a learned complex global projection, and then applies a nonlinear
energy operation, `|z|²`. Squaring and summing the real and imaginary outputs
makes the prediction exactly invariant to a common phase rotation. This is why
adding random phase does not require testing many discrete phase hypotheses.

The selected rank-one model has 3,085 trainable real parameters and an estimated
36,887 real operations per window. That is 16.2% below the old network's 44,039
operations and 35.7% below the magnitude correlator's 57,359. At -24 dB and
approximately 1% false alarms, its pooled random-phase Pd is 70.06%, versus
78.44% for the magnitude correlator. At -22 dB the values are 93.18% and 96.51%.
Thus the result is a useful compute/accuracy trade-off, not a claim that the CNN
beats the proper classical phase-invariant baseline in detection accuracy.

## Run it

Python 3.10+ is required. Install the dependencies, then run the focused tests:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Regenerate plots from the supplied numerical results without retraining:

```bash
python main.py
python utils/GenerateRandomPhasePlots.py --from-results results_random_phase/random_phase_results.json
```

Run a new, independent three-seed training and evaluation experiment:

```bash
python utils/GeneratePublicationPlots.py --seeds 0 1 2 --output results/my_run
python run_random_phase.py --output results_random_phase/my_run
```

By default this trains the network for 8 epochs on 45,000 training
examples (3,000 at each of 15 SNR levels). It uses 6,000 separate examples for
model-selection validation, 10,000 noise-only calibration examples per SNR,
and 8,000 fresh test examples per SNR. All receivers see identical test data.
`--device cuda` selects a CUDA GPU when available; measurements in the included
report state the device actually used. `--threads` controls PyTorch CPU threads.

A smaller run to check your installation:

```bash
python utils/GeneratePublicationPlots.py --seeds 10 --epochs 2 --train-per-snr 200 --validation-per-snr 50 --test-samples 1000 --calibration-samples 3000 --output results/smoke
```

## Project files

- `main.py`: regenerate supplied figures or launch a new experiment.
- `utils/net.py`: neural detector, frequency correction and training.
- `utils/data_loader.py`: PSS sequences, noise and four-class examples.
- `utils/correlator.py`: magnitude and coherent classical detectors.
- `utils/evaluation.py`: threshold calibration, metrics and timing.
- `utils/benchmark.py`: reproducible training and evaluation.
- `utils/GeneratePublicationPlots.py`: graphs and numerical report.
- `utils/phase_benchmark.py`: random-phase model selection and fair benchmark.
- `utils/GenerateRandomPhasePlots.py`: random-phase graph and table generator.
- `run_random_phase.py`: convenience entry point for the random-phase experiment.
- `tests/test_signal_pipeline.py`: numerical and inference checks.
- `tests/test_random_phase.py`: phase invariance, cost and random-phase checks.
- `results/`: three trained checkpoints, raw measurements and rendered results.
- `results_random_phase/`: new CNN checkpoints and random-phase results.

The network has 1,668 trainable parameters and uses 17,920 linear-layer MACs
per window. Including normalization, the estimate is 44,039 real arithmetic
operations. Actual latency is measured separately because operation counts
alone do not determine speed.

## The network

The network splits each input into 128 blocks of 16 complex samples. Each
block has 32 real numbers: the real and imaginary parts of those samples.
A shared learned projection converts these 32 numbers to four features. A
global learned projection then combines all 512 features into three PSS scores.
This is equivalent to a strided convolution followed by a global linear layer.

There are no BatchNorm, depthwise convolutions, Dropout or intermediate ReLU
operations. Keeping the intermediate signal path linear lets matching signal
contributions add across the entire observation window. Dividing the three
output scores by input RMS controls the noise-amplitude range with less work
than normalizing every input value. This removes absolute-amplitude information;
it does not remove noise or increase SNR. The final logits include learned
biases for the three signal classes and the noise class.

The projections are initialized using a rank-four SVD approximation to the
three known, clean PSS templates. No received test examples enter initialization.
The weights are then trainable. The default initialization retains about 98.9%
of the templates' combined energy in this block representation. This is a
model-based initialization, not a claim that the network discovers PSS from
scratch. `template_init=False` is available for an initialization ablation.

Training uses cross-entropy, AdamW, cosine learning-rate decay, gradient
clipping, and restoration of the lowest-validation-loss checkpoint. The default
learning rate is 1e-4.

## Use the trained detector

Each included checkpoint contains
the trained weights **and** an independently calibrated threshold:

```python
import torch
from utils.net import EfficientPssCnn, correct_frequency_offset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientPssCnn().to(device)
checkpoint = torch.load("results/pss_model_seed0.pth",
                        map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model"])
model.eval()

# samples: complex tensor [batch, 2048]; offsets: known FFT-bin offsets [batch]
# samples = samples.to(device)
# corrected = correct_frequency_offset(samples, offsets, torch.arange(2048, device=device))
# with torch.inference_mode():
#     labels = model.predict(corrected)   # 0, 1, 2 or 3
```

`forward()` returns raw training logits. Use `predict()` for the calibrated
detection decision, rather than applying `argmax` directly to those logits.
Calling `predict()` before calibration raises an error. After retraining,
recalibrate using `utils.evaluation.calibrate_thresholds` and save the state again.
The deployed checkpoint uses one threshold calibrated at reference SNR -30 dB;
its score is invariant to a common amplitude scale on a nonzero noise window.

## How to understand the graphs

The **signal** is a useful known pattern; **noise** is random interference added
to it. SNR describes their relative average powers. At 0 dB they have equal
power; at -30 dB the noise has 1,000 times the signal power. The horizontal axis
uses the input time-sample SNR, before detection processing. For noise-only
examples, "reference SNR" specifies the noise power relative to a hypothetical
PSS of the usual strength; no useful signal is actually present.

Correlation compares an entire noisy window with a known pattern. Matching
contributions add in an organized way; independent noise contributes a random
sum. In this aligned AWGN model the matched-filter output SNR is `2048` times
the input SNR, equivalent to about `33.1 dB` processing gain. Thus a -30 dB
input can still carry useful detection information after processing. The gain
depends on the stated SNR definition and observation duration; it is not a
free improvement from increasing the sampling rate.

The current main figure has four panels:

- **Accuracy:** the fraction of all examples assigned the correct class.
  Uniform guessing has expected accuracy 25%. Always saying "noise" also
  approaches 25% for equally likely classes, despite having no useful detection.
- **Pd:** among examples that contain a signal, the fraction declared to contain
  a signal. A wrong PSS identity still counts as a presence detection, so this
  number alone does not measure correct identification.
- **Pfa:** among noise-only examples, the fraction incorrectly declared to contain
  a signal. A receiver that always says "signal" has Pd=100% and Pfa=100%, so
  Pd must always be read alongside Pfa. The target line is 1%; shaded bands are
  pooled 95% Wilson binomial intervals, not full training-uncertainty intervals.
- **Measured inference time:** time per window for batch-64 processing on the
  reported device. Smaller is faster. This is distinct from the separate
  `arithmetic_costs.png` plot, which estimates numerical operations.

The benchmark JSON also contains full confusion matrices, correct PSS identity
probability given a signal, individual seed results, raw-argmax diagnostics,
calibration thresholds, and batch-1/batch-64 timings with and without correction.
Read the exact measured values and limitations in [RESULTS.md](results/RESULTS.md).

## Fair comparisons and limits

This simulation assumes perfect timing, an exactly known frequency offset,
fixed signal phase, and AWGN only. It supplies no multipath, unknown phase,
blind frequency search, or timing search. PSS identity here is only one of
three PSS indices, not the complete physical cell ID.

The magnitude correlator takes an absolute complex correlation. The additional
coherent correlator uses the known fixed phase. The network also exploits that phase;
its advantage over the magnitude correlator must be interpreted with this
assumption in mind. A coherent correlator is a strong, cheaper classical
reference for this model. These results do not establish that the network beats an
optimal classical detector or demonstrate a new research contribution.

For the multi-model benchmark, each background noise power has a separate
noise-only calibration set. Thresholds are frozen before each test set; no test labels, signal
samples or test accuracy are used for calibration. This presumes access to
representative noise-only observations at deployment, rather than knowledge
of whether the tested window contains a signal.

Arithmetic counts include detector normalization; they exclude common frequency
correction from every model equally. The common correction is measured in the
`+CFO` latency columns. Measurements use warmed inference, resident device data,
and CUDA synchronization when relevant. They are device-specific and do not
predict radio acquisition latency or hardware power consumption.
