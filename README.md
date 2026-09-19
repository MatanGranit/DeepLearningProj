# Compact PSS detection

This project detects one of three known Primary Synchronization Signal (PSS)
sequences, or declares that no PSS is present. Classes 0, 1 and 2 identify the
PSS sequence; class 3 is noise only. Every observation contains 2,048 complex
time samples, and the four classes are sampled with equal probability.

The main experiment studies a compact, phase-invariant complex CNN under two
conditions:

1. **Random phase:** every signal window has an independent phase sampled
   uniformly from 0 to 2π.
2. **Fixed phase:** every signal has phase zero.

The same trained CNN checkpoints are evaluated in both conditions. The new
comparison contains only the phase-invariant CNN, a magnitude correlator and a
coherent correlator.

## Model 1: original factorized detector

`EfficientPssCnn` is the neural receiver used in the preserved fixed-phase
experiment under `results/`. It receives a complex tensor with shape
`(batch, 2048)` and processes it as follows:

1. Split the observation into 128 non-overlapping blocks of 16 complex
   samples.
2. Represent each block by 32 real values: 16 real parts and 16 imaginary
   parts.
3. Apply the same learned projection from 32 values to four features to every
   block.
4. Flatten the resulting 512 features.
5. Apply a global learned projection to obtain three PSS scores.
6. Add learned biases for the three signal classes and the noise-only class.

The shared local projection is mathematically equivalent to
`Conv1d(2, 4, kernel_size=16, stride=16)`. However, the model has no nonlinear
activation between its local and global projections. The composition of those
two layers is therefore still a single linear transformation.

### Original-detector parameters

| Component | Shape | Trainable real parameters |
|---|---:|---:|
| Shared local projection | 32 × 4 | 128 |
| Global projection | 3 × 512 | 1,536 |
| Signal biases | 3 | 3 |
| Noise bias | 1 | 1 |
| **Total** |  | **1,668** |

The three signal scores are divided by the input RMS. This keeps their scale
stable across different noise powers without normalizing every input value
separately. It does not remove noise or increase the physical SNR.

### Initialization and training

The two projections are initialized from a rank-four SVD approximation of the
three analytically generated clean PSS templates. This initialization retains
about 98.9% of the templates' combined energy in the block representation. It
uses no received validation or test examples, and all initialized weights
remain trainable.

Training uses cross-entropy loss, AdamW, cosine learning-rate decay, gradient
clipping and restoration of the checkpoint with the lowest validation loss.
The default learning rate is `1e-4`.

## Model 2: phase-invariant complex CNN

`PhaseInvariantPssCnn` receives a complex tensor with shape `(batch, 2048)`.
Its signal path is:

1. Split the window into 512 non-overlapping blocks of four complex samples.
2. Apply one shared rank-one complex convolution to every block.
3. Combine the 512 complex features using a learned complex global projection
   for each of the three PSS identities.
4. Convert each complex score to energy using `real² + imag²`.
5. Normalize by the energy of the input window and produce four logits: three
   PSS identities and noise.

For a global phase rotation `x -> x * exp(j*theta)`, the complex score rotates
but its energy does not:

```text
|exp(j*theta) * z|² = |z|²
```

Consequently, the network output is invariant to an arbitrary common phase
rotation by construction, up to floating-point roundoff. It does not need to
enumerate a bank of discrete phase hypotheses.

### Architecture hyperparameters

| Hyperparameter | Value |
|---|---:|
| Input length | 2,048 complex samples |
| Convolution kernel | 4 complex samples |
| Convolution stride | 4 |
| Complex feature rank | 1 |
| Feature positions | 512 |
| Signal classes | 3 |
| Total classes | 4 |
| Initialization | Complex SVD of the three clean PSS templates |
| Nonlinearity | Normalized energy followed by `log1p` |

The 3,085 trainable real parameters consist of eight local-convolution
parameters, 3,072 global-projection parameters, one learned scale, three signal
biases and one noise bias. The calibrated detection threshold is stored as a
buffer and is not a trainable parameter.

## Training and evaluation configuration

| Setting | Value |
|---|---:|
| Loss | Cross-entropy |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Epochs | 8 |
| Batch size | 128 |
| Learning-rate schedule | Cosine annealing |
| Gradient clipping | 5.0 |
| Training seeds | 91, 92, 93 |
| Training samples per SNR | 3,000 |
| Validation samples per SNR | 400 |
| Calibration noise windows per SNR | 10,000 |
| Test windows per SNR and phase condition | 8,000 |
| False-alarm target | 1% |

The SNR sweep is:

```text
-50, -45, -40, -38, -36, -34, -32, -30,
-28, -26, -24, -22, -20, -10, 0 dB
```

Training, validation, calibration and test streams are disjoint. The checkpoint
with the lowest validation loss is restored after training. Architecture
selection was also performed on development data before final testing, under a
44,039-operation budget.

## Classical receivers

The **magnitude correlator** computes the magnitude of the normalized complex
correlation with each PSS template. It is invariant to unknown phase and is the
main classical baseline for the random-phase experiment.

The **coherent correlator** uses the signed, phase-aligned correlation. It is a
valid and efficient matched-filter baseline only when the signal phase is fixed
and aligned with the template. In the random-phase plots it is intentionally
mismatched; it is included to demonstrate the consequence of applying a
phase-sensitive receiver without phase knowledge.

## How to understand the graphs

The horizontal axis is the **input time-sample SNR**, before detection
processing. Noise is added before every receiver. At -30 dB, the average noise
power per sample is 1,000 times the average signal power per sample.

An ideal matched filter coherently integrates 2,048 samples. Under this
project's SNR definition, its ideal processing gain is approximately:

```text
10*log10(2048) = 33.1 dB
```

For example, -20 dB input SNR corresponds to approximately 13.1 dB ideal
matched-filter output SNR. The graphs retain the input SNR because it is the
common channel condition presented to all receivers. A single output-SNR axis
would not be meaningful for the nonlinear magnitude receiver and CNN.

The reported metrics are:

- **Four-class accuracy:** correct PSS identity or correct noise declaration.
  Uniform guessing gives 25% expected accuracy.
- **Pd:** probability of declaring that a signal is present when a signal is
  present. A wrong PSS identity still counts as detection, so Pd must be read
  together with four-class accuracy.
- **Pfa:** probability of declaring a signal on a noise-only window. Thresholds
  target 1% Pfa and are frozen before testing.
- **Approximate operations:** detector arithmetic only. The exact CFO
  correction common to every receiver is excluded equally.

## Run the project

Python 3.10 or newer is required.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Regenerate the supplied phase-comparison figures without retraining:

```bash
python utils/GeneratePhaseComparisonPlots.py \
  --from-results results_phase_comparison/phase_comparison_results.json
```

Run a new three-seed training and evaluation experiment:

```bash
python run_phase_comparison.py --output results_phase_comparison/my_run
```

A smaller installation check is:

```bash
python run_phase_comparison.py \
  --seeds 10 \
  --epochs 2 \
  --train-per-snr 200 \
  --validation-per-snr 50 \
  --test-samples 1000 \
  --calibration-samples 3000 \
  --output results_phase_comparison/smoke
```

## Use a trained CNN checkpoint

```python
import torch
from utils.net import PhaseInvariantPssCnn, correct_frequency_offset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(
    "results_phase_comparison/phase_cnn_seed91.pth",
    map_location=device,
    weights_only=True,
)

model = PhaseInvariantPssCnn(**checkpoint["model_kwargs"]).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

# samples: complex tensor [batch, 2048]
# offsets: exact frequency offsets [batch]
# n = torch.arange(2048, device=device)
# corrected = correct_frequency_offset(samples.to(device), offsets.to(device), n)
# with torch.inference_mode():
#     labels = model.predict(corrected)  # 0, 1, 2 or 3
```

`forward()` returns training logits. `predict()` applies the independently
calibrated detection threshold saved in the checkpoint. Applying `argmax`
directly is not equivalent to the reported 1% false-alarm operating point.

## Project files

- `run_phase_comparison.py`: main fixed/random-phase experiment entry point.
- `utils/net.py`: phase-invariant CNN, legacy detector and training code.
- `utils/data_loader.py`: PSS generation, AWGN and optional random phase.
- `utils/correlator.py`: magnitude and coherent classical receivers.
- `utils/evaluation.py`: calibration, confusion matrices, Pd, Pfa and timing.
- `utils/phase_benchmark.py`: model selection and reproducible benchmark.
- `utils/GeneratePhaseComparisonPlots.py`: phase-comparison figures and report.
- `tests/test_random_phase.py`: exact phase-invariance and cost checks.
- `tests/test_signal_pipeline.py`: signal-generation and inference checks.
- `results_phase_comparison/`: main checkpoints, measurements and figures.
- `results/`: preserved historical fixed-phase experiment.

## Assumptions and limitations

The main phase-comparison experiment assumes:

- aligned 2,048-sample observation windows;
- exact frequency-offset knowledge and correction;
- AWGN only;
- one constant phase across each PSS window;
- no multipath, fading, timing search or blind CFO search;
- representative noise-only data for threshold calibration.

The random-phase experiment removes the fixed-phase assumption but does not
make the task fully blind. In particular, timing and CFO remain exact.

