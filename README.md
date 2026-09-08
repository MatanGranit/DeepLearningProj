# PSS Detection Project

This project compares an efficient CNN with a classic correlator for four-class
PSS detection:

- Classes 0, 1 and 2 are the three PSS identities.
- Class 3 is noise only (no PSS present).

The dataset uses the average time-domain definition
`SNR = E[|signal|^2] / E[|noise|^2]` and generates circular complex AWGN. The
CNN is trained once on samples pooled across the configured SNR sweep.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Generate the publication figure

Run from the project root:

```bash
python utils/GeneratePublicationPlots.py
```

This regenerates the PDF, SVG and PNG result figures. The new publication
figure contains four separate panels:

- Four-class accuracy.
- Detection probability `Pd = P(declare signal | PSS is present)`.
- False-alarm probability `Pfa = P(declare signal | noise only)`.
- Approximate real-FLOP complexity.

The correlator threshold is calibrated on an independent noise-only validation
set for a target Pfa of 1%. Test samples are generated only after calibration,
so they are not used to select the threshold. Existing figures in the archive
predate the corrected evaluation and should not be used until the command above
has been run.

## How to understand the graphs

No previous knowledge of mobile communication or neural networks is needed for
this section.

### What is the system trying to do?

A received example contains either one of three possible PSS signals or noise
alone. A PSS is a short, known pattern transmitted by a cellular base station.
The receiver uses it to recognize that a cell is present and to identify which
of the three PSS patterns was transmitted.

The experiment compares two receivers. The **classic correlator** compares the
received samples directly with each known PSS pattern. The **efficient CNN** is
a small neural network that learns how to distinguish the same four cases from
training examples.

### What does SNR mean?

SNR, or signal-to-noise ratio, describes how strong the useful signal is
compared with the noise. It is measured in decibels (dB).

- A larger SNR means that the signal is easier to observe.
- A more negative SNR means that the signal is more deeply hidden in noise.
- Moving from -30 dB to -20 dB means that the signal-to-noise power ratio has
  increased by a factor of ten.

Very low input SNR does not make correlation useless. The correlator combines
2048 correctly aligned samples. The repeated signal pattern adds coherently,
while random noise does not add in the same organized way. This is called
**processing gain**. In this ideal experiment, combining 2048 samples provides
approximately `10*log10(2048) = 33.1 dB` of processing gain. Therefore, some
performance above chance near -30 dB is reasonable. It does not mean that the
original -30 dB signal was clearly visible before correlation.

### Panel (a): four-class accuracy

Accuracy is the percentage of examples assigned to the correct class. Because
there are four equally likely classes, random guessing gives an average
accuracy of 25%, shown by the dotted baseline.

At the lowest SNR values, both methods should approach 25% because the received
samples contain too little useful information. As SNR increases, the curves
rise because the PSS becomes easier to detect and identify. In the intermediate
SNR region, the CNN may rise earlier than the correlator, meaning that it
correctly handles more difficult noisy examples. At sufficiently high SNR,
both methods approach 100%, so there is little room for either method to
improve over the other.

Accuracy combines two different questions: whether a signal was detected and
whether its PSS identity was correct. This is why panels (b) and (c) are also
needed.

### Panel (b): detection probability, Pd

`Pd` answers the question: **when a PSS signal is actually present, how often
does the receiver declare that a signal is present?** Higher is better, and
100% means that no present signal was missed.

At low SNR, Pd decreases because real PSS examples begin to look like noise.
At high SNR, Pd should approach 100%. Pd does not require the receiver to choose
the correct PSS identity; it measures only signal presence. Therefore, Pd can
be higher than the four-class accuracy.

### Panel (c): false-alarm probability, Pfa

`Pfa` answers the opposite question: **when the input is noise only, how often
does the receiver incorrectly claim that a PSS signal is present?** Lower is
better, and 0% would mean no false alarms.

There is normally a trade-off between Pd and Pfa. A receiver that declares a
signal very easily may detect more weak signals, but it will also produce more
false alarms. A stricter decision threshold reduces false alarms but can miss
more real signals.

The correlator threshold is selected using a separate noise-only validation set
with a target Pfa of 1%. The dotted line in panel (c) shows that target. Small
differences around 1% are expected because each plotted point uses a finite
number of random test examples. The CNN is not forced to have the same Pfa, so
its Pd should not be compared with the correlator's Pd without also examining
its Pfa.

### Panel (d): computational complexity

This panel estimates how much arithmetic each method performs for one input.
The unit is approximate real floating-point operations (FLOPs). A shorter bar
means fewer arithmetic operations and potentially easier implementation on
resource-limited hardware.

The FLOP count is an analytical complexity comparison, not a direct measurement
of execution time, memory use, power consumption, or hardware latency. Those
quantities depend on the processor and software implementation and would need
separate measurements.

### Main conclusion

The useful comparison is not simply which method has the highest accuracy at
one SNR. A good detector should maintain high Pd, low Pfa, correct PSS identity,
and low computational cost. The CNN is most useful if it improves performance
in the difficult intermediate-SNR region while keeping Pfa acceptably low and
using fewer operations. At very high SNR, both methods already work well; at
extremely low SNR, neither method can recover information that is effectively
lost in noise.

## Experimental assumptions

The reported experiment assumes:

- Perfect PSS timing and one aligned 2048-sample observation window.
- Exact frequency-offset knowledge supplied to both methods.
- An AWGN-only channel, without multipath fading or synchronization error.
- Three known PSS candidates and a fourth noise-only class.

Consequently, the curves measure detection and PSS identification after ideal
synchronization. They do not represent a blind search over timing and frequency.

## Run the focused tests

```bash
python -m unittest discover -s tests -v
```

The fallback correlator threshold is defined in `utils/correlator.py`. The
publication script replaces it with the independently calibrated threshold.
