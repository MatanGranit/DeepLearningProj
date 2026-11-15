"""Neural PSS receivers: the 128-sample CNN and the convolutional scanning CNN.

* :class:`NarrowbandPssCnn` scores the 128-sample view of a frequency-corrected window
  with a conjugate-symmetric complex layer; trained by the known-bin experiment.
* :class:`ConvScanningPssCnn` consumes the window as received, frequency offset included,
  and scores every allowed offset at once with convolution layers (a global depthwise
  complex convolution, then dilated convolutions); trained by the unknown-bin experiment.

All keep a ``detection_threshold`` buffer that :class:`~utils.evaluation.Evaluator` fills
in from noise-only data; :meth:`PssDetector.predict` refuses to run without it.
"""
import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..data.pss_dataset import FREQ_OFFSETS
from ..data.pss_signal import FFT_SIZE, NARROWBAND_SIZE, NOISE_LABEL, NUM_PSS_IDS, PssSignalGenerator
from .receiver import NORM_FLOOR, Receiver

# Length the scanning network folds the window to. Every allowed offset is a multiple of
# FFT_SIZE / FOLD_LENGTH bins, so after the fold each offset is one bin of a FOLD_LENGTH-point DFT.
FOLD_LENGTH = 128


class PssDetector(Receiver):
    """Shared behaviour of the neural detectors: logits layout, calibration and decisions.

    ``forward`` returns logits of shape (batch, 4): columns 0..2 score the PSS
    identities and column 3 is a learned noise-only constant. The presence decision
    thresholds ``max(signal logits) - noise logit``; the identity is the best of the
    first three columns.
    """

    def __init__(self, fft_binsize: int) -> None:
        """
        Register the calibration buffer and validate the window length.

        :param fft_binsize: window length in complex samples; only FFT_SIZE is
            supported because the PSS generator is hard-wired to that size
        :raises ValueError: if fft_binsize is not FFT_SIZE
        """
        super().__init__()
        if fft_binsize != FFT_SIZE:
            raise ValueError(f"The current PSS generator requires fft_binsize={FFT_SIZE}")
        self.fft_binsize = fft_binsize
        # NaN marks "not yet calibrated" and is what predict() checks for. Being a
        # buffer, it travels with state_dict() into the saved checkpoint.
        self.register_buffer("detection_threshold", torch.tensor(float("nan")))

    def check_input(self, x: torch.Tensor) -> None:
        """
        Reject anything that is not a complex (batch, fft_binsize) tensor.

        :param x: the tensor handed to ``forward``
        :raises ValueError: on a wrong dtype or shape
        """
        if not torch.is_complex(x) or x.ndim != 2 or x.shape[1] != self.fft_binsize:
            raise ValueError(f"Expected complex input shaped (batch, {self.fft_binsize})")

    def l1_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate the noise level of each window by its L1 norm over all its samples.

        :param x: complex tensor of shape (batch, samples)
        :return: real tensor of shape (batch, 1), ``sum |Re| + |Im|``
        """
        return (x.real.abs() + x.imag.abs()).sum(dim=1, keepdim=True).clamp_min(NORM_FLOOR)

    @property
    def expected_l1(self) -> float:
        """Return the mean L1 norm of a unit-power noise-only window: sqrt(4/pi) per complex sample."""
        return self.expected_l1_of(self.fft_binsize)

    @staticmethod
    def expected_l1_of(samples: int) -> float:
        """
        Return the mean L1 norm of unit-power complex noise over a number of samples.

        :param samples: complex samples summed
        :return: ``samples * sqrt(4/pi)``
        """
        return samples * math.sqrt(4 / math.pi)

    def norm_operations(self, samples: Optional[int] = None) -> int:
        """
        Count the real operations of the L1 norm per window.

        ``|.|`` is a sign flip folded into the addition and is not counted; counting it
        would add another 2 * samples.

        :param samples: complex samples the norm runs over (default: the whole window)
        :return: one accumulation per real component of every sample
        """
        return 2 * (self.fft_binsize if samples is None else samples)

    def with_noise_logit(self, signal_logits: torch.Tensor) -> torch.Tensor:
        """
        Append the noise-only logit to the three signal logits.

        :param signal_logits: real tensor of shape (batch, 3)
        :return: real tensor of shape (batch, 4)
        """
        # The noise logit is a learned constant: only the signal logits respond to
        # the input, so the presence decision is scores-versus-constant.
        return torch.cat((signal_logits, self.noise_bias.expand(signal_logits.shape[0], 1)), dim=1)

    def presence_and_id(self, output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the presence score and identity from logits; see :meth:`Receiver.presence_and_id`.

        :param output: logits of shape (batch, 4)
        :return: (best signal logit minus the noise logit, argmax of the signal logits)
        """
        # Not logits.argmax(1): the identity is always the best of the first three
        # columns, and the noise logit only decides whether it is reported at all.
        return output[:, :NUM_PSS_IDS].max(dim=1).values - output[:, NOISE_LABEL], \
            output[:, :NUM_PSS_IDS].argmax(dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify windows at the calibrated false-alarm operating point.

        :param x: complex tensor of shape (batch, FFT_SIZE), frequency corrected unless
            ``RECEIVES_CORRECTED_WINDOW`` is False
        :return: long tensor of shape (batch,) holding 0..2 for a detected PSS
            identity or 3 for "no signal"
        :raises RuntimeError: if the detection threshold has not been calibrated
        """
        if not torch.isfinite(self.detection_threshold):
            raise RuntimeError("Calibrate the network on noise-only data before predict()")
        return self.decide(self(x), self.detection_threshold)


class NarrowbandPssCnn(PssDetector):
    """Unknown-phase detector on the 128-sample view of a frequency-corrected window.

    The PSS spectrum on those 128 bins is real (BPSK, +-1), so each template is
    conjugate symmetric in time: ``t[128 - n] = conj(t[n])``, with ``t[0]`` and ``t[64]``
    real. The complex layer keeps that symmetry as weight tying -- 65 real and 63
    imaginary weights per identity -- which lets every pair of samples ``n, 128 - n``
    share its pre-additions and halves the multiplies. The score is ``|z_k|^2 / L1^2``
    over the 128 samples.

    The weights start random and are learned from the data; training recovers the PSS
    templates. With the template weights (:meth:`set_template_weights`) ``z_k`` is the
    magnitude correlator's correlation.
    """

    def __init__(self) -> None:
        """Draw the tied weights at random, at unit norm per identity."""
        super().__init__(FFT_SIZE)
        half = NARROWBAND_SIZE // 2
        # The trained half: Re w[0..64] and Im w[1..63]; w[0] and w[64] are real.
        self.weight_real = nn.Parameter(torch.randn(NUM_PSS_IDS, half + 1))
        self.weight_imag = nn.Parameter(torch.randn(NUM_PSS_IDS, half - 1))
        with torch.no_grad():
            norms = self.weights().norm(dim=1, keepdim=True)
            self.weight_real.div_(norms)
            self.weight_imag.div_(norms)
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.noise_bias = nn.Parameter(torch.zeros(1))
        # Unit-norm weights give z_k unit noise power for unit-power samples, so this puts
        # the noise-only score near mean 1, the scale cross-entropy training needs.
        self.register_buffer("gain", torch.tensor(self.expected_l1_of(NARROWBAND_SIZE) ** 2))

    def set_template_weights(self) -> None:
        """Set the tied weights to the conjugate-symmetric 128-sample templates: the matched filter."""
        templates = PssSignalGenerator.narrowband(PssSignalGenerator.pss_templates())
        templates = templates / templates.norm(dim=1, keepdim=True)
        half = NARROWBAND_SIZE // 2
        with torch.no_grad():
            self.weight_real.copy_(templates.real[:, :half + 1])
            self.weight_imag.copy_(templates.imag[:, 1:half])

    def weights(self) -> torch.Tensor:
        """
        Build the full conjugate-symmetric weights from the trained half.

        :return: complex tensor of shape (NUM_PSS_IDS, NARROWBAND_SIZE)
        """
        zero = torch.zeros(NUM_PSS_IDS, 1, dtype=self.weight_imag.dtype, device=self.weight_imag.device)
        imag = torch.cat((zero, self.weight_imag, zero), dim=1)  # n = 0..64
        first = torch.complex(self.weight_real, imag)
        return torch.cat((first, first[:, 1:-1].flip(1).conj()), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Score a batch of frequency-corrected windows on their 128-sample view.

        :param x: complex tensor of shape (batch, FFT_SIZE), frequency corrected
        :return: real tensor of shape (batch, 4); columns 0..2 are PSS-identity scores
            and column 3 is the noise-only logit
        """
        self.check_input(x)
        y = PssSignalGenerator.narrowband(x)
        z = y @ self.weights().conj().T
        scale = self.gain * self.log_scale.exp()
        return self.with_noise_logit(scale * (z.real.square() + z.imag.square()) / self.l1_norm(y).square())

    def operation_counts(self) -> Tuple[int, int]:
        """
        Count the real arithmetic per window; see :meth:`Receiver.operation_counts`.

        Counted for the paired product the weight tying allows; ``forward`` computes the
        same ``z`` with a plain product. For a pair with ``y[n] = a + jb``,
        ``y[128-n] = e + jf`` and ``w[n] = c + jd``:
        ``z += c(a+e) + d(b-f) + j(c(b+f) + d(e-a))``, four pre-additions shared by the
        identities and four multiply-accumulates per identity. The two unpaired samples
        have real weights: two multiply-accumulates each. The 128-bin band selection is
        the receiver front end and is not counted. :class:`NarrowbandCorrelator` is
        counted without this pairing; its templates are just as symmetric, and paired it
        would need about 2,315 operations.

        :return: (MACs of the complex layer, everything else)
        """
        pairs = NARROWBAND_SIZE // 2 - 1
        macs = NUM_PSS_IDS * (4 * pairs + 2 * 2)
        other = 4 * pairs  # shared pre-additions
        other += NUM_PSS_IDS * 3  # |z|^2 per identity
        other += self.norm_operations(NARROWBAND_SIZE)
        # Squared norm, (gain * scale) / norm, three score multiplies.
        other += 1 + 1 + NUM_PSS_IDS
        return macs, other


class ConvScanningPssCnn(PssDetector):
    """Unknown-bin detector built from convolution layers: a global depthwise convolution, then dilated convolutions.

    The window arrives as received, frequency offset included, and is arranged as
    FOLD_LENGTH channels of FFT_SIZE / FOLD_LENGTH samples (channel r holds samples
    r, r + 128, r + 256, ...). Then, for each PSS identity t:

    1. **Global depthwise complex convolution** (GDConv, the learned layer): one 16-tap
       kernel per channel, as long as the channel, so each channel gives one output and
       no weight is shared between channels -- per channel it is a fully connected layer.
       Channel r gives ``u[r] = sum_q w[r + 128q] x[r + 128q]``, the window weighted by
       ``w`` and folded to 128 samples. ``w`` starts at ``conj(t)`` and is tied to the
       template's conjugate symmetry, ``w[2048 - n] = conj(w[n])``, so half of it is trained.
    2. **Seven dilated convolutions** over the 128 samples, kernel 2 with fixed taps
       ``[1, 1]`` and ``[1, -1]`` shared by every position, dilations ``h`` = 64, 32, ..., 1.
       The difference at position ``i`` of a block is then scaled by the fixed twiddle
       ``exp(-2j pi i / (2h))``, which varies with ``i``. Together: the radix-2 butterflies
       of a 128-point DFT, whose bin ``k / 16`` is the correlation at offset ``k``.
    3. **Max pooling** of ``|z|^2`` over the allowed offsets, divided by L1^2.

    With its initial weights ``z`` at offset ``k`` is exactly the magnitude correlator's
    correlation with the template shifted by ``k`` bins.
    """

    RECEIVES_CORRECTED_WINDOW = False
    REPORTS_OFFSET = True

    def __init__(self, fft_binsize: int = FFT_SIZE, freq_offsets: Sequence[int] = FREQ_OFFSETS) -> None:
        """
        Set the depthwise kernels to the conjugate templates and build the fixed dilated layers.

        :param fft_binsize: window length in complex samples; only FFT_SIZE is supported
        :param freq_offsets: offsets to search, in FFT bins (default: every offset the dataset uses)
        :raises ValueError: on an unsupported fft_binsize, or an offset that is not a
            multiple of fft_binsize / FOLD_LENGTH
        """
        super().__init__(fft_binsize)
        step = fft_binsize // FOLD_LENGTH
        if any(offset % step for offset in freq_offsets):
            raise ValueError(f"Every offset must be a multiple of {step} bins, got {list(freq_offsets)}")
        self.stages = int(math.log2(FOLD_LENGTH))
        bins = torch.tensor([offset // step for offset in freq_offsets])
        # The butterflies leave the DFT in bit-reversed order.
        reverse = torch.tensor([int(format(k, f"0{self.stages}b")[::-1], 2) for k in range(FOLD_LENGTH)])
        self.register_buffer("output_index", reverse[bins])
        # The searched offsets in FFT bins, in the order of the max-pool's positions.
        self.register_buffer("freq_offsets", torch.tensor(list(freq_offsets)))
        self.register_buffer("difference_kernel", torch.tensor([1.0, -1.0]))
        self.register_buffer("sum_kernel", torch.tensor([1.0, 1.0]))
        # Every stage's twiddle exp(-2j pi i / (2h)) is one of the 64 powers of exp(-2j pi / 128).
        table = torch.exp(-2j * math.pi * torch.arange(FOLD_LENGTH // 2) / FOLD_LENGTH)
        self.register_buffer("twiddle_table", table.to(torch.complex64))
        half = fft_binsize // 2
        templates = PssSignalGenerator.pss_templates(normalized=True).conj()
        # The trained half: Re w[0..1024] and Im w[1..1023]; w[0] and w[1024] are real.
        self.weight_real = nn.Parameter(templates.real[:, :half + 1].clone())
        self.weight_imag = nn.Parameter(templates.imag[:, 1:half].clone())
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.noise_bias = nn.Parameter(torch.zeros(1))
        # Unit-norm kernels give z_k unit noise power, so this puts the noise-only score near mean 1.
        self.register_buffer("gain", torch.tensor(self.expected_l1 ** 2))

    def kernels(self) -> torch.Tensor:
        """
        Build the full conjugate-symmetric weights from the trained half.

        :return: complex tensor of shape (NUM_PSS_IDS, FFT_SIZE)
        """
        zero = torch.zeros(NUM_PSS_IDS, 1, dtype=self.weight_imag.dtype, device=self.weight_imag.device)
        first = torch.complex(self.weight_real, torch.cat((zero, self.weight_imag, zero), dim=1))
        return torch.cat((first, first[:, 1:-1].flip(1).conj()), dim=1)

    def depthwise_convolution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the global depthwise complex convolution to the window arranged as FOLD_LENGTH channels.

        :param x: complex tensor of shape (batch, FFT_SIZE)
        :return: complex tensor of shape (batch, NUM_PSS_IDS, FOLD_LENGTH)
        """
        blocks = self.fft_binsize // FOLD_LENGTH
        channels = x.reshape(x.shape[0], blocks, FOLD_LENGTH).transpose(1, 2)  # (batch, 128, 16)
        # Grouped by channel: output channel 3r + identity holds identity's 16-tap kernel for channel r.
        kernels = self.kernels().reshape(NUM_PSS_IDS, blocks, FOLD_LENGTH).permute(2, 0, 1)  # (128, 3, 16)
        weight = kernels.reshape(FOLD_LENGTH * NUM_PSS_IDS, 1, blocks)
        out = nn.functional.conv1d(channels, weight, groups=FOLD_LENGTH)  # (batch, 128 * 3, 1)
        return out.reshape(x.shape[0], FOLD_LENGTH, NUM_PSS_IDS).transpose(1, 2)

    def dilated_convolutions(self, u: torch.Tensor) -> torch.Tensor:
        """
        Apply the seven fixed dilated convolutions (the 128-point DFT butterflies).

        :param u: complex tensor of shape (batch, NUM_PSS_IDS, FOLD_LENGTH)
        :return: complex tensor of the same shape, the DFT in bit-reversed order
        """
        batch = u.shape[0]
        for stage in range(self.stages):
            span = FOLD_LENGTH >> (stage + 1)
            twiddle = self.twiddle_table[::FOLD_LENGTH // (2 * span)]  # exp(-2j pi i / (2 span)), i < span
            # Every block of 2*span samples is one sequence; kernel 2 at dilation `span`.
            blocks = torch.view_as_real(u.reshape(-1, 1, 2 * span)).permute(0, 3, 1, 2).reshape(-1, 1, 2 * span)
            pair = torch.cat((self.sum_kernel, self.difference_kernel)).to(blocks.dtype).reshape(2, 1, 2)
            out = nn.functional.conv1d(blocks, pair, dilation=span)  # (blocks*2, 2, span): sum, difference
            out = torch.view_as_complex(out.reshape(-1, 2, 2, span).permute(0, 2, 3, 1).contiguous())
            u = torch.cat((out[:, 0], out[:, 1] * twiddle), dim=1).reshape(batch, NUM_PSS_IDS, -1)
        return u

    def offset_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Score every identity at every searched offset: the input of the max-pool.

        :param x: complex tensor of shape (batch, FFT_SIZE), frequency offset not removed
        :return: real tensor of shape (batch, NUM_PSS_IDS, offsets), ``|z|^2`` times the
            score scale over L1^2
        """
        self.check_input(x)
        z = self.dilated_convolutions(self.depthwise_convolution(x))[..., self.output_index]
        scale = self.gain * self.log_scale.exp()
        return scale * (z.real.square() + z.imag.square()) / self.l1_norm(x).square()[..., None]

    def forward_with_offsets(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Score a batch of windows and report, per identity, the offset the max-pool picked.

        :param x: complex tensor of shape (batch, FFT_SIZE), frequency offset not removed
        :return: (logits as :meth:`forward` gives them, long tensor of shape
            (batch, NUM_PSS_IDS) holding each identity's best offset in FFT bins)
        """
        pooled, position = self.offset_scores(x).max(dim=2)
        return self.with_noise_logit(pooled), self.freq_offsets[position]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Score a batch of windows as received, over every allowed offset.

        :param x: complex tensor of shape (batch, FFT_SIZE), frequency offset not removed
        :return: real tensor of shape (batch, 4); columns 0..2 are PSS-identity scores
            (max-pooled over the offsets) and column 3 is the noise-only logit
        """
        return self.forward_with_offsets(x)[0]

    def predict_with_offset(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classify windows at the calibrated operating point and report where each PSS was found.

        The offset is the max-pool's argmax for the reported identity: the position on the
        frequency axis whose correlation won. It costs only comparisons.

        :param x: complex tensor of shape (batch, FFT_SIZE), frequency offset not removed
        :return: (labels as :meth:`PssDetector.predict` gives them, offsets in FFT bins;
            the offset is -1 where no PSS is reported)
        :raises RuntimeError: if the detection threshold has not been calibrated
        """
        if not torch.isfinite(self.detection_threshold):
            raise RuntimeError("Calibrate the network on noise-only data before predict_with_offset()")
        with torch.no_grad():
            logits, best_offsets = self.forward_with_offsets(x)
        labels = self.decide(logits, self.detection_threshold)
        return labels, self.reported_offsets(labels, best_offsets)

    @staticmethod
    def reported_offsets(labels: torch.Tensor, best_offsets: torch.Tensor) -> torch.Tensor:
        """
        Pick the offset of each reported identity.

        :param labels: four-class decisions, shape (batch,)
        :param best_offsets: each identity's best offset in FFT bins, shape (batch, NUM_PSS_IDS)
        :return: long tensor of shape (batch,): the offset, or -1 where no PSS is reported
        """
        offsets = best_offsets.gather(1, labels.clamp(max=NUM_PSS_IDS - 1)[:, None])[:, 0]
        return torch.where(labels == NOISE_LABEL, torch.full_like(offsets, -1), offsets)

    def butterfly_operations(self) -> int:
        """
        Count the real operations of the dilated layers for one identity, pruned to the output bins used.

        Each node computed costs one complex add or subtract (2). A difference is then
        scaled by its twiddle: free for 1, -1, j or -j; 4 for an odd power of
        ``exp(-j pi / 4)``, since ``(a + jb)(1 - j) / sqrt(2)`` is two adds and two
        multiplies; 6 (a complex multiply) otherwise. Nodes that feed no searched offset
        are skipped.

        :return: additions plus multiplications
        """
        needed = {int(i) for i in self.output_index}
        operations = 0
        for stage in reversed(range(self.stages)):
            span = FOLD_LENGTH >> (stage + 1)
            inputs = set()
            for node in needed:
                block, position = divmod(node, 2 * span)
                operations += 2
                # The twiddle is exp(-2j pi * eighths / 8); its eighth-turns decide the cost.
                eighths = (position - span) * 8 / (2 * span)
                if position >= span and eighths % 2:
                    operations += 4 if eighths == int(eighths) else 6
                inputs |= {block * 2 * span + position % span, block * 2 * span + position % span + span}
            needed = inputs
        return operations

    def operation_counts(self) -> Tuple[int, int]:
        """
        Count the real arithmetic per window; see :meth:`Receiver.operation_counts`.

        The depthwise convolution uses the three-multiply complex product with the stored
        sums ``Re w + Im w`` and ``Im w - Re w``: ``Re x + Im x`` once per sample, then per
        identity three multiplies and two adds per sample, and the 16-tap accumulation
        (two real adds per tap after the first).

        :return: (0, every operation)
        """
        n, offsets = self.fft_binsize, len(self.output_index)
        per_identity = 5 * n + 2 * (n - FOLD_LENGTH) + self.butterfly_operations()
        other = n + NUM_PSS_IDS * per_identity
        other += NUM_PSS_IDS * offsets * 3  # |z|^2 per identity and offset
        other += self.norm_operations()
        # Squared norm, (gain * scale) / norm, three score multiplies.
        other += 1 + 1 + NUM_PSS_IDS
        return 0, other

    def stored_bytes(self) -> int:
        """
        Count the bytes stored when deployed with the three-multiply product of :meth:`operation_counts`.

        That product uses ``Re w``, ``Re w + Im w`` and ``Im w - Re w``; with tied weights
        their second halves are mirrors of the first (``Re w[2048-n] = Re w[n]``,
        ``(Re w + Im w)[2048-n] = -(Im w - Re w)[n]``), so 3 x 1025 real values per
        identity are stored. The 64 distinct twiddles and the buffers are stored too.

        :return: bytes
        """
        weight = self.weight_real.element_size()
        tied_arrays = 3 * NUM_PSS_IDS * (self.fft_binsize // 2 + 1)
        trained = self.weight_real.numel() + self.weight_imag.numel()
        # The parameters are replaced by the three tied arrays; buffers (twiddles included) stay as they are.
        return super().stored_bytes() + weight * (tied_arrays - trained)
