import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim

if __package__:
    from .data_loader import generate_pss
else:
    from data_loader import generate_pss


class EfficientPssCnn(nn.Module):
    """Template-initialized, factorized coherent detector for aligned PSS.

    A block projection is equivalent to Conv1d(2, 4, 16, stride=16), followed
    by a global linear classifier. Keeping the path linear preserves coherent
    integration across the entire window. This is a model-based neural network,
    explicitly using the known templates and the generator's fixed phase.
    It is not a blind timing/CFO or unknown-phase receiver.
    """

    def __init__(self, fft_binsize=2048, rank=4, template_init=True):
        super().__init__()
        if fft_binsize != 2048:
            raise ValueError("The current PSS generator requires fft_binsize=2048")
        if not 1 <= rank <= 32:
            raise ValueError("rank must be between 1 and 32")
        self.fft_binsize = fft_binsize
        self.rank = rank
        self.block_size = 16
        self.local_projection = nn.Linear(32, rank, bias=False)
        self.global_projection = nn.Linear(128 * rank, 3, bias=False)
        self.class_bias = nn.Parameter(torch.full((3,), -2.5))
        self.noise_bias = nn.Parameter(torch.zeros(1))
        self.register_buffer("detection_threshold", torch.tensor(float("nan")))
        if template_init:
            # SVD uses the three known noise-free waveforms, never test data.
            with torch.no_grad():
                templates = torch.stack([generate_pss(i) for i in range(3)])
                real = torch.view_as_real(templates).reshape(3, -1)
                real = real / real.norm(dim=1, keepdim=True)
                blocks = real.reshape(3, 128, 32)
                _, _, vh = torch.linalg.svd(blocks.reshape(-1, 32), full_matrices=False)
                basis = vh[:rank]
                coefficients = blocks @ basis.T
                self.local_projection.weight.copy_(basis)
                self.global_projection.weight.copy_(coefficients.reshape(3, -1))

    def forward(self, x):
        if not torch.is_complex(x) or x.ndim != 2 or x.shape[1] != self.fft_binsize:
            raise ValueError("Expected complex input shaped (batch, 2048)")
        real = torch.view_as_real(x).reshape(x.shape[0], -1)
        projected = self.local_projection(real.reshape(-1, 128, 32))
        scores = self.global_projection(projected.flatten(1))
        # Normalize only three output scores rather than all 4096 input values.
        rms = real.square().mean(dim=1, keepdim=True).clamp_min(1e-20).sqrt()
        signal_logits = scores / rms + self.class_bias
        return torch.cat((signal_logits, self.noise_bias.expand(x.shape[0], 1)), dim=1)

    def predict(self, x):
        if not torch.isfinite(self.detection_threshold):
            raise RuntimeError("Calibrate the network on noise-only data before predict()")
        return predict_from_logits(self(x), self.detection_threshold)


class PhaseInvariantPssCnn(nn.Module):
    """Small complex CNN for aligned PSS with an unknown constant phase.

    The first layer is a shared non-overlapping complex convolution. A complex
    global projection integrates all blocks coherently, and the final modulus
    removes the unknown common phase. Consequently, for every theta,
    ``forward(x * exp(1j*theta)) == forward(x)`` up to roundoff.

    The default block_size=4, rank=1 design is validation-selected under a
    lower-operation constraint. It is initialized from a complex SVD of the
    three clean PSS templates and then trained end to end.
    """

    def __init__(self, fft_binsize=2048, block_size=4, rank=1,
                 template_init=True):
        super().__init__()
        if fft_binsize != 2048:
            raise ValueError("The current PSS generator requires fft_binsize=2048")
        if fft_binsize % block_size:
            raise ValueError("block_size must divide fft_binsize")
        if not 1 <= rank <= block_size:
            raise ValueError("rank must be between 1 and block_size")
        self.fft_binsize = fft_binsize
        self.block_size = block_size
        self.rank = rank
        self.positions = fft_binsize // block_size

        # Real tensors make the number of real trainable degrees of freedom
        # explicit while implementing a tied complex convolution.
        self.local_real = nn.Parameter(torch.empty(rank, block_size))
        self.local_imag = nn.Parameter(torch.empty(rank, block_size))
        self.global_real = nn.Parameter(torch.empty(3, self.positions * rank))
        self.global_imag = nn.Parameter(torch.empty(3, self.positions * rank))
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        # For normalized complex Gaussian noise, N*|corr|^2 is roughly unit
        # exponential and log1p(.) is near log(2). Subtracting log(2) makes
        # the four logits approximately neutral at uninformative SNRs.
        self.class_bias = nn.Parameter(torch.full((3,), -math.log(2.0)))
        self.noise_bias = nn.Parameter(torch.zeros(1))
        self.register_buffer("detection_threshold", torch.tensor(float("nan")))
        self.reset_parameters(template_init)

    def reset_parameters(self, template_init=True):
        if not template_init:
            nn.init.orthogonal_(self.local_real)
            nn.init.zeros_(self.local_imag)
            nn.init.normal_(self.global_real, std=.02)
            nn.init.normal_(self.global_imag, std=.02)
            return
        with torch.no_grad():
            templates = torch.stack([generate_pss(i) for i in range(3)])
            templates = templates / templates.norm(dim=1, keepdim=True)
            blocks = templates.reshape(3, self.positions, self.block_size)
            _, _, vh = torch.linalg.svd(
                blocks.reshape(-1, self.block_size), full_matrices=False
            )
            basis = vh[:self.rank].conj()
            coefficients = blocks @ basis.conj().T
            self.local_real.copy_(basis.real)
            self.local_imag.copy_(basis.imag)
            flattened = coefficients.reshape(3, -1)
            self.global_real.copy_(flattened.real)
            self.global_imag.copy_(flattened.imag)

    def complex_features(self, x):
        if not torch.is_complex(x) or x.ndim != 2 or x.shape[1] != self.fft_binsize:
            raise ValueError("Expected complex input shaped (batch, 2048)")
        blocks = x.reshape(x.shape[0], self.positions, self.block_size)
        # f = block @ conj(kernel): a tied complex Conv1d with
        # kernel_size=stride=block_size and no bias.
        real = (blocks.real @ self.local_real.T
                + blocks.imag @ self.local_imag.T)
        imag = (blocks.imag @ self.local_real.T
                - blocks.real @ self.local_imag.T)
        return real.flatten(1), imag.flatten(1)

    def forward(self, x):
        feature_real, feature_imag = self.complex_features(x)
        # z = feature @ conj(global_filter), three complex class scores.
        score_real = (feature_real @ self.global_real.T
                      + feature_imag @ self.global_imag.T)
        score_imag = (feature_imag @ self.global_real.T
                      - feature_real @ self.global_imag.T)
        correlation_energy = score_real.square() + score_imag.square()
        real_input = torch.view_as_real(x).reshape(x.shape[0], -1)
        norm_squared = real_input.square().sum(dim=1, keepdim=True).clamp_min(1e-20)
        normalized_energy = self.fft_binsize * correlation_energy / norm_squared
        signal_logits = (self.log_scale.exp() * normalized_energy.log1p()
                         + self.class_bias)
        return torch.cat((signal_logits, self.noise_bias.expand(x.shape[0], 1)), dim=1)

    def predict(self, x):
        if not torch.isfinite(self.detection_threshold):
            raise RuntimeError("Calibrate the network on noise-only data before predict()")
        return predict_from_logits(self(x), self.detection_threshold)


def detection_score(logits):
    """Presence score; separate the noise decision from PSS-ID selection."""
    return logits[:, :3].max(dim=1).values - logits[:, 3]


def predict_from_logits(logits, threshold=0.0):
    predicted_id = logits[:, :3].argmax(dim=1)
    return torch.where(detection_score(logits) > threshold, predicted_id, 3)


def correct_frequency_offset(samples, freq_offset, n):
    freq_offset = freq_offset.to(samples.device, dtype=torch.float32)
    n = n.to(samples.device, dtype=torch.float32)
    exponent = torch.exp(
        2j * math.pi * torch.outer(freq_offset, n / samples.shape[1])
    )
    return samples * torch.conj(exponent)


def train_model(model, dataloader, epochs=8, validation_loader=None, lr=None):
    if epochs < 1 or len(dataloader) < 1:
        raise ValueError("Training requires positive epochs and a nonempty loader")
    criterion = nn.CrossEntropyLoss()
    learning_rate = lr if lr is not None else 1e-4
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    device = next(model.parameters()).device
    n = torch.arange(2048, device=device)

    print("Starting training phase...")
    history = []
    best_loss = float("inf")
    best_state = None
    if hasattr(model, "detection_threshold"):
        model.detection_threshold.fill_(float("nan"))

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for samples, labels, freq_offset in dataloader:
            samples = samples.to(device)
            labels = labels.to(device)
            freq_offset = freq_offset.to(device)

            optimizer.zero_grad(set_to_none=True)

            samples_corrected = correct_frequency_offset(samples, freq_offset, n)

            outputs = model(samples_corrected)

            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

            predicted = torch.argmax(outputs, dim=1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_acc = 100 * correct_predictions / total_samples
        val_loss = None
        if validation_loader is not None:
            model.eval()
            val_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for samples, labels, offsets in validation_loader:
                    samples, labels = samples.to(device), labels.to(device)
                    logits = model(correct_frequency_offset(samples, offsets, n))
                    val_sum += criterion(logits, labels).item() * labels.numel()
                    val_count += labels.numel()
            val_loss = val_sum / val_count
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch + 1, "loss": epoch_loss,
                        "accuracy": epoch_acc, "validation_loss": val_loss})
        scheduler.step()

        print(
            f"Epoch {epoch + 1} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Train Accuracy: {epoch_acc:.2f}% | "
            f"Validation loss: {val_loss if val_loss is not None else 'n/a'} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    return history
