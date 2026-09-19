"""Independent threshold calibration, detection metrics, and latency measurements."""
import math
import time

import torch

if __package__:
    from .data_loader import generate_noise_vector, generate_pss
    from .net import correct_frequency_offset, detection_score, predict_from_logits
    from .correlator import Correlator
else:
    from data_loader import generate_noise_vector, generate_pss
    from net import correct_frequency_offset, detection_score, predict_from_logits
    from correlator import Correlator


def score_and_id(model, corrected):
    output = model(corrected)
    if isinstance(model, Correlator):
        return output.max(dim=1)
    return detection_score(output), output[:, :3].argmax(dim=1)


def calibrate_thresholds(models, snr_db=-30., num_samples=10000,
                         target_pfa=0.01, batch_size=256):
    """All receivers use identical independent noise windows. No signal/test data.

    One threshold per receiver. The returned thresholds are frozen during test.
    The normalized network and correlators have noise-scale-invariant scores.
    """
    if not 0 < target_pfa < 1 or num_samples < 1:
        raise ValueError("Require 0 < target_pfa < 1 and num_samples > 0")
    scores = {name: [] for name in models}
    reference = generate_pss(0)
    for model in models.values():
        model.eval()
    with torch.inference_mode():
        for start in range(0, num_samples, batch_size):
            size = min(batch_size, num_samples - start)
            noise = generate_noise_vector(size, snr_db, reference)
            offsets = torch.randint(0, 1920, (size,))
            # Include the same frequency-correction path used in testing.
            corrected = correct_frequency_offset(noise, offsets, torch.arange(2048))
            for name, model in models.items():
                device = next(model.parameters(), model.projection if isinstance(model, Correlator)
                              else torch.empty(0)).device
                score, _ = score_and_id(model, corrected.to(device))
                scores[name].append(score.cpu())
    thresholds = {}
    for name, parts in scores.items():
        values = torch.cat(parts)
        threshold = torch.quantile(values, 1-target_pfa, interpolation="higher").item()
        thresholds[name] = threshold
        if hasattr(models[name], "detection_threshold"):
            models[name].detection_threshold.fill_(threshold)
    return thresholds


def wilson_interval(successes, total, z=1.96):
    if total == 0:
        return [float("nan"), float("nan")]
    p = successes / total
    denominator = 1 + z*z / total
    centre = (p + z*z/(2*total)) / denominator
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0., centre-radius), min(1., centre+radius)]


def metrics_from_confusion(confusion):
    """Rows: truth; columns: prediction. Wrong PSS IDs count in Pd, not accuracy."""
    total = int(confusion.sum())
    signals = int(confusion[:3].sum())
    noise = int(confusion[3].sum())
    correct = int(confusion.diagonal().sum())
    detected = int(confusion[:3, :3].sum())
    false_alarms = int(confusion[3, :3].sum())
    correct_id = int(confusion[:3, :3].diagonal().sum())
    if not total or not signals or not noise:
        raise ValueError("Evaluation requires signal-present and noise-only examples")
    return {"accuracy": correct/total, "pd": detected/signals,
            "pfa": false_alarms/noise, "correct_id_given_signal": correct_id/signals,
            "signal_count": signals, "noise_count": noise,
            "pd_ci95": wilson_interval(detected, signals),
            "pfa_ci95": wilson_interval(false_alarms, noise),
            "confusion": confusion.tolist()}


def evaluate_receivers(models, loader, thresholds, raw_diagnostics=False):
    confusion = {name: torch.zeros(4, 4, dtype=torch.long) for name in models}
    raw = {name: torch.zeros(4, 4, dtype=torch.long) for name, model in models.items()
           if not isinstance(model, Correlator)}
    for model in models.values():
        model.eval()
    with torch.inference_mode():
        for samples, labels, offsets in loader:
            corrected = correct_frequency_offset(samples, offsets, torch.arange(samples.shape[1]))
            for name, model in models.items():
                device = next(model.parameters(), model.projection if isinstance(model, Correlator)
                              else torch.empty(0)).device
                output = model(corrected.to(device))
                if isinstance(model, Correlator):
                    score, predicted = output.max(dim=1)
                    predicted = torch.where(score > thresholds[name], predicted, 3)
                else:
                    predicted = predict_from_logits(output, thresholds[name])
                    if raw_diagnostics:
                        raw[name] += torch.bincount(4*labels.cpu()+output.argmax(1).cpu(),
                                                   minlength=16).reshape(4, 4)
                confusion[name] += torch.bincount(4*labels.cpu()+predicted.cpu(),
                                                 minlength=16).reshape(4, 4)
    result = {name: metrics_from_confusion(matrix) for name, matrix in confusion.items()}
    if raw_diagnostics:
        for name in raw:
            result[name]["raw_argmax"] = metrics_from_confusion(raw[name])
    return result


def latency_us(model, samples, offsets=None, repetitions=100, warmup=30):
    """Median wall-clock microseconds/window; include correction if offsets given.

    Input/output are resident on the tested device; transfer time is excluded.
    Includes the presence/ID decision and synchronizes CUDA when needed.
    """
    model.eval()
    n = torch.arange(samples.shape[1], device=samples.device)
    def run():
        corrected = samples if offsets is None else correct_frequency_offset(samples, offsets, n)
        scores, ids = score_and_id(model, corrected)
        return torch.where(scores > 0, ids, 3)
    def sync():
        if samples.is_cuda:
            torch.cuda.synchronize(samples.device)
    times = []
    with torch.inference_mode():
        for _ in range(warmup):
            run()
        sync()
        for _ in range(repetitions):
            sync()
            start = time.perf_counter_ns()
            run()
            sync()
            times.append((time.perf_counter_ns()-start)/1000/samples.shape[0])
    return sorted(times)[len(times)//2]
