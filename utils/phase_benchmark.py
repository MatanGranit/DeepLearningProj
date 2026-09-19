"""Random-phase benchmark for the compact phase-invariant complex CNN."""
import argparse
import copy
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

if __package__:
    from .benchmark import ROOT, SNR_RANGE, arithmetic_cost
    from .correlator import Correlator
    from .data_loader import PssDataset
    from .evaluation import calibrate_thresholds, evaluate_receivers, latency_us
    from .net import EfficientPssCnn, PhaseInvariantPssCnn, train_model
else:
    from benchmark import ROOT, SNR_RANGE, arithmetic_cost
    from correlator import Correlator
    from data_loader import PssDataset
    from evaluation import calibrate_thresholds, evaluate_receivers, latency_us
    from net import EfficientPssCnn, PhaseInvariantPssCnn, train_model


def phase_loader(snrs, samples_per_snr, batch_size, seed, shuffle=True):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        dataset = ConcatDataset([
            PssDataset(samples_per_snr, snr_db=snr, random_phase=True)
            for snr in snrs
        ])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed + 1))


def phase_cnn_cost(model):
    """Approximate real arithmetic, using the original report convention."""
    n = model.fft_binsize
    local_macs = 4 * n * model.rank
    global_macs = 4 * 3 * model.positions * model.rank
    macs = local_macs + global_macs
    # Input energy: 2N squares/additions. Three complex energies, scaling,
    # divisions, log1p, learned scale and biases are counted as unit ops.
    overhead = 4*n + 23
    return {"conv_linear_macs": int(macs), "other_real_ops": int(overhead),
            "total_real_ops": int(2*macs + overhead),
            "parameters": sum(p.numel() for p in model.parameters()),
            "stored_bytes": sum(t.numel()*t.element_size() for t in
                                list(model.parameters())+list(model.buffers()))}


def develop_architecture(device, batch_size):
    """Pre-test selection under the old detector's 44,039-operation budget."""
    candidates = ((16, 1), (8, 1), (4, 1), (16, 2))
    old_budget = 44039
    records = []
    for block_size, rank in candidates:
        train = phase_loader(SNR_RANGE, 300, batch_size, 9_110_001)
        validation = phase_loader(SNR_RANGE, 150, batch_size, 9_120_001, False)
        torch.manual_seed(9_130_000 + 10*block_size + rank)
        model = PhaseInvariantPssCnn(block_size=block_size, rank=rank).to(device)
        history = train_model(model, train, epochs=4,
                              validation_loader=validation, lr=3e-4)
        cost = phase_cnn_cost(model)
        records.append({"block_size": block_size, "rank": rank,
                        "validation_loss": min(h["validation_loss"] for h in history),
                        "total_real_ops": cost["total_real_ops"],
                        "eligible": cost["total_real_ops"] < old_budget})
    eligible = [record for record in records if record["eligible"]]
    selected = min(eligible, key=lambda value: value["validation_loss"])
    return records, {"block_size": selected["block_size"], "rank": selected["rank"]}


def run(args):
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "random_phase_results.json").exists():
        raise FileExistsError("Choose a new output directory; results are not overwritten")

    development, selected = develop_architecture(device, args.batch_size)
    print(f"Selected phase CNN: {selected}; development={development}", flush=True)
    report = {
        "experiment": "Aligned AWGN with an independent uniform phase per window",
        "config": {**vars(args), "output": str(args.output),
                   "selected_architecture": selected},
        "environment": {"python": platform.python_version(),
                        "torch": str(torch.__version__), "platform": platform.platform(),
                        "device": str(device), "threads": torch.get_num_threads()},
        "development": development,
        "notes": [
            "The original fixed-phase results directory is not modified.",
            "All signal windows have independent phase uniform on [0, 2*pi).",
            "Timing is aligned and the exact frequency offset is corrected for every receiver.",
            "Training, validation, calibration and test random streams are disjoint.",
            "Every receiver sees the same test IQ and is calibrated to the same Pfa target.",
            "The legacy linear detector is retrained on the same random-phase training data.",
        ],
        "runs": [],
    }
    started = time.perf_counter()
    for seed in args.seeds:
        train = phase_loader(SNR_RANGE, args.train_per_snr, args.batch_size,
                             110_000 + seed)
        validation = phase_loader(SNR_RANGE, args.validation_per_snr,
                                  args.batch_size, 120_000 + seed, False)
        models, training = {}, {}
        specifications = (
            ("phase_cnn", PhaseInvariantPssCnn(**selected), 3e-4),
            ("legacy_linear", EfficientPssCnn(), 1e-4),
        )
        for index, (name, model, learning_rate) in enumerate(specifications):
            torch.manual_seed(130_000 + 10*seed + index)
            train.generator.manual_seed(110_001 + seed)
            model = model.to(device)
            print(f"Seed {seed}: training {name}", flush=True)
            before = time.perf_counter()
            history = train_model(model, train, args.epochs,
                                  validation_loader=validation, lr=learning_rate)
            training[name] = {"seconds": time.perf_counter()-before,
                              "learning_rate": learning_rate, "history": history}
            models[name] = model.eval()
        del train, validation
        models["correlator_magnitude"] = Correlator().to(device)
        models["correlator_coherent"] = Correlator(coherent=True).to(device)
        trial = {"seed": seed, "training": training, "cost": {},
                 "latency": {}, "points": []}
        for name, model in models.items():
            trial["cost"][name] = (phase_cnn_cost(model) if name == "phase_cnn"
                                    else arithmetic_cost(model))
            trial["latency"][name] = {}
            for batch in (1, 64):
                x = torch.randn(batch, 2048, dtype=torch.complex64, device=device)
                offsets = torch.randint(0, 1920, (batch,), device=device)
                trial["latency"][name][str(batch)] = {
                    "core_us_per_window": latency_us(model, x),
                    "with_correction_us_per_window": latency_us(model, x, offsets),
                }
        for index, snr in enumerate(args.snrs):
            torch.manual_seed(140_000 + 100*seed + index)
            thresholds = calibrate_thresholds(
                models, snr_db=snr, num_samples=args.calibration_samples,
                target_pfa=args.target_pfa, batch_size=args.batch_size)
            torch.manual_seed(150_000 + 100*seed + index)
            loader = DataLoader(
                PssDataset(args.test_samples, snr_db=snr, random_phase=True),
                batch_size=args.batch_size)
            values = evaluate_receivers(models, loader, thresholds,
                                        raw_diagnostics=True)
            trial["points"].append({"snr_db": snr, "thresholds": thresholds,
                                    "metrics": values})
            print(f"Seed {seed}, random phase, SNR {snr:5.1f}: " + " | ".join(
                f"{name}: Pd={100*m['pd']:.1f}, Pfa={100*m['pfa']:.2f}"
                for name, m in values.items()), flush=True)
        for name in ("phase_cnn", "legacy_linear"):
            torch.manual_seed(160_000 + 10*seed + (name == "legacy_linear"))
            calibrate_thresholds({name: models[name]}, snr_db=-30.,
                                 num_samples=args.calibration_samples,
                                 target_pfa=args.target_pfa,
                                 batch_size=args.batch_size)
            torch.save({"architecture": name, "model": models[name].state_dict(),
                        "target_pfa": args.target_pfa, "seed": seed,
                        "random_phase_training": True,
                        "model_kwargs": selected if name == "phase_cnn" else {}},
                       args.output / f"{name}_seed{seed}.pth")
        report["runs"].append(trial)
        (args.output / "progress.json").write_text(json.dumps(report, indent=2))
    report["elapsed_seconds"] = time.perf_counter()-started
    (args.output / "random_phase_results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False))
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[91, 92, 93])
    parser.add_argument("--train-per-snr", type=int, default=3000)
    parser.add_argument("--validation-per-snr", type=int, default=400)
    parser.add_argument("--test-samples", type=int, default=8000)
    parser.add_argument("--calibration-samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-pfa", type=float, default=.01)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--snrs", type=float, nargs="+", default=SNR_RANGE)
    parser.add_argument("--output", type=Path, default=ROOT / "results_random_phase")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
