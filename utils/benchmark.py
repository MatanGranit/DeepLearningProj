"""Train the compact PSS detector and evaluate two correlator baselines on shared test data."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

if __package__:
    from .data_loader import PssDataset
    from .net import EfficientPssCnn, train_model
    from .correlator import Correlator
    from .evaluation import calibrate_thresholds, evaluate_receivers, latency_us
else:
    from data_loader import PssDataset
    from net import EfficientPssCnn, train_model
    from correlator import Correlator
    from evaluation import calibrate_thresholds, evaluate_receivers, latency_us

SNR_RANGE = [-50., -45., -40., -38., -36., -34., -32., -30., -28., -26., -24., -22., -20., -10., 0.]
ROOT = Path(__file__).resolve().parents[1]


def pooled_loader(snrs, samples_per_snr, batch_size, seed, shuffle=True):
    # Model initialization cannot change the data or minibatch order.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        dataset = ConcatDataset([PssDataset(samples_per_snr, snr_db=s) for s in snrs])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed+1))


def arithmetic_cost(model):
    """Real arithmetic after common frequency correction, including normalization.

    Count multiplication/addition/division/sqrt as one each (an approximation).
    Count multiply-accumulate as two; comparisons, memory and kernel launches
    are excluded.
    Timing is measured separately. Fixed template initialization is offline.
    """
    n = 2048
    if isinstance(model, Correlator):
        macs = (3 if model.coherent else 6) * 2*n
        overhead = 4*n + (3 if model.coherent else 15)
    else:
        macs = 0
        overhead = 0
        def count(layer, inputs, output):
            nonlocal macs, overhead
            if isinstance(layer, torch.nn.Linear):
                macs += output.numel() * layer.in_features
                overhead += output.numel() if layer.bias is not None else 0
        hooks = [m.register_forward_hook(count) for m in model.modules() if isinstance(
            m, torch.nn.Linear)]
        was_training = model.training
        try:
            model.eval()
            with torch.inference_mode():
                model(torch.zeros(1, n, dtype=torch.complex64, device=next(model.parameters()).device))
        finally:
            for hook in hooks:
                hook.remove()
            model.train(was_training)
        if isinstance(model, EfficientPssCnn):
            overhead += 4*n + 7  # energy, RMS, three divisions and class biases
    return {"conv_linear_macs": int(macs), "other_real_ops": int(overhead),
            "total_real_ops": int(2*macs+overhead),
            "parameters": sum(p.numel() for p in model.parameters()),
            "stored_bytes": sum(t.numel()*t.element_size() for t in
                                list(model.parameters())+list(model.buffers()))}


def run(args):
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"config": {**vars(args), "output": str(args.output)},
              "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                              "platform": platform.platform(), "device": str(device),
                              "threads": torch.get_num_threads()},
              "notes": ["Calibration uses independent noise at each tested background noise power.",
                        "Train, model-selection validation, calibration and test seeds are disjoint.",
                        "All receivers see the same test windows and known frequency offset.",
                        "The network uses template initialization and a known fixed phase; coherent baseline included.",
                        "Latencies exclude data transfer, use median warmed inference, and include decisions."],
              "runs": []}
    for seed in args.seeds:
        validation = pooled_loader(SNR_RANGE, args.validation_per_snr, args.batch_size,
                                   20000+seed, shuffle=False)
        train = pooled_loader(SNR_RANGE, args.train_per_snr, args.batch_size, 10000+seed)
        models = {}
        training = {}
        for name in ("compact",):
            torch.manual_seed(30000+seed)
            model = EfficientPssCnn().to(device)
            train.generator.manual_seed(10001+seed)
            print(f"Seed {seed}: training {name}", flush=True)
            start = time.perf_counter()
            history = train_model(model, train, args.epochs, validation_loader=validation)
            training[name] = {"seconds": time.perf_counter()-start, "history": history}
            models[name] = model.eval()
        del train, validation
        models["correlator_magnitude"] = Correlator().to(device)
        models["correlator_coherent"] = Correlator(coherent=True).to(device)
        seed_result = {"seed": seed, "training": training, "cost": {}, "latency": {}, "points": []}
        for name, model in models.items():
            seed_result["cost"][name] = arithmetic_cost(model)
            seed_result["latency"][name] = {}
            for batch in (1, 64):
                x = torch.randn(batch, 2048, dtype=torch.complex64, device=device)
                offsets = torch.randint(0, 1920, (batch,), device=device)
                seed_result["latency"][name][str(batch)] = {
                    "core_us_per_window": latency_us(model, x),
                    "with_correction_us_per_window": latency_us(model, x, offsets)}
        for index, snr in enumerate(args.snrs):
            torch.manual_seed(40000+100*seed+index)
            thresholds = calibrate_thresholds(models, snr_db=snr, num_samples=args.calibration_samples,
                                               target_pfa=args.target_pfa, batch_size=args.batch_size)
            torch.manual_seed(60000+100*seed+index)
            loader = DataLoader(PssDataset(args.test_samples, snr_db=snr), batch_size=args.batch_size)
            metrics = evaluate_receivers(models, loader, thresholds, raw_diagnostics=True)
            seed_result["points"].append({"snr_db": snr, "thresholds": thresholds, "metrics": metrics})
            print(f"Seed {seed}, SNR {snr:5.1f}: " + " | ".join(
                f"{name}: acc={100*m['accuracy']:.1f}, Pd={100*m['pd']:.1f}, Pfa={100*m['pfa']:.2f}"
                for name, m in metrics.items()), flush=True)
        # The network is scale-invariant: calibrate ONE final deployment threshold at -30dB.
        if "compact" in models:
            torch.manual_seed(80000+seed)
            calibrate_thresholds({"compact": models["compact"]}, num_samples=args.calibration_samples,
                                  target_pfa=args.target_pfa, batch_size=args.batch_size)
            torch.save({"architecture": "compact", "model": models["compact"].state_dict(),
                        "target_pfa": args.target_pfa, "seed": seed},
                       args.output / f"pss_model_seed{seed}.pth")
        report["runs"].append(seed_result)
        with (args.output / "benchmark_results.json").open("w") as file:
            json.dump(report, file, indent=2, allow_nan=False)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
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
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
