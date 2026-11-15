"""The trainer: runs one experiment end to end and writes its results.

For every run seed, :meth:`Trainer.run_seed` does, in order:

1. :meth:`~Trainer.create_datasets` -- training and validation windows pooled over every SNR;
2. :meth:`~Trainer.train_models` -- train each of the experiment's neural detectors with
   :meth:`~Trainer.fit`, which alternates :meth:`~Trainer.train_epoch` and
   :meth:`~Trainer.validate` and keeps the weights with the highest validation Pd;
3. :meth:`~Trainer.test` (optional) -- add the two correlators, record each receiver's
   arithmetic cost, then at each SNR calibrate thresholds on noise and score every receiver;
4. :meth:`~Trainer.save_checkpoints` -- calibrate a deployment threshold per trained
   detector and save it.

After every seed the partial results are saved (the results file itself, or the
experiment's progress file); the final results file is written at the end. Plots are
not made here; that is :mod:`utils.reporting`'s job.
"""
import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .data.pss_dataset import PssDataset
from .data.pss_signal import NOISE_LABEL, PssSignalGenerator
from .evaluation import DEFAULT_TARGET_PFA, REFERENCE_SNR_DB, Evaluator
from .experiments import SNR_RANGE, Experiment
from .models.networks import PssDetector

WEIGHT_DECAY = 1e-4
# The normalized scores can produce large gradients at high SNR.
MAX_GRADIENT_NORM = 5.0
# Results of a session run with tests skipped. A separate name, so such a session
# never overwrites or blocks a directory's full results.
TRAINING_ONLY_RESULTS_FILE = "training_results.json"


class Trainer:
    """Execute an experiment: create datasets, train, validate, optionally test, save."""

    def __init__(self, experiment: Experiment) -> None:
        """
        Bind the trainer to the experiment it will run.

        :param experiment: which experiment to run, with its parsed options
        """
        self.experiment = experiment
        self.options = experiment.options
        self.device = experiment.device
        self.results_path = experiment.results_path

    def run(self, run_tests: bool = True) -> dict:
        """
        Run every seed and write the results JSON.

        :param run_tests: also run the test sweep (default: True). Without it the
            results hold training histories only, cannot be plotted, and are written to
            ``training_results.json`` instead of the experiment's results file.
        :return: the results dict; :attr:`results_path` says where it was written
        :raises FileExistsError: if the experiment does not overwrite results and the
            output directory already holds its results or training-only results
        """
        experiment = self.experiment
        torch.set_num_threads(self.options.threads)
        self.options.output.mkdir(parents=True, exist_ok=True)
        if not run_tests:
            self.results_path = self.options.output / TRAINING_ONLY_RESULTS_FILE
        # Checked against both files: a session writes checkpoints with or without tests.
        if not experiment.OVERWRITE_RESULTS and (experiment.results_path.exists() or self.results_path.exists()):
            raise FileExistsError("Choose a new output directory; results are not overwritten")
        results = experiment.results_header()
        started = time.perf_counter()
        for seed in self.options.seeds:
            results["runs"].append(self.run_seed(seed, run_tests))
            self.save_progress(results)
        self.save_final(results, time.perf_counter() - started)
        return results

    def save_progress(self, results: dict) -> None:
        """
        Save the partial results after a seed, so an interrupted run leaves usable JSON.

        :param results: the results so far
        """
        if self.experiment.PROGRESS_FILE is None:
            self.write_json(self.results_path, results, allow_nan=False)
        else:
            self.write_json(self.options.output / self.experiment.PROGRESS_FILE, results)

    def save_final(self, results: dict, elapsed_seconds: float) -> None:
        """
        Write the final results file of an experiment that keeps a separate progress file.

        Its presence then means the run completed. Experiments without a progress file
        already wrote it after the last seed.

        :param results: the complete results
        :param elapsed_seconds: wall-clock time of all seeds, in seconds
        """
        if self.experiment.PROGRESS_FILE is not None:
            results["elapsed_seconds"] = elapsed_seconds
            self.write_json(self.results_path, results, allow_nan=False)

    @staticmethod
    def write_json(path: Path, results: dict, allow_nan: bool = True) -> None:
        """
        Write results as indented JSON.

        :param path: destination file
        :param results: the results dict
        :param allow_nan: False makes a NaN anywhere in the results an error (default: True)
        """
        path.write_text(json.dumps(results, indent=2, allow_nan=allow_nan))

    def run_seed(self, seed: int, run_tests: bool = True) -> dict:
        """
        Create datasets, train and validate, optionally test, and save checkpoints for one seed.

        :param seed: the run seed; every stream seed is derived from it via the
            experiment's :class:`~utils.experiments.SeedPlan`
        :param run_tests: run the test sweep (default: True)
        :return: the run record: ``seed``, ``training``, ``cost`` and one ``points``
            entry per tested SNR (the last two empty without tests)
        """
        train_loader, validation_loader = self.create_datasets(seed)
        models, training = self.train_models(seed, train_loader, validation_loader)
        # Free the training windows before allocating the test ones.
        del train_loader, validation_loader
        record = {"seed": seed, "training": training, "cost": {}, "points": []}
        if run_tests:
            record.update(self.test(models, seed))
        self.save_checkpoints(models, seed)
        return record

    def create_datasets(self, seed: int) -> Tuple[DataLoader, DataLoader]:
        """
        Build the training and validation loaders, pooled over every SNR in SNR_RANGE.

        :param seed: the run seed
        :return: (shuffled training loader, unshuffled validation loader)
        """
        options, plan = self.options, self.experiment.SEEDS
        random_phase = self.experiment.RANDOM_PHASE
        train_loader = PssDataset.pooled_loader(SNR_RANGE, options.train_per_snr, options.batch_size,
                                                plan.train + seed, random_phase=random_phase)
        validation_loader = PssDataset.pooled_loader(SNR_RANGE, options.validation_per_snr, options.batch_size,
                                                     plan.validation + seed, shuffle=False, random_phase=random_phase)
        return train_loader, validation_loader

    def train_models(self, seed: int, train_loader: DataLoader,
                     validation_loader: DataLoader) -> Tuple[Dict[str, PssDetector], Dict[str, dict]]:
        """
        Build and train every neural detector of the experiment, in order.

        :param seed: the run seed
        :param train_loader: shuffled training windows
        :param validation_loader: validation windows for model selection
        :return: (name -> trained detector in eval mode, name -> training record)
        """
        plan = self.experiment.SEEDS
        models, training = {}, {}
        for index, spec in enumerate(self.experiment.trained_models()):
            torch.manual_seed(plan.model_init + plan.per_seed_stride * seed + index)
            model = spec.build().to(self.device)
            # Identical batch order for every trained detector, so the comparison is fair.
            train_loader.generator.manual_seed(plan.train + 1 + seed)
            print(f"Seed {seed}: training {spec.name}", flush=True)
            start = time.perf_counter()
            history = self.fit(model, train_loader, validation_loader, self.options.epochs, spec.learning_rate,
                               self.options.target_pfa)
            training[spec.name] = self.experiment.training_record(time.perf_counter() - start,
                                                                  spec.learning_rate, history)
            models[spec.name] = model.eval()
        return models, training

    def fit(self, model: PssDetector, train_loader: DataLoader, validation_loader: Optional[DataLoader],
            epochs: int, learning_rate: float, target_pfa: float = DEFAULT_TARGET_PFA) -> List[Dict]:
        """
        Train one detector with cross-entropy, AdamW and cosine learning-rate decay.

        Each epoch is one :meth:`train_epoch` followed by one :meth:`validate`; the
        weights with the highest validation Pd at ``target_pfa`` -- the starting weights
        included -- are restored at the end. Any calibrated detection threshold is
        cleared on entry, because new weights invalidate it.

        :param model: the detector to train; it is mutated in place
        :param train_loader: yields ``(samples, labels, freq_offset)`` batches
        :param validation_loader: loader for model selection, or None to skip validation
        :param epochs: passes over the training loader; at least 1
        :param learning_rate: initial learning rate
        :param target_pfa: false-alarm rate the validation Pd is measured at (default: DEFAULT_TARGET_PFA)
        :return: one dict per epoch with keys ``epoch``, ``loss``, ``accuracy``
            (percent, train), ``validation_loss`` and ``validation_pd`` (both None
            without validation)
        :raises ValueError: if epochs < 1 or the training loader is empty
        """
        if epochs < 1 or len(train_loader) < 1:
            raise ValueError("Training requires positive epochs and a nonempty loader")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        print("Starting training phase...")
        history = []
        best_pd = -1.0
        best_state = None
        model.detection_threshold.fill_(float("nan"))
        if validation_loader is not None:
            # Selection is by Pd at the operating Pfa, the quantity the test reports:
            # cross-entropy can fall while detection at a fixed Pfa gets worse. The
            # starting weights compete with every epoch, so training never loses Pd.
            initial_loss, best_pd = self.validate(model, validation_loader, criterion, target_pfa)
            best_state = copy.deepcopy(model.state_dict())
            print(f"Initial validation loss: {initial_loss} | Initial validation Pd: {best_pd:.4f}")

        for epoch in range(epochs):
            epoch_loss, epoch_acc = self.train_epoch(model, train_loader, optimizer, criterion)
            val_loss = val_pd = None
            if validation_loader is not None:
                val_loss, val_pd = self.validate(model, validation_loader, criterion, target_pfa)
                if val_pd > best_pd:
                    best_pd = val_pd
                    # deepcopy, not a reference: the live weights keep changing.
                    best_state = copy.deepcopy(model.state_dict())
            history.append({"epoch": epoch + 1, "loss": epoch_loss, "accuracy": epoch_acc,
                            "validation_loss": val_loss, "validation_pd": val_pd})
            scheduler.step()

            print(
                f"Epoch {epoch + 1} | "
                f"Loss: {epoch_loss:.4f} | "
                f"Train Accuracy: {epoch_acc:.2f}% | "
                f"Validation loss: {val_loss if val_loss is not None else 'n/a'} | "
                f"Validation Pd: {val_pd if val_pd is not None else 'n/a'} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )
        if best_state is not None:
            model.load_state_dict(best_state)
        return history

    @staticmethod
    def train_epoch(model: PssDetector, loader: DataLoader, optimizer: optim.Optimizer,
                    criterion: nn.Module) -> Tuple[float, float]:
        """
        Run one optimization pass over ``loader``.

        Every batch is handed to the model exactly as at test time.

        :param model: the detector being trained
        :param loader: yields ``(samples, labels, freq_offset)`` batches
        :param optimizer: the optimizer stepping the model's parameters
        :param criterion: the training loss
        :return: (mean training loss, training accuracy in percent)
        """
        model.train()
        device = model.device
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        for samples, labels, freq_offset in loader:
            samples = samples.to(device)
            labels = labels.to(device)
            freq_offset = freq_offset.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(PssSignalGenerator.receiver_input(samples, freq_offset, model.RECEIVES_CORRECTED_WINDOW))
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRADIENT_NORM)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            # Plain argmax here: this is a training diagnostic, not the calibrated
            # operating point that the test reports.
            predicted = torch.argmax(outputs, dim=1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
        return running_loss / total_samples, 100 * correct_predictions / total_samples

    @staticmethod
    def validate(model: PssDetector, loader: DataLoader, criterion: nn.Module,
                 target_pfa: float = DEFAULT_TARGET_PFA) -> Tuple[float, float]:
        """
        Compute the mean loss and the detection probability over ``loader`` without updating the model.

        Pd is measured at a threshold set on the loader's own noise-only windows for
        ``target_pfa``, pooled over every SNR the loader holds.

        :param model: the detector being trained
        :param loader: yields ``(samples, labels, freq_offset)`` batches, with both
            signal and noise-only windows
        :param criterion: the training loss
        :param target_pfa: false-alarm rate of the threshold (default: DEFAULT_TARGET_PFA)
        :return: (mean loss per window, Pd)
        """
        model.eval()
        device = model.device
        loss_sum = 0.0
        scores, all_labels = [], []
        with torch.no_grad():
            for samples, labels, offsets in loader:
                samples, labels = samples.to(device), labels.to(device)
                logits = model(PssSignalGenerator.receiver_input(samples, offsets, model.RECEIVES_CORRECTED_WINDOW))
                loss_sum += criterion(logits, labels).item() * labels.numel()
                scores.append(model.presence_and_id(logits)[0].cpu())
                all_labels.append(labels.cpu())
        scores, labels = torch.cat(scores), torch.cat(all_labels)
        noise = labels == NOISE_LABEL
        threshold = Evaluator.threshold_for_pfa(scores[noise], target_pfa)
        return loss_sum / labels.numel(), (scores[~noise] > threshold).float().mean().item()

    def test(self, trained: Dict[str, PssDetector], seed: int) -> dict:
        """
        Test the trained detectors against both correlators on shared test windows.

        The arithmetic cost is recorded per receiver. Then, at each SNR, thresholds are
        calibrated on noise only and frozen before every receiver is scored on the
        same test windows.

        :param trained: name -> trained detector
        :param seed: the run seed
        :return: ``cost`` (name -> record) and ``points`` (one per SNR)
        """
        options, plan = self.options, self.experiment.SEEDS
        receivers = {**trained, **{name: correlator.to(self.device)
                                   for name, correlator in self.experiment.correlators().items()}}
        cost = {name: Evaluator.arithmetic_cost(receiver) for name, receiver in receivers.items()}
        points = []
        for index, snr in enumerate(options.snrs):
            # Calibrate at this SNR's noise power, then test -- in that order, so the
            # threshold never sees the windows it will be judged on.
            torch.manual_seed(plan.calibration + plan.snr_stride * seed + index)
            thresholds = Evaluator.calibrate_thresholds(receivers, snr_db=snr, num_samples=options.calibration_samples,
                                                        target_pfa=options.target_pfa, batch_size=options.batch_size)
            torch.manual_seed(plan.test + plan.snr_stride * seed + index)
            test_loader = DataLoader(PssDataset(options.test_samples, snr_db=snr,
                                                random_phase=self.experiment.RANDOM_PHASE),
                                     batch_size=options.batch_size)
            metrics = Evaluator.evaluate_receivers(receivers, test_loader, thresholds, raw_diagnostics=True)
            points.append({"snr_db": snr, "thresholds": thresholds, "metrics": metrics})
            print(self.experiment.progress_line(seed, snr, metrics), flush=True)
        return {"cost": cost, "points": points}

    def save_checkpoints(self, trained: Dict[str, PssDetector], seed: int) -> None:
        """
        Calibrate a deployment threshold for each trained detector and save its checkpoint.

        The threshold rides along inside ``state_dict()`` as a registered buffer.

        :param trained: name -> trained detector, in training order
        :param seed: the run seed
        """
        options, plan = self.options, self.experiment.SEEDS
        for index, (name, model) in enumerate(trained.items()):
            torch.manual_seed(plan.deployment + plan.per_seed_stride * seed + index)
            Evaluator.calibrate_thresholds({name: model}, snr_db=REFERENCE_SNR_DB,
                                           num_samples=options.calibration_samples, target_pfa=options.target_pfa,
                                           batch_size=options.batch_size)
            torch.save({"architecture": name, "model_class": type(model).__name__, "model": model.state_dict(),
                        "target_pfa": options.target_pfa, "seed": seed, **self.experiment.checkpoint_extras(name)},
                       options.output / self.experiment.checkpoint_name(name, seed))
