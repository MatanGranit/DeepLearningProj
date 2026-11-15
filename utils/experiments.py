"""The kinds of experiment: declarative descriptions that :class:`~utils.trainer.Trainer` executes.

An experiment says *what* to run -- which detectors to train, whether the data has
random phase, which seeds to use, how its results and checkpoints are named and laid
out. It trains, tests and writes nothing itself.

* :class:`KnownBinExperiment` -- every signal window carries an independent phase
  uniform on [0, 2*pi); the frequency offset (bin) is removed and both receivers work on
  the 128 samples that hold the PSS band. Trains ``NarrowbandPssCnn``. Stored results:
  ``results_known_bin_128/``.
* :class:`UnknownBinExperiment` -- random phase, and the network gets the window without
  its frequency offset removed. Trains ``ConvScanningPssCnn``. Stored results:
  ``results_unknown_bin/``.

New training runs go to ``runs/<RUN_NAME>-<date>-<time>/`` unless ``--output`` names a folder.
"""
import argparse
import platform
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import torch

from .data.pss_signal import SNR_DEFINITION
from .evaluation import DEFAULT_CALIBRATION_SAMPLES, DEFAULT_TARGET_PFA
from .models.correlator import Correlator, NarrowbandCorrelator
from .models.networks import ConvScanningPssCnn, NarrowbandPssCnn, PssDetector

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The SNRs (dB, per PSS subcarrier) swept for training, calibration and test: dense
# around the detection knee (50% Pd near -14 dB), with both tails.
SNR_RANGE = [-26.0, -23.0, -21.0, -19.0, -18.0, -17.0, -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -9.0, -7.0,
             -5.0, 5.0]

DEFAULT_LEARNING_RATE = 1e-4
# Training from random weights has the whole template to learn, not a small correction
# to it, so it takes larger steps.
RANDOM_START_LEARNING_RATE = 1e-3


@dataclass(frozen=True)
class SeedPlan:
    """Base seed of every random stream in one experiment.

    The bases are far apart, so for the default run seeds the train, validation,
    model-initialization, calibration, test and deployment streams do not overlap.

    :param train: training data; its shuffling uses ``train + 1 + seed``
    :param validation: validation data
    :param model_init: global RNG before building each trained detector
    :param calibration: per-SNR noise-only calibration windows
    :param test: per-SNR test windows
    :param deployment: final deployment-threshold calibration
    :param per_seed_stride: multiplier of the run seed in the ``model_init`` and
        ``deployment`` seeds, which also add the detector's index
    :param snr_stride: multiplier of the run seed in the ``calibration`` and ``test``
        seeds, which also add the SNR's index
    """

    train: int
    validation: int
    model_init: int
    calibration: int
    test: int
    deployment: int
    per_seed_stride: int
    snr_stride: int = 100


@dataclass(frozen=True)
class TrainedModelSpec:
    """A neural detector an experiment trains: its results key, constructor and learning rate."""

    name: str
    build: Callable[[], PssDetector]
    learning_rate: float


class Experiment:
    """Base description of an experiment; the subclasses fill in what differs.

    Holds the parsed options and answers the trainer's questions: which detectors
    to train, whether the data has random phase, which seeds to use, and how the
    results JSON, console lines and checkpoints look.
    """

    DEFAULT_SEEDS: List[int]
    DEFAULT_EPOCHS = 8
    # The stored results a bare ``main.py report`` renders.
    RESULTS_DIR: Path
    # A training run without --output writes to runs/<RUN_NAME>-<date>-<time>, so it never
    # touches the stored results.
    RUN_NAME: str
    RESULTS_FILE: str
    # None: the results file itself is rewritten after every seed. Otherwise partial
    # results go to this file, and the results file is written once, at the end.
    PROGRESS_FILE: Optional[str] = None
    # False: the trainer refuses to run into a directory that already holds results.
    OVERWRITE_RESULTS: bool = True
    RANDOM_PHASE: bool
    SEEDS: SeedPlan

    def __init__(self, options: argparse.Namespace) -> None:
        """
        Bind the experiment to its parsed options.

        :param options: options from :meth:`parse_options`; ``vars(options)`` is recorded
            as the results' ``config`` block, so it must hold only those options
        """
        self.options = options
        self.device = torch.device(options.device)

    @classmethod
    def parse_options(cls, argv: Optional[Sequence[str]] = None, prog: Optional[str] = None) -> argparse.Namespace:
        """
        Parse this experiment's options: data sizes, epochs, seeds, SNR sweep, output.

        :param argv: arguments to parse (default: ``sys.argv[1:]``)
        :param prog: program name shown in usage and help (default: the script name)
        :return: the parsed options
        """
        parser = argparse.ArgumentParser(prog=prog, description=cls.__doc__,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("--seeds", type=int, nargs="+", default=cls.DEFAULT_SEEDS,
                            help="run seeds; each trains and tests the detectors once")
        parser.add_argument("--train-per-snr", type=int, default=3000,
                            help="training windows per SNR of the fixed training sweep SNR_RANGE")
        parser.add_argument("--validation-per-snr", type=int, default=400, help="validation windows per SNR")
        parser.add_argument("--test-samples", type=int, default=8000, help="test windows per tested SNR and seed")
        parser.add_argument("--calibration-samples", type=int, default=DEFAULT_CALIBRATION_SAMPLES,
                            help="noise-only windows per tested SNR and seed for the threshold")
        parser.add_argument("--epochs", type=int, default=cls.DEFAULT_EPOCHS, help="training epochs")
        parser.add_argument("--batch-size", type=int, default=128, help="windows per batch")
        parser.add_argument("--target-pfa", type=float, default=DEFAULT_TARGET_PFA,
                            help="false-alarm rate the thresholds are calibrated to")
        parser.add_argument("--threads", type=int, default=1, help="CPU threads for torch")
        parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                            help="torch device")
        parser.add_argument("--snrs", type=float, nargs="+", default=SNR_RANGE,
                            help="SNRs (dB) to calibrate and test at; training always uses SNR_RANGE")
        parser.add_argument("--output", type=Path, default=cls.new_run_dir(),
                            help="folder for results and checkpoints (default: a new folder under runs/)")
        return parser.parse_args(argv)

    @classmethod
    def new_run_dir(cls) -> Path:
        """
        Name a fresh output folder for a training run.

        :return: ``runs/<RUN_NAME>-<YYYYmmdd-HHMMSS>`` under the project root
        """
        return PROJECT_ROOT / "runs" / f"{cls.RUN_NAME}-{datetime.now():%Y%m%d-%H%M%S}"

    @property
    def results_path(self) -> Path:
        """Return the path of the final results JSON inside the output directory."""
        return self.options.output / self.RESULTS_FILE

    def models_record(self) -> Dict[str, str]:
        """
        Name the network class of each trained detector, so reports can label it.

        :return: results key -> class name
        """
        return {spec.name: spec.build.__name__ for spec in self.trained_models()}

    def config_record(self) -> dict:
        """
        Return the options as recorded in the results, with the output path as a string.

        :return: ``vars(options)`` with ``output`` converted to ``str``
        """
        return {**vars(self.options), "output": str(self.options.output)}

    def environment_record(self) -> dict:
        """
        Describe the machine and software the run was recorded on.

        :return: dict with ``python``, ``torch``, ``platform``, ``device`` and ``threads``
        """
        return {"python": platform.python_version(), "torch": str(torch.__version__),
                "platform": platform.platform(), "device": str(self.device),
                "threads": torch.get_num_threads()}

    def trained_models(self) -> List[TrainedModelSpec]:
        """
        List the neural detectors this experiment trains, in training order.

        :return: one :class:`TrainedModelSpec` per detector
        :raises NotImplementedError: in the base class
        """
        raise NotImplementedError

    def correlators(self) -> Dict[str, Correlator]:
        """
        Build the classical receivers the trained detectors are tested against.

        :return: results key -> correlator
        :raises NotImplementedError: in the base class
        """
        raise NotImplementedError

    def results_header(self) -> dict:
        """
        Return the results skeleton, with an empty ``runs`` list.

        :return: the results dict
        :raises NotImplementedError: in the base class
        """
        raise NotImplementedError

    def training_record(self, seconds: float, learning_rate: float, history: List[Dict]) -> dict:
        """
        Build the results entry describing one detector's training.

        :param seconds: wall-clock training time in seconds
        :param learning_rate: the initial learning rate used
        :param history: per-epoch records from :meth:`~utils.trainer.Trainer.fit`
        :return: the entry stored under ``training[name]``
        :raises NotImplementedError: in the base class
        """
        raise NotImplementedError

    def progress_line(self, seed: int, snr: float, metrics: Dict[str, Dict]) -> str:
        """
        Format the console line printed after each tested SNR point.

        :param seed: the run seed
        :param snr: the SNR point in dB
        :param metrics: name -> metrics dict at this SNR
        :return: one line of text
        :raises NotImplementedError: in the base class
        """
        raise NotImplementedError

    def checkpoint_name(self, name: str, seed: int) -> str:
        """
        Name the checkpoint file of one trained detector.

        :param name: the detector's results key
        :param seed: the run seed
        :return: a file name inside the output directory
        :raises NotImplementedError: in the base class
        """
        raise NotImplementedError

    def checkpoint_extras(self, name: str) -> dict:
        """
        Return checkpoint keys beyond architecture, model_class, model, target_pfa and seed.

        :param name: the detector's results key
        :return: extra entries for the checkpoint dict (default: none)
        """
        return {}


class KnownBinExperiment(Experiment):
    """Known-bin experiment: NarrowbandPssCnn against a magnitude correlator, both on 128 samples.

    Every signal window carries an independent phase uniform on [0, 2*pi). Timing and
    frequency offset are known exactly; both receivers keep only the 128 FFT bins that
    hold the PSS band before scoring. The network starts from random weights, so it
    trains longer and with larger steps. --output must name a directory without an
    existing known_bin_128_results.json: results are never overwritten.
    """

    DEFAULT_SEEDS = [91, 92, 93]
    DEFAULT_EPOCHS = 20
    RESULTS_DIR = PROJECT_ROOT / "results_known_bin_128"
    RUN_NAME = "known-bin-128"
    RESULTS_FILE = "known_bin_128_results.json"
    PROGRESS_FILE = "progress.json"
    OVERWRITE_RESULTS = False
    RANDOM_PHASE = True
    SEEDS = SeedPlan(train=110_000, validation=120_000, model_init=130_000, calibration=140_000, test=150_000,
                     deployment=160_000, per_seed_stride=10)

    def trained_models(self) -> List[TrainedModelSpec]:
        """
        List the 128-sample CNN only, under the results key ``narrow_cnn``.

        :return: one spec, ``narrow_cnn``
        """
        return [TrainedModelSpec("narrow_cnn", NarrowbandPssCnn, RANDOM_START_LEARNING_RATE)]

    def correlators(self) -> Dict[str, Correlator]:
        """
        Build the magnitude correlator on the 128-sample view.

        :return: ``correlator_magnitude`` only
        """
        return {"correlator_magnitude": NarrowbandCorrelator()}

    def results_header(self) -> dict:
        """
        Return the known-bin results skeleton.

        :return: the results dict
        """
        return {
            "experiment": "Aligned AWGN with an independent uniform phase; receivers on the 128 PSS-band samples",
            "config": self.config_record(),
            "models": self.models_record(),
            "snr_definition": SNR_DEFINITION,
            "environment": self.environment_record(),
            "notes": [
                "All signal windows have independent phase uniform on [0, 2*pi).",
                "Timing is aligned and the exact frequency offset is corrected for every receiver.",
                "Both receivers keep the 128 FFT bins from the PSS's first bin (an ideal band-pass and decimation "
                "by 16); that front end is not counted in their operations.",
                "Training, validation, calibration and test random streams are disjoint.",
                "Every receiver sees the same test IQ and is calibrated to the same Pfa target.",
            ],
            "runs": [],
        }

    def training_record(self, seconds: float, learning_rate: float, history: List[Dict]) -> dict:
        """
        Record training time, learning rate and history.

        :param seconds: wall-clock training time in seconds
        :param learning_rate: the initial learning rate used
        :param history: per-epoch records
        :return: ``{"seconds", "learning_rate", "history"}``
        """
        return {"seconds": seconds, "learning_rate": learning_rate, "history": history}

    def progress_line(self, seed: int, snr: float, metrics: Dict[str, Dict]) -> str:
        """
        Report Pd and Pfa of every receiver at one SNR.

        :param seed: the run seed
        :param snr: the SNR point in dB
        :param metrics: name -> metrics dict at this SNR
        :return: one line of text
        """
        return f"Seed {seed}, random phase, SNR {snr:5.1f}: " + " | ".join(
            f"{name}: Pd={100*m['pd']:.1f}, Pfa={100*m['pfa']:.2f}" for name, m in metrics.items())

    def checkpoint_name(self, name: str, seed: int) -> str:
        """
        Name the checkpoint ``<name>_seed<N>.pth``.

        :param name: the detector's results key
        :param seed: the run seed
        :return: the file name
        """
        return f"{name}_seed{seed}.pth"

    def checkpoint_extras(self, name: str) -> dict:
        """
        Mark random-phase training; ``model_kwargs`` is empty because both architectures are fixed.

        :param name: the detector's results key
        :return: ``random_phase_training`` and ``model_kwargs``
        """
        return {"random_phase_training": True, "model_kwargs": {}}


class UnknownBinExperiment(KnownBinExperiment):
    """Unknown-bin experiment: the scanning CNN gets the window as received, the correlator gets it corrected.

    Every signal window carries an independent uniform phase and one of the dataset's
    frequency offsets. The network (scan_cnn, ConvScanningPssCnn) is not told the
    offset and searches all of them. The magnitude correlator is handed the window with
    its offset removed, as in the other experiments. --output must name a directory
    without an existing unknown_bin_results.json: results are never overwritten.
    """

    DEFAULT_EPOCHS = Experiment.DEFAULT_EPOCHS
    RESULTS_DIR = PROJECT_ROOT / "results_unknown_bin"
    RUN_NAME = "unknown-bin"
    RESULTS_FILE = "unknown_bin_results.json"

    def trained_models(self) -> List[TrainedModelSpec]:
        """
        List the scanning CNN only, under the results key ``scan_cnn``.

        :return: one spec, ``scan_cnn``
        """
        return [TrainedModelSpec("scan_cnn", ConvScanningPssCnn, DEFAULT_LEARNING_RATE)]

    def correlators(self) -> Dict[str, Correlator]:
        """
        Build the magnitude correlator, handed the window with its offset removed.

        :return: ``correlator_magnitude`` only
        """
        return {"correlator_magnitude": Correlator()}

    def results_header(self) -> dict:
        """
        Return the unknown-bin results skeleton.

        :return: the results dict
        """
        header = super().results_header()
        header["experiment"] = "AWGN, independent uniform phase, frequency offset unknown to the network"
        header["notes"] = [
            "All signal windows have independent phase uniform on [0, 2*pi) and a random frequency offset "
            "from 0 to 1840 bins in steps of 80.",
            "scan_cnn gets the window as received and searches every offset; "
            "correlator_magnitude gets it with the exact offset removed.",
            "Training, validation, calibration and test random streams are disjoint.",
            "Every receiver sees the same test IQ and is calibrated to the same Pfa target.",
        ]
        return header

    def progress_line(self, seed: int, snr: float, metrics: Dict[str, Dict]) -> str:
        """
        Report Pd and Pfa of every receiver at one SNR.

        :param seed: the run seed
        :param snr: the SNR point in dB
        :param metrics: name -> metrics dict at this SNR
        :return: one line of text
        """
        return super().progress_line(seed, snr, metrics).replace("random phase", "unknown bin")
