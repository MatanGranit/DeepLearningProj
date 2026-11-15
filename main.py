"""Entry point of the PSS detection project.

Three modes; ``python main.py`` alone runs ``train`` with every default:

    python main.py train       [--experiment E] [--no-test] [--no-report] [training options]
    python main.py report      [--experiment E] [--results FILE]
    python main.py correlation [--output DIR]

train   Create the datasets, train, validate and test, save the checkpoints and the
        results JSON into --output (default: a new folder under runs/), then write
        the report (figures and RESULTS.md) for that session and print the test
        results. --no-test skips the
        test sweep: checkpoints are still saved, the results go to
        training_results.json, and there is nothing to report. --no-report skips
        figures and RESULTS.md.
report  Only write the report, next to an existing results JSON (default: the
        experiment's stored results). Nothing is trained.
correlation
        Draw the PSS auto-, phase-shifted and cross-correlation figure into --output
        (default: figures/).

E is unknown-bin (default) or known-bin-128.
"python main.py train --help" lists the training options.
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Type

from utils.correlation_figure import PssCorrelationFigure
from utils.experiments import PROJECT_ROOT, Experiment, KnownBinExperiment, UnknownBinExperiment
from utils.reporting import KnownBinReporter, Reporter, UnknownBinReporter
from utils.trainer import Trainer


class Main:
    """Parse the command line and run the requested mode: ``train``, ``report`` or ``correlation``."""

    # --experiment value -> (experiment description, reporter for its results)
    EXPERIMENTS: Dict[str, Tuple[Type[Experiment], Type[Reporter]]] = {
        "unknown-bin": (UnknownBinExperiment, UnknownBinReporter),
        "known-bin-128": (KnownBinExperiment, KnownBinReporter),
    }
    DEFAULT_EXPERIMENT = "unknown-bin"
    FIGURES_DIR = PROJECT_ROOT / "figures"

    def __init__(self) -> None:
        """Build the command-line parser with its ``train``, ``report`` and ``correlation`` modes."""
        self.parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        modes = self.parser.add_subparsers(dest="mode", required=True, metavar="{train,report,correlation}")
        # add_help=False: --help is answered together with the experiment's own options.
        self.train_parser = modes.add_parser("train", add_help=False, allow_abbrev=False,
                                             help="train + validate (+ test), then report")
        self.add_experiment_option(self.train_parser)
        self.train_parser.add_argument("--no-test", action="store_true",
                                       help="skip the test sweep; results go to training_results.json")
        self.train_parser.add_argument("--no-report", action="store_true", help="do not write figures or RESULTS.md")
        report_parser = modes.add_parser("report", help="only write figures and RESULTS.md from a results JSON")
        self.add_experiment_option(report_parser)
        report_parser.add_argument("--results", type=Path, metavar="FILE",
                                   help="results JSON to report on (default: the experiment's stored results)")
        correlation_parser = modes.add_parser("correlation", help="draw the PSS correlation figure")
        correlation_parser.add_argument("--output", type=Path, default=self.FIGURES_DIR, metavar="DIR",
                                        help="folder for pss_correlation.png/.pdf/.svg (default: figures/)")

    def add_experiment_option(self, parser: argparse.ArgumentParser) -> None:
        """
        Add the ``--experiment`` choice to one mode's parser.

        :param parser: the mode's parser
        """
        parser.add_argument("--experiment", choices=list(self.EXPERIMENTS), default=self.DEFAULT_EXPERIMENT,
                            help=f"which experiment (default: {self.DEFAULT_EXPERIMENT})")

    def run(self, argv: Optional[Sequence[str]] = None) -> None:
        """
        Run the mode the command line requests.

        :param argv: arguments (default: ``sys.argv[1:]``); none at all means ``train``
            with every default. In ``train`` mode, every option other than
            ``--experiment``, ``--no-test`` and ``--no-report`` is a training option of the
            experiment.
        """
        argv = list(sys.argv[1:] if argv is None else argv) or ["train"]
        options, experiment_argv = self.parser.parse_known_args(argv)
        if options.mode == "correlation":
            if experiment_argv:
                self.parser.error(f"correlation takes only --output, got {experiment_argv}")
            PssCorrelationFigure().render(options.output)
            return
        experiment_type, reporter_type = self.EXPERIMENTS[options.experiment]
        if options.mode == "report":
            if experiment_argv:
                self.parser.error(f"report takes no training options, got {experiment_argv}")
            self.report(experiment_type, reporter_type, options.results)
        else:
            if "-h" in experiment_argv or "--help" in experiment_argv:
                # The experiment parser prints the training options and exits; show ours first.
                print(self.train_parser.format_help())
            self.train(experiment_type, reporter_type, experiment_argv,
                       run_tests=not options.no_test, write_report=not options.no_report)

    @staticmethod
    def train(experiment_type: Type[Experiment], reporter_type: Type[Reporter], experiment_argv: Sequence[str],
              run_tests: bool, write_report: bool) -> None:
        """
        Train, validate and optionally test one experiment, then report on that session.

        :param experiment_type: which experiment to run
        :param reporter_type: the reporter for its results
        :param experiment_argv: the experiment's training options
        :param run_tests: run the test sweep after training and validation
        :param write_report: write figures and RESULTS.md afterwards and print the test
            results (only possible with tests)
        """
        experiment = experiment_type(experiment_type.parse_options(experiment_argv, prog="main.py train"))
        trainer = Trainer(experiment)
        results = trainer.run(run_tests=run_tests)
        if not run_tests:
            print(f"Training finished without tests. Results: {trainer.results_path}")
        if write_report:
            if run_tests:
                reporter_type(results).render(experiment.options.output)
                summary = experiment.options.output / "RESULTS.md"
                print("\n" + summary.read_text(encoding="utf-8"))
                print(f"Figures and RESULTS.md: {experiment.options.output}")
            else:
                print("No report: the figures and RESULTS.md need the test sweep that --no-test skipped.")

    @staticmethod
    def report(experiment_type: Type[Experiment], reporter_type: Type[Reporter],
               results_file: Optional[Path]) -> None:
        """
        Write figures and RESULTS.md next to an existing results JSON.

        :param experiment_type: the experiment that produced the results
        :param reporter_type: the reporter for those results
        :param results_file: the results JSON (default: the experiment's stored results).
            A file named like another experiment's results file is reported with that
            experiment's reporter.
        """
        results_file = results_file or experiment_type.RESULTS_DIR / experiment_type.RESULTS_FILE
        for other_experiment, other_reporter in Main.EXPERIMENTS.values():
            if results_file.name == other_experiment.RESULTS_FILE:
                reporter_type = other_reporter
        reporter_type.from_file(results_file).render(results_file.parent)


if __name__ == "__main__":
    Main().run()
