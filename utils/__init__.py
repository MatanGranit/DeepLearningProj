"""PSS detection library: datasets, models, experiments, training, evaluation and reporting.

Layout, in the order a run uses it:

* :mod:`.data.pss_signal`    -- :class:`PssSignalGenerator`: PSS sequences, AWGN, frequency shift
* :mod:`.data.pss_dataset`   -- :class:`PssDataset`: four-class windows and pooled-SNR loaders
* :mod:`.models.receiver`    -- :class:`Receiver`: the interface every detector implements
* :mod:`.models.networks`    -- :class:`NarrowbandPssCnn`, :class:`ConvScanningPssCnn`
* :mod:`.models.correlator`  -- :class:`Correlator`: magnitude and coherent baselines
* :mod:`.experiments`        -- :class:`KnownBinExperiment`, :class:`UnknownBinExperiment`:
  what each experiment trains and how it is seeded and stored
* :mod:`.trainer`            -- :class:`Trainer`: datasets, training, validation, tests, checkpoints
* :mod:`.evaluation`         -- :class:`Evaluator`: thresholds, test metrics, arithmetic cost
* :mod:`.metrics`            -- :class:`DetectionMetrics`: confusion matrix to accuracy, Pd, Pfa
* :mod:`.reporting`          -- :class:`KnownBinReporter`, :class:`UnknownBinReporter`:
  figures and RESULTS.md from a results JSON

``main.py`` at the project root is the entry point.
"""
