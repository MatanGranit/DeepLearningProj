import os
import json
from dataclasses import dataclass
from pathlib import Path

OUTPUT_DIR = os.path.join(Path(__file__).resolve().parent.parent, 'out')
CHECKPOINT_DIR = os.path.join(Path(__file__).resolve().parent.parent, 'checkpoints')
for d in [OUTPUT_DIR, CHECKPOINT_DIR]:
    if not os.path.isdir(d):
        os.makedirs(d)

@dataclass
class LoggingParameters:
    """Data class holding parameters for logging."""
    model_name: str
    optimizer_name: str
    optimizer_params: dict


def write_output(logging_parameters: LoggingParameters, data: dict):
    """
    Write logs to JSON.
    :param logging_parameters: Logging parameters. Some parameters to log.
    :param data: dict. Holding a dictionary to dump to the output JSON.
    """
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    output_filename = f"{logging_parameters.model_name}_{logging_parameters.optimizer_name}.json"
    output_filepath = os.path.join(OUTPUT_DIR, output_filename)

    print(f"Writing output to {output_filepath}")
    # Load output file
    try:
        with open(output_filepath, 'r', encoding='utf-8') as f:
            all_output_data = json.load(f)
    except:
        all_output_data = []

    # Add new data and write to file
    all_output_data.append(data)
    # pylint: disable=C0103
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(all_output_data, f, indent=4)