import torch
from torch import optim

from bin.Trainer import Trainer
from utils.PssNnDetector import PssNnDetector
from utils.common_functions import LoggingParameters

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PssNnDetector()
    optimizer = optim.Adam(model.parameters())

    logging_parameters = LoggingParameters(model_name='PssNnDetector',
                                           optimizer_name='Adam',
                                           optimizer_params=optimizer.param_groups[0].copy())
    trainer = Trainer(optimizer=optimizer, model=model, dev=device)
    trainer.run(logging_parameters=logging_parameters)
    trainer.test(logging_parameters=logging_parameters)


if __name__ == "__main__":
    main()