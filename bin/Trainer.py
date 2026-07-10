import os

import torch
from torch import optim, nn
from torch._C import device
from torch.utils.data import DataLoader

from utils.DataLoader import PssDataset
from utils.PssNnDetector import PssNnDetector
from utils.common_functions import write_output, CHECKPOINT_DIR, LoggingParameters
from utils.correlator import PssCorrelator



class Trainer:
    """
    class used to train the model
    """
    def __init__(self, optimizer: optim.Optimizer, model: PssNnDetector,
                 dev: device, batch_size: int=32, n_epoch: int=20):
        self.correlator = PssCorrelator().to(dev).eval()
        self.model = model.to(dev)
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.device = dev
        self.current_snr = [30, 0]

    def _current_snr(self, epoch: int = 1) -> float:
        """
        compute current snr value for given epoch.
        :param epoch: epoch number
        :return: current_snr: current snr for given epoch
        """
        max_snr = 30
        min_snr = 0
        snr_diff_between_epochs = (max_snr - min_snr) / self.n_epoch
        current_snr = 30 - snr_diff_between_epochs * epoch
        return current_snr

    def _train_one_epoch(self, epoch: int = 1) -> tuple[float, float]:
        """
        Train the model for a single epoch on the training dataset.
        :param epoch: epoch number
        :return: (avg_loss, accuracy): tuple containing the average loss and accuracy across all dataset samples.
        """
        # self.model.train()
        total_loss = 0
        avg_loss = 0
        accuracy = 0
        nof_samples = 0
        correct_labeled_samples = 0

        self.train_dataset = PssDataset(num_samples=int(1e4), snr_db=self.current_snr)

        train_dataloader = DataLoader(self.train_dataset, self.batch_size, shuffle=True)
        print_every = int(len(train_dataloader) / 20)

        for batch_idx, (inputs, labels, _) in enumerate(train_dataloader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # zero the gradient
            cur_nof_samples = labels.shape[0]

            # zero gradient
            self.model.zero_grad()

            # compute forward pass
            model_output = self.model(inputs)
            classification = model_output.argmax(axis=-1)

            # compute loss
            loss = self.criterion(model_output, labels)
            total_loss += float(loss.item())

            # compute the backward pass
            loss.backward()

            # step the optimizer
            self.optimizer.step()

            # update average loss and accuracy
            nof_samples += cur_nof_samples
            correct_labeled_samples += int(torch.sum(classification == labels))
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = correct_labeled_samples * 100 / nof_samples

            # if batch_idx % print_every == 0 or \
            #         batch_idx == len(train_dataloader) - 1:
        print(f'Epoch [{epoch:03d}] | Loss: {avg_loss:.3f} | '
              f'Acc: {accuracy:.2f}[%] ')

        return avg_loss, accuracy

    def _validate(self, epoch: int = 1, use_correlator: bool = False) -> tuple[float, float]:
        """
        Evaluate model loss and accuracy for dataset.
        :param epoch: epoch number
        :param use_correlator: use correlation or not
        :return: (avg_loss, accuracy): tuple containing the average loss and accuracy across all dataset samples.
        """
        if use_correlator:
            model = self.correlator
        else:
            model = self.model
        model.eval()
        self.valid_dataset = PssDataset(num_samples=int(500), snr_db=self.current_snr)
        dataloader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)
        total_loss = 0
        nof_samples = 0
        correct_labeled_samples = 0
        n_batches = len(dataloader)

        for batch_idx, (inputs, labels, _) in enumerate(dataloader):
            cur_nof_samples = labels.shape[0]

            # no gradient
            with torch.no_grad():
                # compute forward pass
                model_output = model(inputs)
                classification = model_output.argmax(axis=-1)

                # compute loss
                loss = self.criterion(model_output, labels)
            total_loss += float(loss.item())

            nof_samples += cur_nof_samples
            correct_labeled_samples += int(torch.sum(classification == labels))

        avg_loss = total_loss / (n_batches + 1)
        accuracy = correct_labeled_samples * 100 / nof_samples
        print(f'Epoch [{epoch:03d}] | Loss: {avg_loss:.3f} | '
              f'Acc: {accuracy:.2f}[%] '
              f'({correct_labeled_samples}/{nof_samples})')

        return avg_loss, accuracy

    def run(self, logging_parameters: LoggingParameters) -> None:
        """
        Train and evaluate model on dataset, finally log results.
        :param logging_parameters: logging parameters
        """
        output_data = {
            "model": logging_parameters.model_name,
            "optimizer": {
                "name": logging_parameters.optimizer_name,
            },
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        best_acc = 0
        model_filename = f"{logging_parameters.model_name}_{logging_parameters.optimizer_name}.pt"
        checkpoint_filename = os.path.join(CHECKPOINT_DIR, model_filename)
        for epoch in range(1, self.n_epoch + 1):
            print(f'Epoch {epoch}/{self.n_epoch}')

            print("Starting Train")
            train_loss, train_acc = self._train_one_epoch(epoch=epoch)
            print("Starting Validation")
            val_loss, val_acc = self._validate(epoch=epoch)

            output_data["train_loss"].append(train_loss)
            output_data["train_acc"].append(train_acc)
            output_data["val_loss"].append(val_loss)
            output_data["val_acc"].append(val_acc)

            # Save checkpoint
            if val_acc > best_acc:
                print(f'Saving checkpoint {checkpoint_filename}')
                state = {
                    'model': self.model.state_dict(),
                    'val_acc': val_acc,
                    'epoch': epoch,
                }
                torch.save(state, checkpoint_filename)
                best_acc = val_acc

            if val_acc == 1 and val_loss < 1e-8:
                break

        write_output(logging_parameters, output_data)


    def test(self, logging_parameters: LoggingParameters) -> None:
        """
        test model on dataset, check results per SNR value vs. the correlator.
        :param logging_parameters: logging parameters
        """
        output_data = {
            "model": logging_parameters.model_name,
            "optimizer": {
                "name": logging_parameters.optimizer_name,
                # "params": logging_parameters.optimizer_params,
            },
            "model_loss": [],
            "model_acc": [],
            "correlator_loss": [],
            "correlator_acc": [],
            'snr': [],
        }

        # define models and load the weights
        model_filename = f"{logging_parameters.model_name}_{logging_parameters.optimizer_name}.pt"
        self.model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, model_filename)))
        for snr in torch.arange(-10 ,30, 0.25):
            self.current_snr = snr
            mode_loss, model_acc = self._validate()
            output_data["model_loss"].append(mode_loss)
            output_data["model_acc"].append(test_acc)

            mode_loss, model_acc = self._validate(use_correlator=True)
            output_data["correlator_loss"].append(mode_loss)
            output_data["correlator_loss"].append(test_acc)

            output_data['snr'].append(snr.item())


        write_output(logging_parameters, output_data)