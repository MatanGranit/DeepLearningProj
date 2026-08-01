import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math

from data_loader import PssDataset
from correlator import Correlator


import torch
import torch.nn as nn

class EfficientPssCnn(nn.Module):
    def __init__(self, fft_binsize: int = 2048):
        super().__init__()
        self.fft_binsize = fft_binsize
        
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=12, stride=8, padding=2),
            nn.LeakyReLU(0.1),
            
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=12, stride=8, padding=2),
            nn.LeakyReLU(0.1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=8 * 32, out_features=4) 
        )

    def forward(self, x):
        x_real_imag = torch.view_as_real(x)
        x_cnn_input = x_real_imag.permute(0, 2, 1)
        
        features = self.features(x_cnn_input)
        output = self.classifier(features)
        
        return output

class UltraFastPssClassifier(nn.Module):
    def __init__(self, fft_binsize: int = 2048, stride: int = 2):
        super().__init__()
        self.fft_binsize = fft_binsize
        self.stride = stride

        reduced_len = fft_binsize // stride
        # Updated to 4 classes to match the dataset (Noise class included)
        self.fc = nn.Linear(2 * reduced_len, 4)

    def forward(self, x):
        x = x[:, ::self.stride]
        x = torch.view_as_real(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)


def correct_frequency_offset(samples, freq_offset, n):
    exponent = torch.exp(
        2j * math.pi * torch.outer(freq_offset, n / samples.shape[1])
    )
    return samples * torch.conj(exponent)


def train_model(model: nn.Module, dataloader: DataLoader, epochs: int = 5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = next(model.parameters()).device
    n = torch.arange(2048, device=device)

    print("Starting training phase...")

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for samples, labels_hot_ones, freq_offset in dataloader:
            samples = samples.to(device)
            labels_hot_ones = labels_hot_ones.to(device)
            freq_offset = freq_offset.to(device)

            optimizer.zero_grad()

            samples_corrected = correct_frequency_offset(samples, freq_offset, n)

            labels = torch.argmax(labels_hot_ones, dim=1)
            outputs = model(samples_corrected)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct_predictions / total_samples

        print(
            f"Epoch {epoch + 1} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Train Accuracy: {epoch_acc:.2f}%"
        )


def evaluate_models(model: nn.Module, correlator, test_dataloader: DataLoader):
    model.eval()

    device = next(model.parameters()).device
    n = torch.arange(2048, device=device)

    total_samples = 0
    nn_correct = 0
    corr_correct = 0

    print("\nStarting evaluation on test set...")

    with torch.no_grad():
        for samples, labels_hot_ones, freq_offset in test_dataloader:
            samples = samples.to(device)
            labels_hot_ones = labels_hot_ones.to(device)
            freq_offset = freq_offset.to(device)

            true_labels = torch.argmax(labels_hot_ones, dim=1)

            samples_corrected = correct_frequency_offset(samples, freq_offset, n)

            nn_outputs = model(samples_corrected)
            nn_predicted = torch.argmax(nn_outputs, dim=1)
            nn_correct += (nn_predicted == true_labels).sum().item()

            # The correlator threshold logic goes here. (Assuming correlator uses standard correlation)
            corr_outputs = correlator.find_cell_id(freq_offset.cpu(), samples.cpu())
            corr_predicted = torch.argmax(corr_outputs, dim=1).to(device)
            corr_correct += (corr_predicted == true_labels).sum().item()

            total_samples += true_labels.size(0)

    print("-" * 30)
    print("Evaluation Results:")
    print(f"Total Test Samples: {total_samples}")
    print(f"Neural Network Accuracy: {100 * nn_correct / total_samples:.2f}%")
    print(f"Classic Correlator Accuracy: {100 * corr_correct / total_samples:.2f}%")
    print("-" * 30)


if __name__ == "__main__":
    train_dataset = DataLoader(
        PssDataset(num_samples=5000, noise_db=10.0),
        batch_size=32,
        shuffle=True
    )

    test_dataset = DataLoader(
        PssDataset(num_samples=2000, noise_db=10.0),
        batch_size=32,
        shuffle=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use the new CNN model
    net = EfficientPssCnn().to(device)

    # Then test this for speed (if needed)
    # net = UltraFastPssClassifier(stride=2).to(device)
    # net = UltraFastPssClassifier(stride=4).to(device)

    corr = Correlator()

    train_model(net, train_dataset, epochs=5)
    evaluate_models(net, corr, test_dataset)

    # Save the updated model weights
    torch.save(net.state_dict(), "pss_cnn_model.pth")
    print("Model weights saved to pss_cnn_model.pth")