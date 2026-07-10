import torch
from torch import nn
from torch.nn import functional as F


def out_size(input_size: int = 2048, k: int = 3, s: int = 1, p: int = 0):
    """
    Compute the output size of the given input size, kernel_size, stride, and padding.
    """
    return int((input_size + 2 * p - k) / s) + 1


class ComplexConv1DLayer(nn.Module):
    """True Complex-Valued 1D Convolution using native PyTorch complex math."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ComplexConv1DLayer, self).__init__()
        self.stride = stride
        self.padding = padding

        # Initialize complex weights natively: [out_channels, in_channels, kernel_size]
        r_w = torch.randn(out_channels, in_channels, kernel_size) * 0.05
        i_w = torch.randn(out_channels, in_channels, kernel_size) * 0.05
        self.weight = nn.Parameter(torch.complex(r_w, i_w))

        # Initialize complex bias
        self.bias = nn.Parameter(torch.complex(torch.zeros(out_channels), torch.zeros(out_channels)))

    def forward(self, x):
        return F.conv1d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)


class NativeComplexLinear(nn.Module):
    """True Complex-Valued Dense Linear Layer using pure PyTorch complex math."""

    def __init__(self, in_features, out_features):
        super(NativeComplexLinear, self).__init__()
        r_w = torch.randn(out_features, in_features) * 0.01
        i_w = torch.randn(out_features, in_features) * 0.01
        self.weight = nn.Parameter(torch.complex(r_w, i_w))
        self.bias = nn.Parameter(torch.complex(torch.zeros(out_features), torch.zeros(out_features)))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class PssNnDetector(nn.Module):
    """Fixed Convolutional and Fully Connected Network for Complex RF Signals."""

    def __init__(self):
        super(PssNnDetector, self).__init__()
        kernel_size = 5
        stride = 2
        padding = 0  # Explicitly tracking your default layer padding

        n_kernels_1 = 64
        n_kernels_2 = 32
        n_kernels_3 = 4

        # 3 True Complex Conv Layers
        self.conv1 = ComplexConv1DLayer(in_channels=1, out_channels=n_kernels_1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = ComplexConv1DLayer(in_channels=n_kernels_1, out_channels=n_kernels_2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = ComplexConv1DLayer(in_channels=n_kernels_2, out_channels=n_kernels_3, kernel_size=kernel_size, stride=stride, padding=padding)

        # Track accurate output sequence size through all 3 stages sequentially (Length = 253)
        l1 = out_size(2048, k=kernel_size, s=stride, p=padding)
        l2 = out_size(l1, k=kernel_size, s=stride, p=padding)
        l3 = out_size(l2, k=kernel_size, s=stride, p=padding)

        # Flatten size calculation: 8 channels * 253 length = 2024 features
        flattened_features = n_kernels_3 * l3

        # FC1 keeps complex representations intact
        self.fc1 = NativeComplexLinear(flattened_features, 64)

        # FC2 maps the concatenated real/imag components (128 * 2 = 256) safely to 3 output categories
        self.fc2 = nn.Linear(in_features=64 * 2, out_features=3)

    @staticmethod
    def complex_leaky_relu(x):
        """Applies Leaky ReLU to real and imaginary components separately to keep phase intact."""
        return torch.complex(F.leaky_relu(x.real, 0.01), F.leaky_relu(x.imag, 0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute a forward pass.
        :param x: Native complex tensor [Batch, 2048] with dtype=torch.complex64
        """
        x = x.unsqueeze(1)

        # Convolution sequence
        x = self.complex_leaky_relu(self.conv1(x))
        x = self.complex_leaky_relu(self.conv2(x))
        x = self.complex_leaky_relu(self.conv3(x))

        # Flatten step: Reshapes tensor to [Batch, 2024]
        x = x.view(x.size(0), -1)

        # Dense Layer 1 (Outputs Complex array: [Batch, 128])
        x = self.complex_leaky_relu(self.fc1(x))

        # Fix 3: Combine Real & Imaginary features into a flat real vector [Batch, 256]
        # to prevent data-type casting failures during final classification output
        x_real_features = torch.cat([x.real, x.imag], dim=1)

        # Final Real Linear Output layer mapping down to [Batch, 3]
        cell_id_score = self.fc2(x_real_features)

        return cell_id_score


