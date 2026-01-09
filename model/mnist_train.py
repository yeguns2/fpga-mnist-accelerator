import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# =============================================================================
# SmallCNN
# -----------------------------------------------------------------------------
# A compact convolutional neural network designed specifically for MNIST.
#
# Architecture summary:
#   Input        : 1 x 28 x 28 (grayscale MNIST image)
#   Conv1        : 1 -> 8 channels, 3x3 kernel, stride 1, padding 1
#   ReLU
#   MaxPool     : 2x2 (spatial downsampling by factor of 2)
#   Conv2        : 8 -> 16 channels, 3x3 kernel, stride 1, padding 1
#   ReLU
#   MaxPool     : 2x2
#   Flatten
#   Fully-Connected: 16 * 7 * 7 -> 10 (digit classes 0–9)
#
# =============================================================================
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # ---------------------------------------------------------------------
        # First convolution layer
        # Input  : [B, 1, 28, 28]
        # Output : [B, 8, 28, 28]
        # Padding=1 keeps spatial resolution unchanged
        # ---------------------------------------------------------------------
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)

        # ---------------------------------------------------------------------
        # Second convolution layer
        # Input  : [B, 8, 14, 14] (after first max-pool)
        # Output : [B, 16, 14, 14]
        # Padding=1 keeps spatial resolution unchanged before pooling
        # ---------------------------------------------------------------------
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)

        # ---------------------------------------------------------------------
        # Activation function
        # ReLU is chosen for simplicity and hardware efficiency
        # ---------------------------------------------------------------------
        self.relu = nn.ReLU()

        # ---------------------------------------------------------------------
        # Max pooling layer
        # Kernel size 2, stride 2
        # Reduces spatial resolution by factor of 2
        # ---------------------------------------------------------------------
        self.pool = nn.MaxPool2d(2, 2)

        # ---------------------------------------------------------------------
        # Fully connected layer
        # After two pooling operations:
        #   28x28 -> 14x14 -> 7x7
        # Feature map shape before FC: [B, 16, 7, 7]
        # Flattened size: 16 * 7 * 7 = 784
        # Output classes: 10 (digits 0–9)
        # ---------------------------------------------------------------------
        self.fc = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        """
        Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, 1, 28, 28]

        Returns
        -------
        torch.Tensor
            Logits tensor of shape [B, 10]
        """

        # Conv1 -> ReLU -> MaxPool
        # [B, 1, 28, 28] -> [B, 8, 28, 28] -> [B, 8, 14, 14]
        x = self.pool(self.relu(self.conv1(x)))

        # Conv2 -> ReLU -> MaxPool
        # [B, 8, 14, 14] -> [B, 16, 14, 14] -> [B, 16, 7, 7]
        x = self.pool(self.relu(self.conv2(x)))

        # Flatten spatial and channel dimensions
        # [B, 16, 7, 7] -> [B, 16*7*7]
        x = x.reshape(x.size(0), -1)

        # Fully connected classification layer
        # [B, 16*7*7] -> [B, 10]
        x = self.fc(x)

        return x


# =============================================================================
# Training and Evaluation Script
# -----------------------------------------------------------------------------
# Trains SmallCNN on MNIST using:
#   - Adam optimizer
#   - Cross-entropy loss
#   - CPU execution (deterministic, hardware-export friendly)
#
# Outputs:
#   - Prints training loss and test accuracy per epoch
#   - Saves trained weights to "mnist_smallcnn_state_dict.pt"
# =============================================================================
def main():
    # -------------------------------------------------------------------------
    # Use CPU explicitly to avoid nondeterminism from GPU kernels
    # -------------------------------------------------------------------------
    device = torch.device("cpu")

    # Fix random seed for reproducibility
    torch.manual_seed(0)

    # -------------------------------------------------------------------------
    # Dataset and preprocessing
    # - MNIST images are converted to tensors in range [0, 1]
    # - No normalization beyond ToTensor() is applied
    # -------------------------------------------------------------------------
    transform = transforms.ToTensor()

    train_ds = datasets.MNIST(
        root="./data",
        train=True,
        download=False,
        transform=transform
    )

    test_ds = datasets.MNIST(
        root="./data",
        train=False,
        download=False,
        transform=transform
    )

    # -------------------------------------------------------------------------
    # Data loaders
    # - Training: smaller batch size, shuffled
    # - Testing : larger batch size, no shuffle
    # -------------------------------------------------------------------------
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=64, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=256, shuffle=False
    )

    # -------------------------------------------------------------------------
    # Model, loss, optimizer
    # -------------------------------------------------------------------------
    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # -------------------------------------------------------------------------
    # Training loop
    # Epoch count kept small for fast iteration and experimentation
    # -------------------------------------------------------------------------
    epochs = 10
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        # -------------------------
        # Training phase
        # -------------------------
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # -------------------------
        # Evaluation phase
        # -------------------------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()

        acc = 100.0 * correct / total
        print(
            f"Epoch {ep}: "
            f"train_loss={total_loss/len(train_loader):.4f}  "
            f"test_acc={acc:.2f}%"
        )

    # -------------------------------------------------------------------------
    # Save trained model parameters
    # This file is later consumed by export_weight.py for HW deployment
    # -------------------------------------------------------------------------
    torch.save(model.state_dict(), "mnist_smallcnn_state_dict.pt")
    print("Saved: mnist_smallcnn_state_dict.pt")


if __name__ == "__main__":
    main()
