import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from unet import create_unet


class SynthDataset(Dataset):
    """Dataset for synthetic MNIST+CIFAR data"""

    def __init__(self, data_dir="data/synth", num_samples=None):
        self.data_dir = data_dir

        # Get all data files
        data_files = [
            f
            for f in os.listdir(data_dir)
            if f.startswith("data_") and f.endswith(".png")
        ]
        segm_files = [
            f
            for f in os.listdir(data_dir)
            if f.startswith("segm_") and f.endswith(".png")
        ]

        # Sort to ensure matching pairs
        data_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        segm_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

        self.data_files = data_files
        self.segm_files = segm_files

        if num_samples is not None:
            self.data_files = self.data_files[:num_samples]
            self.segm_files = self.segm_files[:num_samples]

        print(f"Found {len(self.data_files)} data samples")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        # Load data image (RGB)
        data_path = os.path.join(self.data_dir, self.data_files[idx])
        image = Image.open(data_path).convert("RGB")
        image = np.array(image)

        # Load segmentation mask (grayscale)
        segm_path = os.path.join(self.data_dir, self.segm_files[idx])
        mask = Image.open(segm_path).convert("L")
        mask = np.array(mask)

        # Convert to tensors and normalize
        # Image: (H, W, C) -> (C, H, W), normalize to [-1, 1]
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = (image - 0.5) / 0.5  # Normalize to [-1, 1]

        # Mask: binary segmentation (0 or 1)
        mask = torch.from_numpy(mask).long()
        mask = (mask > 127).long()  # Convert to binary (0 or 1)

        return image, mask


class DummyDataset(Dataset):
    """Dummy dataset for demonstration purposes"""

    def __init__(self, num_samples=100, image_size=256):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random image (grayscale)
        image = torch.randn(1, self.image_size, self.image_size)

        # Generate random binary mask for segmentation
        mask = torch.randint(0, 2, (self.image_size, self.image_size), dtype=torch.long)

        return image, mask


def dice_loss(pred, target, smooth=1):
    """Dice loss for segmentation"""
    pred = torch.sigmoid(pred)
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = 1 - (
        (2.0 * intersection + smooth)
        / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    )

    return loss.mean()


def combined_loss(pred, target):
    """Combined Cross Entropy and Dice loss"""
    ce_loss = nn.CrossEntropyLoss()(pred, target)
    dc_loss = dice_loss(pred[:, 1:, :, :], target.unsqueeze(1).float())
    return ce_loss + dc_loss


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Calculate loss
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total_pixels = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            # Calculate pixel accuracy
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total_pixels += target.numel()

    accuracy = 100.0 * correct / total_pixels
    avg_loss = total_loss / len(dataloader)

    return avg_loss, accuracy


def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 8
    learning_rate = 1e-4
    num_epochs = 3
    image_size = 32

    # Create datasets using synthetic data
    try:
        full_dataset = SynthDataset(data_dir="data/synth")

        # Split dataset into train and validation
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

    except Exception as e:
        print(f"Error loading synthetic dataset: {e}")
        print("Falling back to dummy dataset...")
        train_dataset = DummyDataset(num_samples=80, image_size=image_size)
        val_dataset = DummyDataset(num_samples=20, image_size=image_size)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model - 3 input channels (RGB), 2 output classes (background, foreground)
    model = create_unet(n_channels=3, n_classes=2)
    model = model.to(device)

    # Loss function and optimizer
    criterion = combined_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    print(
        f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters"
    )

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_unet_model.pth")
            print("New best model saved!")

    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


def test_model():
    """Test the trained model with a single sample"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model - updated for 3 channel input
    model = create_unet(n_channels=3, n_classes=2)
    model = model.to(device)

    try:
        model.load_state_dict(torch.load("best_unet_model.pth", map_location=device))
        print("Loaded trained model")
    except FileNotFoundError:
        print("No trained model found, using random weights")

    model.eval()

    # Test with synthetic data if available
    try:
        test_dataset = SynthDataset(data_dir="data/synth", num_samples=1)
        test_image, test_mask = test_dataset[0]
        test_image = test_image.unsqueeze(0).to(device)  # Add batch dimension
        test_mask = test_mask.to(device)
        print("Using real synthetic data for testing")
    except Exception as e:
        print(f"Could not load synthetic data: {e}")
        # Create a dummy test sample with 3 channels (RGB) and 32x32 size
        test_image = torch.randn(1, 3, 32, 32).to(device)
        test_mask = None
        print("Using random test data")

    with torch.no_grad():
        output = model(test_image)
        prediction = torch.softmax(output, dim=1)
        segmentation_map = output.argmax(dim=1)

    print(f"Test image shape: {test_image.shape}")
    print(f"Model output shape: {output.shape}")
    print(f"Prediction probabilities shape: {prediction.shape}")
    print(f"Segmentation map shape: {segmentation_map.shape}")
    print(f"Unique values in segmentation: {torch.unique(segmentation_map)}")

    if test_mask is not None:
        print(f"Ground truth mask shape: {test_mask.shape}")
        print(f"Unique values in ground truth: {torch.unique(test_mask)}")

        # Calculate accuracy for this test sample
        correct = (segmentation_map.squeeze() == test_mask).sum().item()
        total = test_mask.numel()
        accuracy = 100.0 * correct / total
        print(f"Test accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    print("=== U-Net Training Demo ===")
    print("1. Training model...")
    main()

    print("\n=== Model Testing ===")
    print("2. Testing model...")
    test_model()
