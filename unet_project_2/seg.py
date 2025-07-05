import torch
from typing import List


class SegmentationMetrics:
    """
    simpler approach to compute segmentation metrics
    """

    def __init__(self, num_classes: int, class_names: List[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()

    def reset(self):
        """reset internal state"""
        self.total_pixels = torch.zeros(self.num_classes)
        self.correct_pixels = torch.zeros(self.num_classes)
        self.union_pixels = torch.zeros(self.num_classes)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update internal state with preds and targets
        both preds and targets has shape [B, C, H, W] and one-hot encoded
        """
        assert (
            preds.shape == targets.shape
        ), "Predictions and targets must have the same shape"
        preds = preds.argmax(dim=1)  # convert to class indices & one-hot encode
        preds = torch.nn.functional.one_hot(
            preds, num_classes=self.num_classes
        ).permute(
            0, 3, 1, 2
        )  # [B, H, W, C] to [B, C, H, W]
        self.total_pixels += torch.sum(targets, dim=(0, 2, 3))
        self.correct_pixels += torch.sum(preds * targets, dim=(0, 2, 3))
        self.union_pixels += torch.sum((preds + targets) > 0, dim=(0, 2, 3))

    def compute(self):
        """
        Compute metrics: pixel accuracy and IoU with current state
        """
        # class-wise pixel accuracy
        pixel_accuracy = [
            correct / total if total > 0 else 0
            for correct, total in zip(self.correct_pixels, self.total_pixels)
        ]
        # total pixel accuracy
        pixel_accuracy.append(
            sum(self.correct_pixels) / sum(self.total_pixels)
            if sum(self.total_pixels) > 0
            else 0
        )
        # class-wise IoU
        iou = [
            correct / union if union > 0 else 0
            for correct, union in zip(self.correct_pixels, self.union_pixels)
        ]
        # total IoU
        iou.append(
            sum(self.correct_pixels) / sum(self.union_pixels)
            if sum(self.union_pixels) > 0
            else 0
        )

        return {
            "pixel_accuracy": pixel_accuracy,
            "iou": iou,
        }

    def get_summary_string(self):
        """
        Get a summary string of the metrics for logging
        """
        metrics = self.compute()

        summary = []
        summary.append(
            f"Pixel Accuracy: {100*metrics['pixel_accuracy'][self.num_classes]:.2f}%"
        )
        summary.append(f"Mean IoU: {100*metrics['iou'][self.num_classes]:.2f}%")

        for i, class_name in enumerate(self.class_names):
            iou = metrics["iou"][i]
            acc = metrics["pixel_accuracy"][i]
            summary.append(f"{class_name}: IoU={100*iou:.2f}%, Acc={100*acc:.2f}%")

        return " | ".join(summary)


if __name__ == "__main__":
    # Example usage
    metrics = SegmentationMetrics(
        num_classes=3, class_names=["background", "unknown", "foreground"]
    )
    # Simulate some predictions and targets
    preds = torch.randint(0, 2, (1, 3, 256, 256))  # Random binary predictions
    targets = torch.randint(0, 2, (1, 3, 256, 256))  # Random binary targets
    metrics.update(preds, targets)
    print(metrics.get_summary_string())
