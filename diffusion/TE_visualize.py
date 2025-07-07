import torch
import torch.nn as nn
from torch import Tensor

import matplotlib.pyplot as plt
import math


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


if __name__ == "__main__":
    # Example usage
    t = torch.tensor([i for i in range(32)], dtype=torch.float32)  # Example time steps
    time_embedding = TimeEmbedding(dim=4)
    emb = time_embedding(t)
    print(emb.shape)  # Should print: torch.Size([5, 16])
    print(emb)  # Print the time embeddings

    # Example visualization
    plt.figure(figsize=(10, 5))
    plt.plot(t.numpy(), emb.numpy())
    plt.title("Time Embeddings")
    plt.xlabel("Time Steps")
    plt.ylabel("Embedding Values")
    plt.legend([f"Dim {i}" for i in range(emb.shape[1])])
    plt.grid()
    plt.show()
