import torch
import torch.nn as nn

a = torch.tensor([[0, 1], [2, 3]], dtype=torch.float32)
b = torch.tensor([[4, 1], [2, 3]], dtype=torch.float32)

# Fix 1: Correct tensor shapes
# Input should be (batch_size, in_channels, height, width)
input_tensor = a.view(1, 1, 2, 2)  # batch=1, channels=1, height=2, width=2

# Weight should be (in_channels, out_channels, kernel_height, kernel_width)
weight_tensor = b.view(1, 1, 2, 2)  # in_ch=1, out_ch=1, kernel=2x2

print("Input shape:", input_tensor.shape)
print("Weight shape:", weight_tensor.shape)
print("Result:")
print(torch.conv_transpose2d(input=input_tensor, weight=weight_tensor))

# Alternative: Using nn.ConvTranspose2d (more common in practice)
print("\nUsing nn.ConvTranspose2d:")
conv_transpose = nn.ConvTranspose2d(
    in_channels=1, out_channels=1, kernel_size=2, bias=False
)
# Set the weight manually
conv_transpose.weight.data = weight_tensor
result = conv_transpose(input_tensor)
print("Result shape:", result.shape)
print("Result:")
print(result)
