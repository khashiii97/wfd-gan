from model import DF, MyConv1dPadSame, MyMaxPool1dPadSame
from torchsummaryX import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
# df_model = DF(length= 1400)

# summary(df_model, torch.zeros((32,1400)))


class TestLayer(nn.Module):
    def __init__(self):
        super(TestLayer, self).__init__()
        self.layer1 = nn.Sequential(
            MyConv1dPadSame(1, 32, 8, 1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            MyConv1dPadSame(32, 32, 8, 1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            MyMaxPool1dPadSame(8, 1),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.layer1(x)

# Initialize the test layer
test_layer = TestLayer()

# Create a random input tensor of shape (batch_size, channels, length)
# For example, a single-channel input vector of length 1000 in a batch of size 1
input_tensor = torch.rand(1, 1, 1000)

# Apply the test layer to the input tensor
output_tensor = test_layer(input_tensor)

# Print the shape of the output tensor
print("Output shape:", output_tensor.shape)

