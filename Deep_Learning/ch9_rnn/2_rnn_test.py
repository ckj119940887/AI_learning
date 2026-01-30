import torch
import torch.nn as nn

rnn = nn.RNN(input_size=8, hidden_size=16, num_layers=2)

# define data (L=1, N=3, H=8)
input = torch.rand(1, 3, 8)
hx = torch.randn(2, 3, 16)

#
output, hidden = rnn(input, hx)
print(output.size())
print(hidden.size())
