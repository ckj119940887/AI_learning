import torch
import torch.nn as nn
from torchsummary import summary

x = torch.randn(10, 3, device='cpu')

model = nn.Sequential(
    nn.Linear(3, 4),
    nn.Tanh(),
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 2),
    nn.Softmax(dim=1),
)

# model param init
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.01)

model.apply(init_weights)

output = model(x)
print(output)

summary(model, input_size=(3,), batch_size=10, device='cpu')