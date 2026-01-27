import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # define linear layer
        self.linear1 = nn.Linear(3, 4, device=device)
        nn.init.xavier_normal_(self.linear1.weight)
        self.linear2 = nn.Linear(4, 4, device=device)
        nn.init.kaiming_normal_(self.linear2.weight)
        self.out = nn.Linear(4, 2, device=device)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.out(x)
        x = torch.softmax(x, dim=1)
        return x


# define dataset
x = torch.randn(10, 3, device=device)

# define model
model = Model()
output = model(x)
print(output)

# 查看模型参数
# method1
for name,param in model.named_parameters():
    print(name, param)
    print()

print("------------------")

# method2
print(model.state_dict())

print("------------------")

from torchsummary import summary
summary(model, input_size=(3,), batch_size=10, device=device)