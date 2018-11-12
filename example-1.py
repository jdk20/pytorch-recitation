import torch
import torch.nn as nn
import torch.nn.functional as F

class CaffeNet(nn.Module):
    def __init__(self, N):
        super().__init__()

        self.N = N
        self.conv_1 = nn.Conv2d(C, 96, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        self.d_1 = nn.Dropout2d(p=0.5)
        self.mp_1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv_2 = nn.Conv2d(96, 256, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.conv_3 = nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.conv_4 = nn.Conv2d(384, 348, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.fc_1 = nn.Linear(348 * 21 * 21, 10)

    def forward(self, inputs):
        inputs = self.mp_1(F.relu(self.d_1(self.conv_1(inputs))))
        inputs = F.relu(self.conv_2(inputs))
        inputs = F.relu(self.conv_3(inputs))
        inputs = F.relu(self.conv_4(inputs))
        inputs = inputs.view(self.N, -1)
        inputs = self.fc_1(inputs)

        return inputs


N = 20
C = 3  # color
H = 224  # pixels
W = 224  # pixels
n_iterations = 100
inputs = torch.rand(N, C, H, W)

targets = torch.randint(10, (N, 1))
targets = targets[:, 0]
targets = targets.long()

# Model
model = CaffeNet(N)  # Arch
optimizer = torch.optim.Adam(model.parameters())  # Optimizer
loss_function = nn.CrossEntropyLoss()  # Loss function

for i in range(n_iterations):
    # Optimizer
    optimizer.zero_grad()

    # Forward prop
    outputs = model(inputs)

    # Backprop
    loss = loss_function(outputs, targets)
    loss.backward()
    optimizer.step()

    print('iteration:', i, 'loss:', loss.detach().numpy())
