import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Layer 1
        self.conv1 = nn.Conv2d(3, 48, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        self.mp1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.bn1 = nn.BatchNorm2d(48)
        # Layer 2
        self.conv2 = nn.Conv2d(48, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.mp2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.bn2 = nn.BatchNorm2d(128)
        # Layer 3
        self.conv3 = nn.Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # Layer 4
        self.conv4 = nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # Layer 5
        self.conv5 = nn.Conv2d(192, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.mp3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

    def forward(self, inputs):
        inputs = F.relu(self.bn1(self.mp1(self.conv1(inputs))))
        inputs = F.relu(self.bn2(self.mp2(self.conv2(inputs))))
        inputs = F.relu(self.conv3(inputs))
        inputs = F.relu(self.conv4(inputs))
        inputs = F.relu(self.mp3(self.conv5(inputs)))

        return inputs


class AlexNet(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N

        # Encoder
        self.encoder_1 = Encoder()
        self.encoder_2 = Encoder()

        # Decoder
        self.fc6_1 = nn.Linear(2*128*6*6, 20)
        self.fc6_2 = nn.Linear(2*128*6*6, 20)
        self.fc7_1 = nn.Linear(40, 20)
        self.fc7_2 = nn.Linear(40, 20)
        self.fc8 = nn.Linear(40, 10)

    def forward(self, inputs):
        inputs_1 = self.encoder_1(inputs)
        inputs_2 = self.encoder_2(inputs)

        inputs_1 = inputs_1.view(self.N, -1)
        inputs_2 = inputs_2.view(self.N, -1)

        temp = torch.cat((inputs_1, inputs_2), dim=1)
        inputs_1 = temp.clone()
        inputs_2 = temp.clone()

        inputs_1 = F.relu(self.fc6_1(inputs_1))
        inputs_2 = F.relu(self.fc6_2(inputs_2))

        temp = torch.cat((inputs_1, inputs_2), dim=1)
        inputs_1 = temp.clone()
        inputs_2 = temp.clone()

        inputs_1 = F.relu(self.fc7_1(inputs_1))
        inputs_2 = F.relu(self.fc7_2(inputs_2))

        inputs = torch.cat((inputs_1, inputs_2), dim=1)
        inputs = self.fc8(inputs)

        return inputs


class CaffeNet(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N

        # Encoder
        self.encoder = Encoder()

        # Decoder
        self.fc6 = nn.Linear(128*6*6, 20)
        self.fc7 = nn.Linear(20, 20)
        self.fc8 = nn.Linear(20, 10)

    def forward(self, inputs):
        inputs = self.encoder(inputs)

        inputs = inputs.view(self.N, -1)
        inputs = F.relu(self.fc6(inputs))
        inputs = F.relu(self.fc7(inputs))
        inputs = self.fc8(inputs)

        return inputs


# Training
N = 32
n_iterations = 100
inputs = torch.rand(N, 3, 224, 224)
targets = torch.randint(10, (N, 1))
targets = targets[:, 0]
targets = targets.long()

model = CaffeNet(N)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# torch.onnx.export(model, inputs, 'caffenet.onnx', verbose=True) # open in Netron

for i in range(n_iterations):
    # Forward
    optimizer.zero_grad()
    outputs = model(inputs)

    # Loss
    loss = loss_function(outputs, targets)

    # Backwards
    loss.backward()
    optimizer.step()

    print('iteration:', i, 'loss:', loss.detach().numpy())