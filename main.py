import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from lenet import LeNet

import matplotlib.pyplot as plt

# Fixing parameters
lr = .001
batch_size = 100

torch.manual_seed(2021)

# Creating dataset and pre-processing
transforms = transforms.Compose([transforms.Resize((32, 32)),
                                 transforms.ToTensor()])
train_dataset = datasets.MNIST(root='data/mnist',
                               train=True,
                               transform=transforms,
                               download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)


# Instantiating model, optimizer and loss function
model = LeNet(10)
optimizer = Adam(model.parameters(), lr=lr)
criterion = CrossEntropyLoss()

losses = []

print('training in progress')
for x, y in train_loader:
    optimizer.zero_grad()
    y_hat = model(x)
    loss = criterion(y_hat[0], y)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()

print('training done')

plt.plot(losses)
plt.xlabel("Number of iterations")
plt.ylabel("Loss")
plt.show()
