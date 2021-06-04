import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from lenet import LeNet5

import matplotlib.pyplot as plt

# Fixing training parameters
lr = .001
batch_size = 100
num_workers = 0
num_epochs = 20
torch.manual_seed(2021)

# Creating dataset and pre-processing
transforms = transforms.Compose([transforms.Resize((32, 32)),
                                 transforms.ToTensor()])
train_dataset = datasets.MNIST(root='data/mnist', train=True, transform=transforms, download=True)
test_dataset = datasets.MNIST(root='data/mnist', train=False, transform=transforms, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

# Instantiating model, optimizer and loss function
model = LeNet5(10)
optimizer = Adam(model.parameters(), lr=lr)
criterion = CrossEntropyLoss()  # Original network was trained using a different loss function


# Training loop
train_losses = []
val_losses = []

print('Training in progress...')
for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        y_hat, _ = model(x)
        train_loss = criterion(y_hat, y)
        running_loss += train_loss.item() * x.size(0)
        train_loss.backward()
        optimizer.step()
    running_loss = running_loss / len(train_loader.dataset)
    train_losses.append(running_loss)

    # Validation
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            y_hat, _ = model(x)
            val_loss = criterion(y_hat, y)
            running_loss += val_loss.item() * x.size(0)
    running_loss = running_loss / len(test_loader.dataset)
    val_losses.append(val_loss.item())
    # Print out progress message
    if epoch % 5 == 0:
        print(f'Epoch: {epoch}| Training loss: {train_loss}| Validation loss: {val_loss}')


print('Training done!')

# Displaying training curves
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
