# load mnist data and test the model from the saved checkpoint

from operator import le
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist_train import SimpleNN

# Define a simple neural network
model = SimpleNN()

# Load the saved model checkpoint
model.load_state_dict(torch.load('model_5.pth'))

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Test the model
model.eval()

correct = 0
total = 0

# visualize the test images
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataiter = iter(test_loader)
if len(test_loader) > 0:
    images, labels = next(dataiter)
else:
    print("Test loader is empty")

# print images
imshow(torchvision.utils.make_grid(images))
plt.show()
# output the prediction
outputs = model(images)
_, predicted = torch.max(outputs, 1)
# print number of good predictions and total number of predictions
good_predictions = (predicted == labels).sum().item()
total_predictions = labels.size(0)
print(f'Number of good predictions: {good_predictions}')
print(f'Total number of predictions: {total_predictions}')
print(f'Accuracy: {100 * good_predictions / total_predictions}%')

with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
