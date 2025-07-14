import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))

class MNISTReader(datasets.VisionDataset):
    def __init__(self, root: str) -> None:
        super().__init__(root)
        self.data_label = torch.load(root)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self) -> int:
        return len(self.data_label)

    def __getitem__(self, index):
        image, target = self.data_label[index]
        return self.transform(image), target

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 6, 5, 1, 2),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, 1, 0),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),  
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, 2),
        )

    def forward(self, x):
        return self.model(x)

def train_one_model(train_path, epochs=10, batch_size=32, learning_rate=0.01):
    print(f"Loading training data from: {train_path}")
    train_dataset = MNISTReader(train_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = LeNet().to(device)
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_correct = 0
        total_samples = 0
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total_correct += (outputs.argmax(1) == targets).sum().item()
            total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = total_correct / total_samples
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    return model

def test_model(model, test_path, batch_size=1000):
    print(f"Testing on dataset: {test_path}")
    test_dataset = MNISTReader(test_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_correct += (outputs.argmax(1) == targets).sum().item()
            total_samples += inputs.size(0)

    accuracy = total_correct / total_samples
    print(f"Test accuracy: {accuracy:.4f}")
    return accuracy

def main():
    # 训练集文件列表
    train_files = [
        './data/ColoredMNIST/train1.pt',
        './data/ColoredMNIST/train2.pt',
        './data/ColoredMNIST/train3.pt',
    ]
    # 测试集文件列表
    test_files = [
        './data/ColoredMNIST/test1.pt',
        './data/ColoredMNIST/test2.pt',
    ]

    results = {}

    for train_path in train_files:
        print("="*50)
        print(f"Training model on {os.path.basename(train_path)}")
        model = train_one_model(train_path, epochs=10)

        # 测试模型在两个测试集上的表现
        accuracies = {}
        for test_path in test_files:
            acc = test_model(model, test_path)
            accuracies[os.path.basename(test_path)] = acc
        results[os.path.basename(train_path)] = accuracies

    print("\nAll results:")
    for train_set, test_accs in results.items():
        print(f"Model trained on {train_set}:")
        for test_set, acc in test_accs.items():
            print(f"  Test on {test_set}: Accuracy = {acc:.4f}")

if __name__ == '__main__':
    main()
