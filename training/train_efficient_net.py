import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm

from load_yoga_dataset import YogaPoseDataset

FOLDER_PATH = '../dataset/yoga_poses/'
num_epochs = 10
batch_size = 32
checkpoint_path = 'efficientnet_epoch_1.pt'
num_classes = 82
learning_rate = 0.001

train_file = '../dataset/Yoga-82/yoga_train.txt'
test_file = '../dataset/Yoga-82/yoga_test.txt'

transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.8, contrast=0.2, saturation=0.8, hue=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_val = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create the dataset instances
train_dataset = YogaPoseDataset(train_file, transform=transform_train)
test_dataset = YogaPoseDataset(test_file, transform=transform_val)

# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)

# Set up loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

for epoch in range(num_epochs):
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
    total_correct_train = 0
    total_samples_train = 0

    for images, label_82 in progress_bar:
        images = images.to(device)
        label_82 = label_82.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, label_82)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Training accuracy
        _, predicted = torch.max(outputs, 1)
        total_samples_train += label_82.size(0)
        total_correct_train += (predicted == label_82).sum().item()
        print(total_correct_train / total_samples_train * 100)
        progress_bar.set_postfix(
            {'Training Accuracy': '{:.2f}%'.format((total_correct_train / total_samples_train) * 100)})

torch.save(model.state_dict(), '../models/efficientnet_test2.pt')

# Evaluation on the test set
model.eval()
total_correct = 0
total_samples = 0

with torch.no_grad():
    for images, label_82 in tqdm(test_loader, desc="Testing", leave=False):
        images = images.to(device)
        label_82 = label_82.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total_samples += label_82.size(0)
        total_correct += (predicted == label_82).sum().item()

accuracy = total_correct / total_samples
print("Test Accuracy: {:.2f}%".format(accuracy * 100))
