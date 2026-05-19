import numpy as np
import torch
import torchvision
from torchvision import models
import os

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

dataset_path = "../datasets/oxford-iiit-pet-dataset/"

class OxfordIIITPetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, type="trainval", transform=None):
        if dataset_path[-1] != "/":
            dataset_path += "/"
        self.images_path = dataset_path + "images/"
        annotations_path = dataset_path + "annotations/"
        self.transform = transform

        with open(annotations_path + type + ".txt", "r") as f:
            lines = f.readlines()
            self.annotations = [line.strip().split() for line in lines]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_file = self.annotations[idx][0] + ".jpg"
        image_path = os.path.join(self.images_path, image_file)
        tensor = torchvision.io.read_image(image_path)
        label = int(image_file[0].isupper()) # 1 for cat, 0 for dog
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, label

torch.utils.data.random_split

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    # Resize so that all images are the same size. 224x224 for ResNet18.
    torchvision.transforms.Resize((224, 224)),
    # Convert RGB images to [0, 1] tensors
    torchvision.transforms.ToTensor(),
])

trainval_dataset = OxfordIIITPetDataset(dataset_path, type="trainval", transform=transform)
test_dataset = OxfordIIITPetDataset(dataset_path, type="test", transform=transform)

# Split trainval dataset into train and val
trainval_length = len(trainval_dataset)
train_length = int(0.8 * trainval_length) # 80% / 20%
val_length = trainval_length - train_length
train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_length, val_length])

print(f"Train dataset length: {len(train_dataset)}")
print(f"Validation dataset length: {len(val_dataset)}")
print(f"Test dataset length: {len(test_dataset)}")

batch_size = 64

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
)

def train_dog_vs_cat():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)

    num_features = model.fc.in_features
    num_classes = 2

    model.fc = torch.nn.Linear(num_features, num_classes).to(device)

    loss = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    smooth_loss = None
    num_epochs = 5
    for epoch in range(num_epochs):
        for batch_num, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            model.train()
            optimizer.zero_grad()
            outputs = model(images)
            batch_loss = loss(outputs, labels)
            batch_loss.backward()
            optimizer.step()

            if smooth_loss is None:
                smooth_loss = batch_loss.item()

            smooth_loss = 0.999 * smooth_loss + 0.001 * batch_loss.item()
            if batch_num % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_num}/{len(train_loader)}, Loss: {smooth_loss:.4f}")

train_dog_vs_cat()
