import numpy as np
import torch
import torchvision
from torchvision import models
import os
from matplotlib import pyplot as plt
import datetime

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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
            """
            Annotations are:
            [file name, class id, species id, breed id]
            """
            self.annotations = [line.strip().split() for line in lines]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_file = self.annotations[idx][0] + ".jpg"
        image_path = os.path.join(self.images_path, image_file)
        tensor = torchvision.io.read_image(image_path, mode=torchvision.io.ImageReadMode.RGB)
        # The dataset has 37 classes, one-indexed. Subtract 1 to get zero-indexed.
        label = int(self.annotations[idx][1]) - 1 # class id [0-36] inclusive.
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, label

# These numbers are from the PyTorch guide and seem to be the values from
# ImageNet which is what ResNet18 was trained on.
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def load_datasets():
    trainval_dataset = OxfordIIITPetDataset(dataset_path, type="trainval", transform=torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        # Resize so that all images are the same size. 224x224 for ResNet18.
        #torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.RandomRotation(10),
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        # Convert RGB images to [0, 1] tensors
        torchvision.transforms.ToTensor(),
        # Normalize.
        torchvision.transforms.Normalize(mean=mean, std=std),
    ]))
    test_dataset = OxfordIIITPetDataset(dataset_path, type="test", transform=torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std),
    ]))

    # Split trainval dataset into train and val
    trainval_length = len(trainval_dataset)
    train_length = int(0.8 * trainval_length)
    val_length = trainval_length - train_length

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_length, val_length], generator=generator)

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")

    batch_size = 32

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # Ensure GPU is kept busy
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader

class_id_to_breed_name = {}

def populate_class_id_to_breed():
    annotations_path = dataset_path + "annotations/"
    with open(annotations_path + "trainval.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            filename = parts[0]
            class_id = int(parts[1]) - 1
            breed_name = " ".join(filename.split("_")[:-1])
            is_cat = filename[0].isupper()
            text = f"{breed_name} ({'cat' if is_cat else 'dog'})"
            class_id_to_breed_name[class_id] = text

def show_batch(data_loader):
    images, labels = next(iter(data_loader))
    # Undo normalization
    meant = torch.tensor(mean)
    stdt = torch.tensor(std)
    images = images * stdt[:, None, None] + meant[:, None, None]
    
    grid = torchvision.utils.make_grid(images, nrow=8)
    plt.figure(figsize=(12, 6))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(f"{labels}")
    plt.axis("off")
    plt.show()

def show_image(image_tensor, title):
    # Undo normalization
    meant = torch.tensor(mean)
    stdt = torch.tensor(std)
    image_tensor = image_tensor * stdt[:, None, None] + meant[:, None, None]
    
    plt.imshow(image_tensor.permute(1, 2, 0))
    plt.title(title)
    plt.axis("off")
    plt.show()

def evaluate_model(model, data_loader, loss_fn):
    model.eval()
    correct = 0
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            n_batch = labels.size(0)
            n += n_batch
            correct += (predicted == labels).sum().item()
            batch_loss = loss_fn(outputs, labels)
            total_loss += batch_loss.item() * n_batch
    accuracy = 100 * correct / n
    loss = total_loss / n
    return accuracy, loss

def evaluate_model_once(model, iterator, data_loader, loss_fn):
    model.eval()
    try:
        data = next(iterator)
    except StopIteration:
        # End of epoch, restart iterator
        iterator = iter(data_loader)
        data = next(iterator)
    images, labels = data
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    n = labels.size(0)
    accuracy = 100 * (predicted == labels).sum().item() / n
    loss = loss_fn(outputs, labels).item()
    return accuracy, loss, iterator

def train_breed_classification():
    train_loader, val_loader, test_loader = load_datasets()

    if False:
        show_batch(train_loader)
        show_batch(val_loader)
        show_batch(test_loader)
    
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)

    num_features = model.fc.in_features
    num_classes = 37

    model.fc = torch.nn.Linear(num_features, num_classes).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Freeze all parameters except last fully connected layer
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    history = {
        'x': [],
        'train_loss': [],
        'val_loss': [],
    }

    # Keep an iterator for the validation loader so we can evaluate only one batch
    # at the time instead of all validation data.
    val_loader_iter = iter(val_loader)

    smooth_loss = None
    smooth_val_loss = None
    num_epochs = 10
    update_step = 0
    for epoch in range(num_epochs):
        for batch_num, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            model.train()
            optimizer.zero_grad()
            outputs = model(images)
            batch_loss = loss_fn(outputs, labels)
            batch_loss.backward()
            optimizer.step()

            
            update_step += 1
            if batch_num % 5 == 0:
                factor = 0.95
                val_accuracy_batch, val_loss_batch, val_loader_iter = evaluate_model_once(model, val_loader_iter, val_loader, loss_fn)
                if smooth_val_loss is None or val_loss_batch is None:
                    smooth_loss = batch_loss.item()
                    smooth_val_loss = val_loss_batch
                smooth_loss = factor * smooth_loss + (1-factor) * batch_loss.item()
                smooth_val_loss = factor * smooth_val_loss + (1-factor) * val_loss_batch
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_num}/{len(train_loader)}, Loss: {smooth_loss:.4f} ({smooth_val_loss:.4f} val)")
                history['x'].append(update_step)
                history['train_loss'].append(smooth_loss)
                history['val_loss'].append(smooth_val_loss)

        # Step scheduler to reduce learning rate if needed
        scheduler.step()
        # if epoch == 2:
        #     # Unfreeze all layers after 3 epochs
        #     print("Unfreezing all layers for fine-tuning")
        #     for param in model.parameters():
        #         param.requires_grad = True

        # Check accuracy on validation set
        val_accuracy, val_loss = evaluate_model(model, val_loader, loss_fn)
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
    
    # Check accuracy on test set
    test_accuracy, test_loss = evaluate_model(model, test_loader, loss_fn)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    # Save the model
    date_identifier = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), f"models/resnet18_breed_{date_identifier}_acc{int(test_accuracy*100)}.pth")

    # Plot training and validation loss
    plt.figure()
    plt.plot(history['x'], history['train_loss'], label='Train Loss')
    plt.plot(history['x'], history['val_loss'], label='Validation Loss')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def apply_model():
    model_names = os.listdir("models")
    print("\nAvailable models:")
    for i, model_file in enumerate(model_names):
        print(f"{i}: {model_file}")
    choice = int(input("Enter the number of the model to load: "))
    model_path = os.path.join("models", model_names[choice])
    model = models.resnet18(weights=None).to(device)
    model.fc = torch.nn.Linear(model.fc.in_features, 37).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    images = os.listdir("test_images")
    print("\nAvailable test images:")
    for i, image_file in enumerate(images):
        print(f"{i}: {image_file}")
    choice = int(input("Enter the number of the image to classify: "))
    image_path = os.path.join("test_images", images[choice])
    
    image = torchvision.io.read_image(image_path, mode=torchvision.io.ImageReadMode.RGB)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(224),
        #torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std),
    ])
    image = transform(image)

    output = model(image.unsqueeze(0).to(device))
    _, predicted = torch.max(output.data, 1)
    predicted = predicted.item()
    confidence = torch.softmax(output, dim=1)[0][predicted].item()
    breed_name = class_id_to_breed_name.get(predicted, "Unknown")

    print(f"Predicted breed: {breed_name} (class {predicted}) with confidence {confidence*100:.2f}%")
    show_image(image, f"{breed_name} ({confidence*100:.2f}% confidence)")

if __name__ == "__main__":
    print(f"Using device: {device}")
    populate_class_id_to_breed()
    print("Do you want to train or apply?")
    choice = input("Enter 'train' or 'apply': ")
    if choice == "train":
        train_breed_classification()
    elif choice == "apply":
        apply_model()



