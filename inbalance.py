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

class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tensor, label = self.dataset[idx]
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, label

class OversamplingDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        labels = [label for _, label in dataset]
        cat_classes = set([0, 32, 33, 5, 6, 7, 9, 11, 20, 23, 26, 27])
        cat_indices = [i for i, label in enumerate(labels) if label in cat_classes]
        dog_indices = [i for i, label in enumerate(labels) if label not in cat_classes]

        # Duplicate cats five times
        self.indices = dog_indices + cat_indices * 5
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        return self.dataset[base_idx]

# These numbers are from the PyTorch guide and seem to be the values from
# ImageNet which is what ResNet18 was trained on.
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def make_inbalanced(dataset, generator):
    #cat_classes = set()
    #f = open("../datasets/oxford-iiit-pet-dataset/annotations/trainval.txt")
    #l = f.readlines()
    #for line in l:
    #    parts = line.strip().split()
    #    class_id = int(parts[1]) - 1
    #    is_cat = line[0].isupper()
    #    if is_cat:
    #        cat_classes.add(class_id)

    cat_classes = set([
        0, 32, 33, 5, 6, 7, 9, 11, 20, 23, 26, 27
    ])
    print("note: doing slow stuff")
    labels = [label for _, label in dataset]
    cat_indicies = [i for i, label in enumerate(labels) if label in cat_classes]
    dog_indicies = [i for i, label in enumerate(labels) if label not in cat_classes]
    num_cats_to_keep = int(0.2 * len(cat_indicies))
    shuffled_cat_indices = torch.randperm(len(cat_indicies), generator=generator).tolist()
    cat_indices_to_keep = [cat_indicies[i] for i in shuffled_cat_indices[:num_cats_to_keep]]
    inbalanced_indices = cat_indices_to_keep + dog_indicies
    return torch.utils.data.Subset(dataset, inbalanced_indices)

    # Only keep 20% of cats. Cats have uppercase first letter
    # base_dataset = dataset.dataset
    # base_indices = dataset.indices

    # cat_indices = [
    #     i for i, base_idx in enumerate(base_indices)
    #     if base_dataset.annotations[base_idx][0][0].isupper()
    # ]
    # dog_indices = [
    #     i for i, base_idx in enumerate(base_indices)
    #     if not base_dataset.annotations[base_idx][0][0].isupper()
    # ]
    # num_cats_to_keep = int(0.2 * len(cat_indices))
    # shuffled_cat_indices = torch.randperm(len(cat_indices), generator=generator).tolist()
    # cat_indices_to_keep = [cat_indices[i] for i in shuffled_cat_indices[:num_cats_to_keep]]
    # inbalanced_indices = cat_indices_to_keep + dog_indices
    # return torch.utils.data.Subset(dataset, inbalanced_indices)

def stratified_split(dataset, train_fraction, generator):
    labels = [dataset.annotations[i][1] for i in range(len(dataset))]
    labels = np.array(labels)

    # Get unique classes and their indices
    classes, class_indices = np.unique(labels, return_inverse=True)

    # Create stratified splits
    train_indices = []
    val_indices = []
    for class_idx in range(len(classes)):
        class_member_indices = np.where(class_indices == class_idx)[0]
        num_train_samples = int(train_fraction * len(class_member_indices))
        if num_train_samples == 0:
            #print(f"Warning: Class {classes[class_idx]} has only {len(class_member_indices)} samples, assigning 1 sample to training set.")
            num_train_samples = 1
        shuffled_indices = torch.randperm(len(class_member_indices), generator=generator).tolist()
        train_indices.extend(class_member_indices[shuffled_indices[:num_train_samples]])
        val_indices.extend(class_member_indices[shuffled_indices[num_train_samples:]])

    return torch.utils.data.Subset(dataset, train_indices), torch.utils.data.Subset(dataset, val_indices)

def load_datasets():
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        # Resize so that all images are the same size. 224x224 for ResNet18.
        torchvision.transforms.Resize(256),
        torchvision.transforms.RandomRotation(10),
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        #torchvision.transforms.Resize(256),
        #torchvision.transforms.CenterCrop(224),
        # Convert RGB images to [0, 1] tensors
        torchvision.transforms.ToTensor(),
        # Normalize.
        torchvision.transforms.Normalize(mean=mean, std=std),
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std),
    ])
    trainval_dataset = OxfordIIITPetDataset(dataset_path, type="trainval", transform=None)
    test_dataset = OxfordIIITPetDataset(dataset_path, type="test", transform=test_transform)

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = stratified_split(trainval_dataset, 0.8, generator)
    train_dataset = make_inbalanced(train_dataset, generator)
    train_dataset = OversamplingDataset(train_dataset)
    train_dataset = TransformedDataset(train_dataset, transform=train_transform)
    val_dataset = TransformedDataset(val_dataset, transform=test_transform)

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

def count_classes(dataset):
    class_counts = {}
    for _, label in dataset:
        label = label.item() if isinstance(label, torch.Tensor) else label
        class_counts[label] = class_counts.get(label, 0) + 1
    return class_counts

def evaluate_model(model, data_loader, loss_fn):
    model.eval()
    correct = 0
    total_loss = 0.0
    n = 0
    num_classes = 37
    tp_total = torch.zeros(num_classes, device=device)
    fp_total = torch.zeros(num_classes, device=device)
    fn_total = torch.zeros(num_classes, device=device)
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            labels_flat = labels.view(-1)
            predicted_flat = predicted.view(-1)
            correct_mask = predicted_flat == labels_flat

            tp_batch = torch.bincount(labels_flat[correct_mask], minlength=tp_total.numel())
            pred_counts = torch.bincount(predicted_flat, minlength=tp_total.numel())
            label_counts = torch.bincount(labels_flat, minlength=tp_total.numel())
            fp_batch = pred_counts - tp_batch
            fn_batch = label_counts - tp_batch

            tp_total += tp_batch
            fp_total += fp_batch
            fn_total += fn_batch
            n_batch = labels.size(0)
            n += n_batch
            correct += (predicted == labels).sum().item()
            batch_loss = loss_fn(outputs, labels)
            total_loss += batch_loss.item() * n_batch
    accuracy = 100 * correct / n
    loss = total_loss / n
    recall_per_class = torch.where(
        (tp_total + fn_total) > 0,
        tp_total / (tp_total + fn_total),
        torch.zeros_like(tp_total),
    )
    precision_per_class = torch.where(
        (tp_total + fp_total) > 0,
        tp_total / (tp_total + fp_total),
        torch.zeros_like(tp_total),
    )
    f1_per_class = torch.where(
        (precision_per_class + recall_per_class) > 0,
        2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class),
        torch.zeros_like(precision_per_class),
    )
    recall = recall_per_class.mean().item() * 100
    f1 = f1_per_class.mean().item() * 100
    return accuracy, loss, recall, f1, f1_per_class

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

def get_resnet_layers(model):
    layers = []
    layers.append(model.conv1)
    for blocks in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for basicblock in blocks:
            layers.append(basicblock.conv1)
            layers.append(basicblock.conv2)
    return layers

def unfreeze_last_layers(model, l):
    it = iter(reversed(get_resnet_layers(model)))
    for layer, _ in zip(it, range(l)):
        for param in layer.parameters():
            param.requires_grad = True

def freeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = False


def train_breed_classification(train_loader, val_loader, test_loader):
    if True:
        print("Training class counts:")
        for cls, count in count_classes(train_loader.dataset).items():
            print(f"  Class {cls}: {count}")
        print("Validation class counts:")
        for cls, count in count_classes(val_loader.dataset).items():
            print(f"  Class {cls}: {count}")
        print("Test class counts:")
        for cls, count in count_classes(test_loader.dataset).items():
            print(f"  Class {cls}: {count}")
    
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)

    num_features = model.fc.in_features
    num_classes = 37

    model.fc = torch.nn.Linear(num_features, num_classes).to(device)

    class_counts = count_classes(train_loader.dataset)
    num_classes = 37
    total_samples = sum(class_counts.values())
    class_weights = torch.zeros(num_classes, dtype=torch.float32)
    for cls in range(num_classes):
        count = class_counts.get(cls, 0)
        if count > 0:
            class_weights[cls] = total_samples / (num_classes * count)
    #class_weights = torch.ones(num_classes)
    #cat_classes = set([0, 32, 33, 5, 6, 7, 9, 11, 20, 23, 26, 27])
    #for cls in cat_classes:
    #    class_weights[cls] = 5.0

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Freeze all parameters except last l layers
    l = 2
    print(f"Freezing all layers except the last {l} layers and the classification layer")
    freeze_all_layers(model)
    unfreeze_last_layers(model, l+1) # +1 to also unfreeze classification layer


    history = {
        'x': [],
        'train_loss': [],
        'val_loss': [],
    }

    # Keep an iterator for the validation loader so we can evaluate only one batch
    # at the time instead of all validation data.
    val_loader_iter = iter(val_loader)

    start_time = datetime.datetime.now()

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
                factor = 0.5
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

        # Check accuracy on validation set
        #val_accuracy, val_loss = evaluate_model(model, val_loader, loss_fn)
        #print(f"Validation Accuracy: {val_accuracy:.2f}%")
    
    # Check accuracy on validation set
    val_accuracy, val_loss, val_recall, val_f1, val_f1_per_class = evaluate_model(model, val_loader, loss_fn)
    print(f"Final Validation Accuracy: {val_accuracy:.2f}%")
    print(f"Final Validation Loss: {val_loss:.4f}")
    print(f"Final Validation Recall (macro): {val_recall:.2f}%")
    print(f"Final Validation F1 (macro): {val_f1:.2f}%")
    cat_classes = set([
        0, 32, 33, 5, 6, 7, 9, 11, 20, 23, 26, 27
    ])
    cat_f1 = []
    dog_f1 = []
    for c in range(37):
        c_f1 = val_f1_per_class[c].item()
        print(f"Class {c} ({'cat' if c in cat_classes else 'dog'}) F1: {c_f1*100:.2f}%")
        if c in cat_classes:
            cat_f1.append(c_f1)
        else:
            dog_f1.append(c_f1)
    avg_cat_f1 = np.mean(np.array(cat_f1))
    avg_dog_f1 = np.mean(np.array(dog_f1))
    print(f"Average cat F1: {avg_cat_f1*100:.2f}%")
    print(f"Average dog F1: {avg_dog_f1*100:.2f}%")

    # Check accuracy on test set
    test_accuracy, test_loss, test_recall, test_f1, test_f1_per_class = evaluate_model(model, test_loader, loss_fn)
    print(f"Test Accuracy: {test_accuracy:.2f}% ONLY USE FOR INTEREST, NOT FOR MODEL SELECTION")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Recall (macro): {test_recall:.2f}%")
    print(f"Test F1 (macro): {test_f1:.2f}%")

    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print(f"Training time: {elapsed_time}")

    # Save the model
    date_identifier = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    acc_identifier = int(val_accuracy*100)
    l_identifier = f"lna"
    torch.save(model.state_dict(), f"models/resnet18_breed_inbalanced_{date_identifier}_valacc{acc_identifier}_{l_identifier}_time{elapsed_time.total_seconds():.0f}.pth")

    # send ntfy notification request
    if elapsed_time.total_seconds() > 2*60:
        os.system(f"curl -d 'Training complete! Validation accuracy: {val_accuracy:.2f}%' -H 'Title: Training Done' https://ntfy.sh/alvin_4132_dd2424_training_done")

    # Plot training and validation loss
    #plt.figure()
    #plt.plot(history['x'], history['train_loss'], label='Train Loss')
    #plt.plot(history['x'], history['val_loss'], label='Validation Loss')
    #plt.xlabel('Update Step')
    #plt.ylabel('Loss')
    #plt.title('Training and Validation Loss')
    #plt.legend()
    #plt.show()

if __name__ == "__main__":
    print(f"Using device: {device}")
    train_loader, val_loader, test_loader = load_datasets()
    train_breed_classification(train_loader, val_loader, test_loader)
    


False