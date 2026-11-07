#     NAME   :   ID    
#Rakib Khalil : 1979020
#Zaid Azkoul : 2261703
#Ali Shaikh : 2334342
#Yousif Husni: 2169551


import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms

TRAIN_PATH = "imagewoof_train/train"
VAL_PATH = "imagewoof_val/val"

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_class_names(path="classes.txt"):
    names = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            names.append(parts[1] if len(parts) == 2 else parts[0])
    return names


def apply_class_order(dataset, class_names):
    mapping = {
        dataset.class_to_idx[name]: idx
        for idx, name in enumerate(class_names)
    }
    dataset.samples = [(path, mapping[label]) for path, label in dataset.samples]
    dataset.targets = [mapping[label] for label in dataset.targets]
    dataset.imgs = dataset.samples
    dataset.classes = list(class_names)
    dataset.class_to_idx = {name: idx for idx, name in enumerate(class_names)}


class NeuralNet(nn.Module):
    """
    *block(...) unpacks the three layers (Conv → BN → ReLU) into the list, so the whole thing is still one plain nn.Sequential.
    There are 13 conv layers, about 9M parameters, and no skip connections, so it’s within the rules.
    """

    def __init__(self, num_classes=10):
        super().__init__()

        def block(cin, cout):
            return [
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            ]

        self.layers = nn.Sequential(
            *block(3, 64), *block(64, 64), nn.MaxPool2d(2),  # 128 -> 64
            *block(64, 128), *block(128, 128), nn.MaxPool2d(2),  # 64 -> 32
            *block(128, 256), *block(256, 256), *block(256, 256), nn.MaxPool2d(2),  # 32 -> 16
            *block(256, 384), *block(384, 384), *block(384, 384), nn.MaxPool2d(2),  # 16 -> 8
            *block(384, 512), *block(512, 512), *block(512, 512), nn.MaxPool2d(2),  # 8 -> 4
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


def load_model(path="model.pth"):
    """Build the model, load weights from path, return it in eval mode."""
    device = get_device()
    model = NeuralNet()
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict(model, images):
    """images: uint8 numpy array, shape (N, 128, 128, 3), RGB, values 0-255.
    Return: integer numpy array, shape (N,), class ids 0-9 in classes.txt order.
    """
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[1:] != (128, 128, 3):
        raise ValueError(
            f"Expected images shape (N, 128, 128, 3), got {images.shape}"
        )

    device = next(model.parameters()).device
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    preds = []
    batch_size = 32
    model.eval()
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        x = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
        x = x.permute(0, 3, 1, 2) / 255.0
        x = (x - mean) / std
        out = model(x).argmax(1)
        preds.append(out.cpu().numpy())

    return np.concatenate(preds).astype(np.int64)


def build_loaders(batch_size=32, num_workers=2):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.35, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_dataset = datasets.ImageFolder(root=TRAIN_PATH, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=VAL_PATH, transform=val_transform)

    class_names = load_class_names()
    apply_class_order(train_dataset, class_names)
    apply_class_order(val_dataset, class_names)
    assert train_dataset.class_to_idx["shih_tzu"] == 0
    assert train_dataset.class_to_idx["dingo"] == 9
    print("Label order (classes.txt):")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_dataset, val_dataset, train_loader, val_loader, class_names


@torch.no_grad()
def evaluate(weights_path="model.pth"):
    device = get_device()
    print("Evaluating on:", device)

    _, val_dataset, _, _, class_names = build_loaders(num_workers=0)
    model = load_model(weights_path)
    print(f"Loaded {weights_path} ({sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params)")

    # Harness-style check: raw uint8 images -> predict()
    images = []
    labels = []
    for path, label in val_dataset.samples:
        img = np.asarray(Image.open(path).convert("RGB").resize((128, 128)), dtype=np.uint8)
        images.append(img)
        labels.append(label)
    images = np.stack(images, axis=0)
    labels = np.asarray(labels, dtype=np.int64)

    preds = predict(model, images)
    val_acc = (preds == labels).mean()
    print(f"Validation accuracy: {val_acc:.3f} ({int((preds == labels).sum())}/{len(labels)})")
    return float(val_acc)


def train(num_epochs=150, save_path="model.pth", seed=SEED):
    set_seed(seed)

    train_dataset, val_dataset, train_loader, val_loader, class_names = build_loaders()

    device = get_device()
    print("Training on:", device)

    torch.manual_seed(seed)
    net = NeuralNet().to(device)
    print(f"Parameters: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")

    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(
        net.parameters(),
        lr=0.1,
        momentum=0.9,
        nesterov=True,
        weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )

    use_cuda_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_cuda_amp else None

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # ---- training ----
        net.train()
        running_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_cuda_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = net(inputs)
                    loss = loss_function(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = net(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # ---- validation ----
        net.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total

        # ---- learning-rate schedule ----
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # ---- timer ----
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        remaining = elapsed / (epoch + 1) * (num_epochs - epoch - 1)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | loss {train_loss:.3f} | "
            f"train acc {train_acc:.3f} | val acc {val_acc:.3f} | "
            f"lr {current_lr:.4f} | {epoch_time:.0f}s | "
            f"elapsed {elapsed / 60:.1f} min | ~{remaining / 60:.0f} min left"
        )

        # ---- save the best model ----
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)
            print(f"  saved new best model to {save_path}")

    total_time = time.time() - start_time
    print(f"Done in {total_time / 60:.1f} min. Best val accuracy: {best_acc:.3f}")
    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", default="eval", choices=["eval", "train"])
    parser.add_argument("--weights", default="model.pth")
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()

    if args.mode == "train":
        train(num_epochs=args.epochs)
    else:
        evaluate(weights_path=args.weights)


if __name__ == "__main__":
    main()
