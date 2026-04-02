from pathlib import Path
import argparse
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Basis-Konfiguration
SEED = 42
EPOCHS = 10
LEARNING_RATE = 1e-3
MAX_TRAIN_BATCHES_PER_EPOCH = 0
PLOT_PATH = Path('plot/model_training_learning_curve.png')

def get_device() -> torch.device:
    """Wählt das beste verfügbare Rechengerät (CUDA, MPS oder CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def resolve_from_script_dir(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    script_dir = Path(__file__).resolve().parent
    return (script_dir / path).resolve()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_dataloaders(
    data_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    train_ds = datasets.ImageFolder(data_root / 'train', transform=train_tf)
    val_ds = datasets.ImageFolder(data_root / 'val', transform=eval_tf)
    test_ds = datasets.ImageFolder(data_root / 'test', transform=eval_tf)

    if set(train_ds.class_to_idx.keys()) != {'n', 'y'}:
        raise ValueError(
            f"Erwartete Klassenordner {{'n', 'y'}}, gefunden: {set(train_ds.class_to_idx.keys())}"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader, train_ds.class_to_idx


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> tuple[float, float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()

        # Binäre Metriken für positive Klasse "y" (Index 1)
        true_positives += ((preds == 1) & (labels == 1)).sum().item()
        false_positives += ((preds == 1) & (labels == 0)).sum().item()
        false_negatives += ((preds == 0) & (labels == 1)).sum().item()

        batch_size = labels.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size

    if total_samples == 0:
        return 0.0, 0.0, 0.0, 0.0

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    recall = true_positives / (true_positives + false_negatives + 1e-12)
    f1 = (2 * true_positives) / (2 * true_positives + false_positives + false_negatives + 1e-12)
    return avg_loss, accuracy, recall, f1


def plot_learning_curves(history: dict[str, list[float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use('ggplot')

    epochs = list(range(1, len(history['train_loss']) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history['train_loss'], marker='o', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], marker='o', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss-Verlauf')
    ax1.legend()

    ax2.plot(epochs, history['train_acc'], marker='o', label='Train Accuracy')
    ax2.plot(epochs, history['val_acc'], marker='o', label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy-Verlauf')
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
    class_names: list[str],
) -> None:
    model.eval()
    cm = torch.zeros((2, 2), dtype=torch.int64)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds = torch.argmax(model(images), dim=1)
            for truth, pred in zip(labels.view(-1), preds.view(-1)):
                cm[truth.long(), pred.long()] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    cm_np = cm.cpu().numpy()
    image = ax.imshow(cm_np, cmap='Blues')
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    for i in range(cm_np.shape[0]):
        for j in range(cm_np.shape[1]):
            value = cm_np[i, j]
            text_color = 'white' if value > cm_np.max() / 2 else 'black'
            ax.text(j, i, f'{value:d}', ha='center', va='center', color=text_color)
    ax.set_xlabel('Vorhersage')
    ax.set_ylabel('Wahrheit')
    ax.set_title('Confusion Matrix (Test)')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description='Trainiert ein CNN auf dem Zürich-Split-Datensatz.')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../data/zürich/split',
        help='Pfad zum Split-Ordner mit train/val/test.',
    )
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Anzahl Trainings-Epochen.')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch-Größe.')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Lernrate.')
    parser.add_argument('--image-size', type=int, default=128, help='Bildgröße (quadratisch).')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Anzahl Worker für DataLoader.',
    )
    parser.add_argument(
        '--max-train-batches',
        type=int,
        default=MAX_TRAIN_BATCHES_PER_EPOCH,
        help='Limit der Train-Batches pro Epoche (<=0 = kein Limit).',
    )
    parser.add_argument(
        '--model-out',
        type=str,
        default='models/cnn_zurich_best.pt',
        help='Output-Pfad für das beste Modell.',
    )
    parser.add_argument(
        '--plot-out',
        type=str,
        default=str(PLOT_PATH),
        help='Output-Pfad für Lernkurven-Plot.',
    )
    parser.add_argument(
        '--cm-out',
        type=str,
        default='plot/test_confusion_matrix.png',
        help='Output-Pfad für die Confusion-Matrix.',
    )
    parser.add_argument('--seed', type=int, default=SEED, help='Zufalls-Seed.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    data_dir = resolve_from_script_dir(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f'Datenordner nicht gefunden: {data_dir}')

    model_out = resolve_from_script_dir(args.model_out)
    plot_out = resolve_from_script_dir(args.plot_out)
    cm_out = resolve_from_script_dir(args.cm_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f'Nutze Device: {device}')

    train_loader, val_loader, test_loader, class_to_idx = build_dataloaders(
        data_root=data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=max(0, args.num_workers),
    )
    print(f'Klassen-Mapping: {class_to_idx}')
    print(
        f"Datensätze: train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}"
    )

    model = SimpleCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    max_batches = None if args.max_train_batches <= 0 else args.max_train_batches

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_recall, train_f1 = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            max_batches=max_batches,
        )
        with torch.no_grad():
            val_loss, val_acc, val_recall, val_f1 = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"train_recall={train_recall:.4f}, train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
            f"val_recall={val_recall:.4f}, val_f1={val_f1:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_out)

    print(f'Bestes Modell gespeichert unter: {model_out}')
    plot_learning_curves(history, plot_out)
    print(f'Lernkurven gespeichert unter: {plot_out}')

    model.load_state_dict(torch.load(model_out, map_location=device))
    model.eval()
    with torch.no_grad():
        test_loss, test_acc, test_recall, test_f1 = run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )
    print(
        f'Test: loss={test_loss:.4f}, accuracy={test_acc:.4f}, '
        f'recall={test_recall:.4f}, f1={test_f1:.4f}'
    )

    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    save_confusion_matrix(model, test_loader, device, cm_out, class_names)
    print(f'Confusion-Matrix gespeichert unter: {cm_out}')


if __name__ == '__main__':
    main()

