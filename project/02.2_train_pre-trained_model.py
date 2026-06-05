from pathlib import Path
import argparse
import random

import matplotlib
# 'Agg' verhindert, dass matplotlib versucht, ein GUI-Fenster zu öffnen.
# Das ist wichtig, wenn der Code auf Servern oder per SSH ohne Display läuft.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torchvision.transforms import functional as TF

# --- Hyperparameter & Einstellungen ---
SEED = 42
EPOCHS = 30             # Da wir ein vortrainiertes Modell nutzen, reichen 30 Epochen völlig aus.
BATCH_SIZE = 128        # 128 Bilder pro Batch lasten die GPU gut aus und halten das Training stabil.
LEARNING_RATE = 5e-5    # Sehr kleine Lernrate, um das vor-trainierte Wissen des ResNets nicht zu zerstören.
WEIGHT_DECAY = 1e-3     # L2-Regularisierung, um extrem große Gewichte zu verhindern (beugt Overfitting vor).
MAX_TRAIN_BATCHES_PER_EPOCH = 0
EARLY_STOPPING_PATIENCE = 5  # Wenn sich die Validation Loss 5 Epochen lang nicht verbessert, brechen wir ab.
EARLY_STOPPING_MIN_DELTA = 1e-6
PLOT_PATH = 'plot/pre-trained_model_training_'

def get_device() -> torch.device:
    """Wählt das beste verfügbare Rechengerät.
    Gibt CUDA für Nvidia-GPUs, MPS für Apple Silicon oder CPU zurück.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def resolve_from_script_dir(path_value: str) -> Path:
    """Konvertiert relative Pfade so, dass sie relativ zum Verzeichnis dieses
    Skripts aufgelöst werden. Verhindert Pfadfehler bei Aufrufen aus anderen Ordnern.
    """
    path = Path(path_value)
    if path.is_absolute():
        return path
    script_dir = Path(__file__).resolve().parent
    return (script_dir / path).resolve()


def seed_everything(seed: int) -> None:
    """Fixiert alle Zufallsgeneratoren (Python, PyTorch, CUDA), damit die
    Ergebnisse bei jedem Durchlauf exakt identisch und reproduzierbar sind.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Erzwingt deterministische Algorithmen auf der GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Random90Rotation:
    """Führt eine zufällige Rotation um 0, 90, 180 oder 270 Grad durch.
    Als Klasse statt Lambda implementiert, damit PyTorchs Multiprozessor-Dataloader
    unter Windows nicht abstürzt (Python-Lambdas können nicht 'gepickelt'/serialisiert werden).
    """
    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        return TF.rotate(img, angle)


def get_model(num_classes: int = 2) -> nn.Module:
    """Lädt ein vortrainiertes ResNet18-Modell (Transfer Learning).
    Das Modell hat bereits auf Millionen Alltagsbildern gelernt, Kanten, Linien
    und Muster zu erkennen. Wir tauschen nur den letzten Klassifikationslayer (fc) aus.
    """
    # DEFAULT lädt die aktuell besten vortrainierten Gewichte
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Der ursprüngliche Klassifikations-Layer (fc) wird durch einen neuen linearen Layer
    # mit der Anzahl unserer Zielklassen (y/n -> 2) ersetzt. Dieser neue Layer startet mit
    # zufälligen Gewichten und wird im Training gelernt.
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def build_dataloaders(
    data_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    """Lädt die Bilder, wendet Augmentierung an und balanciert die Klassen
    im Trainingsset dynamisch über einen WeightedRandomSampler aus.
    """
    # Daten-Augmentierung für das Training: Verhindert Overfitting, indem das Modell
    # in jeder Epoche leicht veränderte (rotierte/gespiegelte) Bilder sieht.
    #transforms.Normalize verschiebt den Pixelbereich auf Mittelwert 0 und Varianz 1.
    # Die hier genutzten Werte sind der Standard für ImageNet-Modelle.
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            Random90Rotation(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # Validierungs- und Testdaten werden NICHT augmentiert (nicht rotiert oder gespiegelt),
    # da wir die echte Leistung messen wollen. Sie werden nur skaliert und normalisiert.
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # ImageFolder ordnet die Bilder automatisch anhand der Ordnerstruktur (train/val/test -> y/n) zu
    train_ds = datasets.ImageFolder(data_root / 'train', transform=train_tf)
    val_ds = datasets.ImageFolder(data_root / 'val', transform=eval_tf)
    test_ds = datasets.ImageFolder(data_root / 'test', transform=eval_tf)

    class_to_idx = train_ds.class_to_idx

    if set(class_to_idx.keys()) != {'n', 'y'}:
        raise ValueError(
            f"Erwartete Klassenordner {{'n', 'y'}}, gefunden: {set(class_to_idx.keys())}"
        )

    # --- Start der Klassenbalancierung (WeightedRandomSampler) ---
    # Ziel: Da wir 20x mehr negative als positive Bilder haben, soll der Dataloader
    # positive Bilder 20x häufiger ziehen, damit jeder Batch ca. 50/50 aufgeteilt ist.
    targets = train_ds.targets
    neg_idx = class_to_idx['n']
    pos_idx = class_to_idx['y']
    
    neg_count = targets.count(neg_idx)
    pos_count = targets.count(pos_idx)

    class_counts = [neg_count, pos_count]
    # Gewicht ist der Kehrwert der Häufigkeit. Seltene Klassen erhalten ein hohes Gewicht.
    class_weights = [1.0 / max(1, count) for count in class_counts]

    # Jedes einzelne Bild in der Liste bekommt das Gewicht seiner Klasse zugewiesen
    sample_weights = [class_weights[label] for label in targets]

    # Der Sampler zieht Bilder basierend auf ihren Gewichten.
    # Da positive Bilder ein 20x höheres Gewicht haben, werden sie im Schnitt 20x öfter gezogen.
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # Erlaubt das mehrfache Ziehen desselben (seltenen) Bildes im selben Epoch.
    )

    # DataLoader Erstellung. WICHTIG: Wenn ein Sampler aktiv ist, darf shuffle=True nicht gesetzt sein.
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,  
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), # Beschleunigt den Datentransfer zur GPU
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
    return train_loader, val_loader, test_loader, class_to_idx


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> tuple[float, float, float, float]:
    """Führt eine einzelne Epoche (Training oder Validierung) durch.
    Gibt den durchschnittlichen Loss, Accuracy, Recall und F1-Score zurück.
    """
    is_train = optimizer is not None
    model.train(is_train) # Aktiviert/Deaktiviert Dropout und BatchNorm entsprechend

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        # Daten auf GPU/MPS verschieben
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad() # Verhindert, dass sich Gradienten aus vorherigen Batches addieren

        # Forward Pass: Vorhersage generieren
        logits = model(images)
        loss = criterion(logits, labels)

        # Backward Pass & Optimierung (nur im Training)
        if is_train:
            loss.backward()
            optimizer.step()

        # Metriken berechnen
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()

        # Binäre Metriken für positive Klasse "y" (Index 1) zur F1-Berechnung
        true_positives += ((preds == 1) & (labels == 1)).sum().item()
        false_positives += ((preds == 1) & (labels == 0)).sum().item()
        false_negatives += ((preds == 0) & (labels == 1)).sum().item()

        batch_size = labels.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size

    if total_samples == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Durchschnittswerte für die gesamte Epoche berechnen
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    # 1e-12 verhindert eine Division durch Null, falls TP, FP oder FN Null sind
    recall = true_positives / (true_positives + false_negatives + 1e-12)
    f1 = (2 * true_positives) / (2 * true_positives + false_positives + false_negatives + 1e-12)
    return avg_loss, accuracy, recall, f1


def plot_learning_curves(history: dict[str, list[float]], output_path: Path) -> None:
    """Erstellt Plots für den Loss- und Accuracy-Verlauf über die Epochen
    und speichert diese als PNG.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use('ggplot')

    epochs = list(range(1, len(history['train_loss']) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss Plot (Links)
    ax1.plot(epochs, history['train_loss'], marker='o', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], marker='o', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss-Verlauf')
    ax1.legend()

    # Accuracy Plot (Rechts)
    ax2.plot(epochs, history['train_acc'], marker='o', label='Train Accuracy')
    ax2.plot(epochs, history['val_acc'], marker='o', label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy-Verlauf')
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig) # Schließt die Figure, um Arbeitsspeicher freizugeben


def save_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
    class_names: list[str],
) -> None:
    """Generiert eine Confusion Matrix für das Testset und speichert sie als Plot.
    Zeigt genau, wie viele 'y' als 'n' vorhergesagt wurden und umgekehrt.
    """
    model.eval()
    cm = torch.zeros((2, 2), dtype=torch.int64)

    # Schnelle und device-sichere Batch-Berechnung
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            # Vorhersagen berechnen und direkt auf CPU verschieben
            preds = torch.argmax(model(images), dim=1).cpu()
            labels = labels.cpu()
            # Vektorisierte Addition auf der CPU verhindert Device-Mismatch und ist schnell
            for t in range(2):
                for p in range(2):
                    cm[t, p] += ((labels == t) & (preds == p)).sum().item()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    cm_np = cm.numpy()
    image = ax.imshow(cm_np, cmap='Blues')
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # Text-Beschriftungen in die Kästchen schreiben
    for i in range(cm_np.shape[0]):
        for j in range(cm_np.shape[1]):
            value = cm_np[i, j]
            # Heller Text auf dunklem Grund, dunkler Text auf hellem Grund
            text_color = 'white' if value > cm_np.max() / 2 else 'black'
            ax.text(j, i, f'{value:d}', ha='center', va='center', color=text_color)
            
    ax.set_xlabel('Vorhersage')
    ax.set_ylabel('Wahrheit')
    ax.set_title('Confusion Matrix (Test)')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args():
    """Definiert die Kommandozeilenargumente für das Skript."""
    parser = argparse.ArgumentParser(description='Trainiert ein ResNet18 auf dem Zürich-Split-Datensatz.')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../data/zürich/split',
        help='Pfad zum Split-Ordner mit train/val/test.',
    )
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Anzahl Trainings-Epochen.')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch-Größe.')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Lernrate.')
    parser.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY, help='Weight Decay.')
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
        '--early-stopping-patience',
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help='Anzahl Epochen ohne ausreichende Val-Loss-Verbesserung bis Stopp (<=0 = deaktiviert).',
    )
    parser.add_argument(
        '--early-stopping-min-delta',
        type=float,
        default=EARLY_STOPPING_MIN_DELTA,
        help='Minimale Val-Loss-Verbesserung, die als Fortschritt gilt.',
    )
    parser.add_argument(
        '--model-out',
        type=str,
        default='models/ResNet18_zurich.pt',
        help='Output-Pfad für das beste Modell.',
    )
    parser.add_argument(
        '--plot-out',
        type=str,
        default=PLOT_PATH + 'learning_curve.png',
        help='Output-Pfad für Lernkurven-Plot.',
    )
    parser.add_argument(
        '--cm-out',
        type=str,
        default=PLOT_PATH + 'confusion_matrix.png',
        help='Output-Pfad für Confusion Matrix.',
    )
    parser.add_argument('--seed', type=int, default=SEED, help='Zufalls-Seed.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    # Pfade auflösen
    data_dir = resolve_from_script_dir(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f'Datenordner nicht gefunden: {data_dir}')

    model_out = resolve_from_script_dir(args.model_out)
    plot_out = resolve_from_script_dir(args.plot_out)
    cm_out = resolve_from_script_dir(args.cm_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f'Nutze Device: {device}')

    # Dataloader bauen
    train_loader, val_loader, test_loader, class_to_idx = build_dataloaders(
        data_root=data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=max(0, args.num_workers),
    )
    print(f'Klassen-Mapping: {class_to_idx}')
    print(
        f"Datensätze (mit 50/50 Sampler): train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}"
    )

    # Modell laden
    model = get_model(num_classes=2).to(device)
    
    # Da der WeightedRandomSampler bereits für perfekte Balance in jedem Batch sorgt,
    # benötigt die Loss-Funktion KEINE Gewichte mehr!
    criterion = nn.CrossEntropyLoss()
    
    # AdamW (entkoppeltes Weight Decay) zur besseren Regularisierung
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # CosineAnnealingLR reduziert die Lernrate über die Epochen hinweg sanft in Form einer Cosinus-Kurve.
    # Das hilft dem Modell, sich am Ende des Trainings präzise einzupendeln.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    max_batches = None if args.max_train_batches <= 0 else args.max_train_batches

    # --- Trainings-Schleife ---
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

        # Lernrate für die nächste Epoche anpassen
        scheduler.step()

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

        # Early Stopping Überprüfung
        if val_loss < (best_val_loss - args.early_stopping_min_delta):
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_out) # Speichert den besten Zustand
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0:
                print(
                    f'Keine ausreichende Verbesserung der Val-Loss seit '
                    f'{epochs_without_improvement} Epoche(n) '
                    f'(patience={args.early_stopping_patience}).'
                )
                if epochs_without_improvement >= args.early_stopping_patience:
                    print(
                        f'Early Stopping nach Epoche {epoch}: '
                        f'best_val_loss={best_val_loss:.4f}'
                    )
                    break

    print(f'Bestes Modell gespeichert unter: {model_out}')
    plot_learning_curves(history, plot_out)
    print(f'Lernkurven gespeichert unter: {plot_out}')

    # --- Test-Evaluation am Ende des Trainings ---
    # Wir laden den Zustand, der die beste Validation-Loss erzielt hat
    model.load_state_dict(torch.load(model_out, map_location=device))
    model.eval()
    with torch.no_grad():
        test_loss, test_acc, test_recall, test_f1 = run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    save_confusion_matrix(model, test_loader, device, cm_out, class_names)
    print(
        f'Test: loss={test_loss:.4f}, accuracy={test_acc:.4f}, '
        f'recall={test_recall:.4f}, f1={test_f1:.4f}'
    )
    print(f'Confusion Matrix gespeichert unter: {cm_out}')


if __name__ == '__main__':
    main()