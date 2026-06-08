"""Optuna-Tuning fuer das Training aus 02_train_model.py."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib
# 'Agg' verhindert, dass matplotlib versucht, ein GUI-Fenster zu öffnen.
# Das ist wichtig, wenn der Code auf Servern oder per SSH ohne Display läuft.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

# --- Hyperparameter & Standard-Einstellungen ---
SEED = 42
N_TRIALS = 30                # Standardanzahl an Optimierungsdurchgängen (Trials).
OPTUNA_EPOCHS = 8            # Kurzes Training pro Trial zur schnellen Trendanalyse.
OPTUNA_PATIENCE = 3          # Early Stopping innerhalb eines einzelnen Trials.
FINAL_EPOCHS = 100           # Volles Training des besten gefundenen Modells.
FINAL_PATIENCE = 6           # Geduldszeit für das finale Modelltraining.
FINAL_MIN_DELTA = 1e-4       # Minimale Verbesserung des Val-Loss im finalen Training.


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


class SimpleCNN(nn.Module):
    """Einfaches Faltungsnetzwerk (CNN) mit 3 Conv-Schichten und anpassbarem Dropout.
    Ermöglicht Optuna die Optimierung der Dropout-Wahrscheinlichkeit.
    """
    def __init__(self, num_classes: int = 2, dropout_p: float = 0.3):
        super().__init__()
        # Feature Extractor: Lernt visuelle Merkmale wie Kanten, Formen und Texturen
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
        # Klassifikationskopf: Mappt gelernte Features auf die Zielklassen (n/y)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout_p), # Durch Optuna zu optimierende Dropout-Rate
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Führt den Forward-Pass des Modells aus."""
        x = self.features(x)
        return self.classifier(x)


def build_dataloaders(
    data_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    positive_multiplier: int,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    """Lädt die Bilder, wendet Augmentierung an und balanciert die positive
    Klasse durch ein einfaches Index-Oversampling aus.
    """
    # Datenaugmentierung für das Training: Verhindert Overfitting, indem das Modell
    # rotierte und gespiegelte Varianten der Bilder sieht.
    rotate_choices = transforms.RandomChoice(
        [
            transforms.Lambda(lambda img: TF.rotate(img, 0)),
            transforms.Lambda(lambda img: TF.rotate(img, 90)),
            transforms.Lambda(lambda img: TF.rotate(img, 180)),
            transforms.Lambda(lambda img: TF.rotate(img, 270)),
        ]
    )
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            rotate_choices,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    )
    # Validierungs- und Testdaten werden nicht augmentiert, sondern nur skaliert,
    # da wir die reale Modellleistung ohne Verzerrung testen wollen.
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    train_ds = datasets.ImageFolder(data_root / 'train', transform=train_tf)
    val_ds = datasets.ImageFolder(data_root / 'val', transform=eval_tf)
    test_ds = datasets.ImageFolder(data_root / 'test', transform=eval_tf)

    class_to_idx = train_ds.class_to_idx
    if set(class_to_idx.keys()) != {'n', 'y'}:
        raise ValueError(
            f"Erwartete Klassenordner {{'n', 'y'}}, gefunden: {set(class_to_idx.keys())}"
        )

    if positive_multiplier < 1:
        raise ValueError('positive_multiplier muss >= 1 sein.')

    # Oversampling: Vervielfachen der Indizes der selteneren positiven Klasse
    positive_idx = class_to_idx['y']
    expanded_indices: list[int] = []
    for sample_idx, (_, class_idx) in enumerate(train_ds.samples):
        repeat = positive_multiplier if class_idx == positive_idx else 1
        expanded_indices.extend([sample_idx] * repeat)

    train_ds = Subset(train_ds, expanded_indices)

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
    Berechnet Verlust, Genauigkeit, Recall und F1-Score für die positive Klasse.
    """
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
    """Erstellt Plots für den Loss- und Accuracy-Verlauf über die Epochen und speichert diese als PNG."""
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
    """Generiert eine Confusion Matrix für das Testset, zeichnet sie auf der CPU
    und speichert sie als Plot-Grafik ab.
    """
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


def make_optimizer(
    optimizer_name: str,
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> optim.Optimizer:
    """Fabrik-Funktion zur Erstellung des gewünschten Optimierers anhand der Optuna-Vorgabe."""
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # SGD mit Momentum als Fallback / Alternative
    return optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=weight_decay,
    )



def parse_args():
    """Definiert die Kommandozeilenargumente für das Skript."""
    parser = argparse.ArgumentParser(description='Optuna-Tuning fuer das CNN auf dem Zuerich-Split.')
    parser.add_argument('--data-dir', type=str, default='../data/zürich/split', help='Pfad zu train/val/test.')
    parser.add_argument('--n-trials', type=int, default=N_TRIALS, help='Anzahl Optuna-Trials.')
    parser.add_argument('--optuna-epochs', type=int, default=OPTUNA_EPOCHS, help='Epochen pro Trial.')
    parser.add_argument('--optuna-patience', type=int, default=OPTUNA_PATIENCE, help='Early Stopping pro Trial.')
    parser.add_argument('--final-epochs', type=int, default=FINAL_EPOCHS, help='Finale Trainings-Epochen.')
    parser.add_argument('--final-patience', type=int, default=FINAL_PATIENCE, help='Patience fuer finales Training.')
    parser.add_argument('--final-min-delta', type=float, default=FINAL_MIN_DELTA, help='Min Delta fuer finale Val-Loss Verbesserung.')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader Worker.')
    parser.add_argument('--max-train-batches', type=int, default=0, help='Train-Batch-Limit pro Epoche (<=0 = kein Limit).')
    parser.add_argument('--study-name', type=str, default='cnn_zurich_optuna', help='Optuna Study-Name.')
    parser.add_argument('--seed', type=int, default=SEED, help='Zufalls-Seed.')
    parser.add_argument('--model-out', type=str, default='models/optuna_model_optimization.pt', help='Pfad fuer bestes Modell.')
    parser.add_argument('--plot-out', type=str, default='plot/optuna_model_training_learning_curve.png', help='Pfad fuer Lernkurven-Plot.')
    parser.add_argument('--cm-out', type=str, default='plot/optuna_test_confusion_matrix.png', help='Pfad fuer Confusion Matrix.')
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
    max_batches = None if args.max_train_batches <= 0 else args.max_train_batches
    print(f'Nutze Device: {device}')
    print('Trainiere alle Modelle von Grund auf (kein Warm-Start).')

    def objective(trial: optuna.Trial) -> float:
        """Die Zielfunktion für Optuna. Definiert den Hyperparameter-Suchraum, 
        initialisiert das Netz für jeden Trial und gibt den besten Validierungsverlust zurück.
        """
        # Seed variieren pro Trial für echte Robustheit
        seed_everything(args.seed + trial.number)

        # Definition des Hyperparameter-Suchraums
        learning_rate = trial.suggest_float('lr', 1e-5, 5e-3, log=True) # Log-skalierte Lernrate
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd'])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        dropout_p = trial.suggest_float('dropout', 0.0, 0.6, step=0.1)
        image_size = trial.suggest_categorical('image_size', [96, 128, 160])
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        positive_multiplier = trial.suggest_int('positive_multiplier', 1, 6)

        # Dataloader für die vorgeschlagene Batch- und Bildgröße erstellen
        train_loader, val_loader, _, _ = build_dataloaders(
            data_root=data_dir,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=max(0, args.num_workers),
            positive_multiplier=positive_multiplier,
        )

        # Modell laden (wird von Grund auf trainiert)
        model = SimpleCNN(num_classes=2, dropout_p=dropout_p).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = make_optimizer(optimizer_name, model, learning_rate, weight_decay)

        best_val_loss = float('inf')
        patience_ct = 0

        # Kurzes Trainings-Intervall pro Trial
        for epoch in range(args.optuna_epochs):
            run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                max_batches=max_batches,
            )
            with torch.no_grad():
                val_loss, val_acc, _, val_f1 = run_epoch(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=device,
                )

            # Optuna den Zwischenstand mitteilen
            trial.report(val_loss, epoch)
            trial.set_user_attr('last_val_acc', val_acc)
            trial.set_user_attr('last_val_f1', val_f1)
            
            # Überprüfen, ob der Trial vorzeitig abgebrochen (gepruned) werden soll
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            # Early Stopping innerhalb des Trials
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_ct = 0
            else:
                patience_ct += 1
                if patience_ct >= args.optuna_patience:
                    break

        return best_val_loss

    # TPE Sampler für intelligente probabilistische Parametersuche
    sampler = TPESampler(seed=args.seed)
    # MedianPruner bricht Trials ab, die schlechter als der Median bisheriger Läufe sind
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(
        study_name=args.study_name,
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
    )

    print(f'Starte Optuna mit {args.n_trials} Trials ...')
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Ausgabe des besten Trials
    best = study.best_trial
    print(f'Bester Trial: #{best.number} | val_loss={best.value:.4f}')
    for key in sorted(best.params.keys()):
        print(f'  {key}: {best.params[key]}')

    # --- Finales Training mit den ermittelten Best-Parametern ---
    params = best.params
    train_loader, val_loader, test_loader, class_to_idx = build_dataloaders(
        data_root=data_dir,
        image_size=params['image_size'],
        batch_size=params['batch_size'],
        num_workers=max(0, args.num_workers),
        positive_multiplier=params['positive_multiplier'],
    )

    # Modell neu initialisieren mit bestem Dropout
    model = SimpleCNN(num_classes=2, dropout_p=params['dropout']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(params['optimizer'], model, params['lr'], params['weight_decay'])

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # Trainieren über die volle Epochenanzahl
    for epoch in range(1, args.final_epochs + 1):
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
            f'Epoch {epoch:02d}/{args.final_epochs} | '
            f'train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, '
            f'train_recall={train_recall:.4f}, train_f1={train_f1:.4f} | '
            f'val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, '
            f'val_recall={val_recall:.4f}, val_f1={val_f1:.4f}'
        )

        # Early Stopping für das finale Training
        if val_loss < (best_val_loss - args.final_min_delta):
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Speichert das optimierte finale Modell mitsamt seinen Hyperparametern
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'best_trial_number': best.number,
                    'best_trial_value': best.value,
                    'best_trial_params': params,
                    'epoch': epoch,
                },
                model_out,
            )
        else:
            epochs_without_improvement += 1
            if args.final_patience > 0 and epochs_without_improvement >= args.final_patience:
                print(
                    f'Early Stopping nach Epoche {epoch}: '
                    f'best_val_loss={best_val_loss:.4f}'
                )
                break

    # Laden des besten Zustands aus dem finalen Lauf zur Evaluation auf dem Test-Set
    checkpoint = torch.load(model_out, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Finale Test-Auswertung
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

    # Lernkurven-Plot & Confusion-Matrix exportieren
    plot_learning_curves(history, plot_out)
    print(f'Lernkurven gespeichert unter: {plot_out}')

    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    save_confusion_matrix(model, test_loader, device, cm_out, class_names)
    print(f'Confusion-Matrix gespeichert unter: {cm_out}')

    print(f'Bestes Modell gespeichert unter: {model_out}')


if __name__ == '__main__':
    main()
