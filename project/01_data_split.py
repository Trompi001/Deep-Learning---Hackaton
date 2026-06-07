import argparse
import random
import shutil
from pathlib import Path

# Unterstützte Bildformate für das Einlesen
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = ("train", "val", "test")


def resolve_from_script_dir(path_value: str) -> Path:
    """Konvertiert relative Pfade so, dass sie relativ zum Verzeichnis dieses
    Skripts aufgelöst werden. Verhindert Pfadfehler bei Aufrufen aus anderen Ordnern.
    """
    path = Path(path_value)
    if path.is_absolute():
        return path
    script_dir = Path(__file__).resolve().parent
    return (script_dir / path).resolve()


def iter_images(input_root: Path):
    """Durchsucht das angegebene Verzeichnis rekursiv nach Dateien
    mit den unterstützten Bildendungen und gibt diese als Generator zurück.
    """
    for path in sorted(input_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def compute_counts(total: int, splits: tuple[float, float, float]) -> tuple[int, int, int]:
    """Berechnet die exakte Anzahl der Bilder, die in Train, Val und Test landen sollen,
    basierend auf den Prozentanteilen und der Gesamtanzahl.
    """
    train_count = int(total * splits[0])
    val_count = int(total * splits[1])
    # Test-Count erhält den verbleibenden Rest, um Rundungsfehler zu vermeiden
    test_count = total - train_count - val_count
    return train_count, val_count, test_count


def split_paths(paths: list[Path], splits: tuple[float, float, float], seed: int):
    """Mischt die Pfadliste reproduzierbar anhand eines Seeds durch
    und teilt sie in Train-, Val- und Test-Listen auf.
    """
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled) # Durchmischen für zufällige Aufteilung

    # Zielanzahlen ermitteln
    train_count, val_count, _ = compute_counts(len(shuffled), splits)

    # Listen-Slicing
    train_paths = shuffled[:train_count]
    val_paths = shuffled[train_count : train_count + val_count]
    test_paths = shuffled[train_count + val_count :]

    return train_paths, val_paths, test_paths


def copy_group(files: list[Path], target_dir: Path) -> int:
    """Kopiert eine Liste von Dateien in das Zielverzeichnis.
    Verhindert Überschreibungen bei namensgleichen Dateien, indem ein Hash angehängt wird.
    Gibt die Anzahl erfolgreich kopierter Dateien zurück.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in files:
        dst = target_dir / src.name
        # Falls eine Datei mit identischem Namen bereits existiert (z. B. aus Unterordnern),
        # wird ein eindeutiger numerischer Suffix angehängt.
        if dst.exists():
            dst = target_dir / f"{src.stem}_{abs(hash(str(src))) % 1_000_000}{src.suffix.lower()}"
        shutil.copy2(src, dst) # copy2 erhält Metadaten wie Erstellungsdatum
        copied += 1
    return copied


def split_dataset(
    neg_dir: Path,
    pos_dir: Path,
    output_dir: Path,
    splits: tuple[float, float, float],
    seed: int,
) -> None:
    """Führt den geschichteten (stratified) Split für beide Klassen (n und y) durch
    und kopiert die Bilddaten in die entsprechende Ordnerstruktur (train/val/test).
    """
    # 1. Alle Bildpfade für beide Klassen sammeln
    neg_paths = list(iter_images(neg_dir))
    pos_paths = list(iter_images(pos_dir))

    # Plausibilitätsprüfung
    if not neg_paths:
        raise ValueError(f"Keine negativen Bilder gefunden in: {neg_dir}")
    if not pos_paths:
        raise ValueError(f"Keine positiven Bilder gefunden in: {pos_dir}")

    # 2. Unabhängiges Aufteilen der Klassen mit dem identischen Seed (erhält Klassenverteilung)
    neg_split = split_paths(neg_paths, splits, seed)
    pos_split = split_paths(pos_paths, splits, seed)

    split_to_idx = {"train": 0, "val": 1, "test": 2}
    class_dirs = {"n": neg_split, "y": pos_split}

    totals = {"train": 0, "val": 0, "test": 0}

    # 3. Kopieren der aufgeteilten Pfade in die Zielordner
    for split_name in SPLIT_NAMES:
        idx = split_to_idx[split_name]
        for class_name, groups in class_dirs.items():
            # Zielpfad z.B.: ../data/split/train/n/
            copied = copy_group(groups[idx], output_dir / split_name / class_name)
            totals[split_name] += copied

    # Konsolen-Zusammenfassung der Split-Ergebnisse
    print("Split abgeschlossen.")
    print(f"Negativ (n): {len(neg_paths)} Bilder")
    print(f"Positiv (y): {len(pos_paths)} Bilder")
    print(f"Train: {totals['train']} Bilder")
    print(f"Val:   {totals['val']} Bilder")
    print(f"Test:  {totals['test']} Bilder")
    print(f"Output: {output_dir}")


def parse_args():
    """Definiert die Kommandozeilenargumente für das Skript."""
    parser = argparse.ArgumentParser(
        description="Splittet Bilder aus n (negativ) und y (positiv) in train/val/test."
    )
    parser.add_argument(
        "--neg-dir",
        type=str,
        default="../data/zürich/n/",
        help="Ordner mit negativ gelabelten Bildern (Standard: ../data/zürich/n/)",
    )
    parser.add_argument(
        "--pos-dir",
        type=str,
        default="../data/zürich/y/",
        help="Ordner mit positiv gelabelten Bildern (Standard: ../data/zürich/y/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/zürich/split/",
        help="Output-Ordner fuer train/val/test (Standard: ../data/zürich/split/)",
    )
    parser.add_argument(
        "--splits",
        type=float,
        nargs=3,
        default=[0.7, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split-Ratios fuer train/val/test (Standard: 0.7 0.15 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Zufalls-Seed fuer reproduzierbaren Split (Standard: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Prüfung, ob die Summe der Split-Anteile exakt 1.0 ergibt
    splits = tuple(args.splits)
    if abs(sum(splits) - 1.0) > 1e-9:
        raise ValueError(f"Split-Ratios muessen 1.0 ergeben, aktuell: {splits}")

    # Auflösung der Pfade relativ zum Skript-Verzeichnis
    neg_dir = resolve_from_script_dir(args.neg_dir)
    pos_dir = resolve_from_script_dir(args.pos_dir)
    output_dir = resolve_from_script_dir(args.output_dir)

    # Prüfung auf Existenz der Quellordner
    if not neg_dir.exists() or not neg_dir.is_dir():
        raise FileNotFoundError(f"Negativ-Ordner existiert nicht oder ist kein Ordner: {neg_dir}")
    if not pos_dir.exists() or not pos_dir.is_dir():
        raise FileNotFoundError(f"Positiv-Ordner existiert nicht oder ist kein Ordner: {pos_dir}")

    # Verzeichnis erstellen und Split ausführen
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dataset(
        neg_dir=neg_dir,
        pos_dir=pos_dir,
        output_dir=output_dir,
        splits=splits,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()