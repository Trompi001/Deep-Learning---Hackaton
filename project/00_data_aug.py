import argparse
from pathlib import Path

import torch
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision.utils import save_image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(input_root: Path):
    """Yield all supported image files recursively under input_root."""
    for path in sorted(input_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def build_augmentations():
    """Create 8 augmentations: 4 rotations x with/without horizontal flip."""
    base = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
    return [
        ("rot0", v2.Compose(base)),
        ("rot0_hflip", v2.Compose([v2.RandomHorizontalFlip(p=1.0), *base])),
        ("rot90", v2.Compose([v2.RandomRotation((90, 90)), *base])),
        (
            "rot90_hflip",
            v2.Compose([v2.RandomRotation((90, 90)), v2.RandomHorizontalFlip(p=1.0), *base]),
        ),
        ("rot180", v2.Compose([v2.RandomRotation((180, 180)), *base])),
        (
            "rot180_hflip",
            v2.Compose([v2.RandomRotation((180, 180)), v2.RandomHorizontalFlip(p=1.0), *base]),
        ),
        ("rot270", v2.Compose([v2.RandomRotation((270, 270)), *base])),
        (
            "rot270_hflip",
            v2.Compose([v2.RandomRotation((270, 270)), v2.RandomHorizontalFlip(p=1.0), *base]),
        ),
    ]


def augment_dataset(input_dir: Path, output_dir: Path) -> None:
    augmentations = build_augmentations()
    image_paths = list(iter_images(input_dir))

    if not image_paths:
        print(f"Keine Bilder gefunden in: {input_dir}")
        return

    total_saved = 0
    for img_path in image_paths:
        rel_parent = img_path.relative_to(input_dir).parent
        target_dir = output_dir / rel_parent
        target_dir.mkdir(parents=True, exist_ok=True)

        img = read_image(str(img_path))

        for aug_name, transform in augmentations:
            out = transform(img)
            out_name = f"{img_path.stem}_{aug_name}{img_path.suffix.lower()}"
            save_image(out, str(target_dir / out_name))
            total_saved += 1

    print(f"Fertig. Eingabebilder: {len(image_paths)}")
    print(f"Gespeicherte augmentierte Bilder: {total_saved}")
    print(f"Output-Ordner: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Augmentiert alle Bilder aus einem Ordner rekursiv und speichert sie unter output."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="../data/zürich/y/",
        help="Eingabeordner mit Bildern (rekursiv), Standard: ../data/zürich/y/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/zürich/aug_y/",
        help="Ausgabeordner fuer augmentierte Bilder, Standard: ../data/zürich/aug_y/",
    )
    return parser.parse_args()


def resolve_from_script_dir(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    script_dir = Path(__file__).resolve().parent
    return (script_dir / path).resolve()


def main():
    args = parse_args()
    input_dir = resolve_from_script_dir(args.input_dir)
    output_dir = resolve_from_script_dir(args.output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Eingabeordner existiert nicht oder ist kein Ordner: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    augment_dataset(input_dir=input_dir, output_dir=output_dir)


if __name__ == "__main__":
    main()