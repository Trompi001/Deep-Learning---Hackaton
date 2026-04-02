import argparse
import random
import shutil
from pathlib import Path


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = ("train", "val", "test")


def resolve_from_script_dir(path_value: str) -> Path:
	path = Path(path_value)
	if path.is_absolute():
		return path
	script_dir = Path(__file__).resolve().parent
	return (script_dir / path).resolve()


def iter_images(input_root: Path):
	for path in sorted(input_root.rglob("*")):
		if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
			yield path


def compute_counts(total: int, splits: tuple[float, float, float]) -> tuple[int, int, int]:
	train_count = int(total * splits[0])
	val_count = int(total * splits[1])
	test_count = total - train_count - val_count
	return train_count, val_count, test_count


def split_paths(paths: list[Path], splits: tuple[float, float, float], seed: int):
	rng = random.Random(seed)
	shuffled = list(paths)
	rng.shuffle(shuffled)

	train_count, val_count, _ = compute_counts(len(shuffled), splits)

	train_paths = shuffled[:train_count]
	val_paths = shuffled[train_count : train_count + val_count]
	test_paths = shuffled[train_count + val_count :]

	return train_paths, val_paths, test_paths


def copy_group(files: list[Path], target_dir: Path) -> int:
	target_dir.mkdir(parents=True, exist_ok=True)
	copied = 0
	for src in files:
		dst = target_dir / src.name
		if dst.exists():
			dst = target_dir / f"{src.stem}_{abs(hash(str(src))) % 1_000_000}{src.suffix.lower()}"
		shutil.copy2(src, dst)
		copied += 1
	return copied


def split_dataset(
	neg_dir: Path,
	pos_dir: Path,
	output_dir: Path,
	splits: tuple[float, float, float],
	seed: int,
) -> None:
	neg_paths = list(iter_images(neg_dir))
	pos_paths = list(iter_images(pos_dir))

	if not neg_paths:
		raise ValueError(f"Keine negativen Bilder gefunden in: {neg_dir}")
	if not pos_paths:
		raise ValueError(f"Keine positiven Bilder gefunden in: {pos_dir}")

	neg_split = split_paths(neg_paths, splits, seed)
	pos_split = split_paths(pos_paths, splits, seed)

	split_to_idx = {"train": 0, "val": 1, "test": 2}
	class_dirs = {"n": neg_split, "y": pos_split}

	totals = {"train": 0, "val": 0, "test": 0}

	for split_name in SPLIT_NAMES:
		idx = split_to_idx[split_name]
		for class_name, groups in class_dirs.items():
			copied = copy_group(groups[idx], output_dir / split_name / class_name)
			totals[split_name] += copied

	print("Split abgeschlossen.")
	print(f"Negativ (n): {len(neg_paths)} Bilder")
	print(f"Positiv (aug_y -> y): {len(pos_paths)} Bilder")
	print(f"Train: {totals['train']} Bilder")
	print(f"Val:   {totals['val']} Bilder")
	print(f"Test:  {totals['test']} Bilder")
	print(f"Output: {output_dir}")


def parse_args():
	parser = argparse.ArgumentParser(
		description="Splittet Bilder aus n (negativ) und aug_y (positiv) in train/val/test."
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
		default="../data/zürich/aug_y/",
		help="Ordner mit positiv gelabelten Bildern (Standard: ../data/zürich/aug_y/)",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default="../data/zürich/split/",
		help="Output-Ordner fuer train/val/test (Standard: ../data/zürich/split/)",
	)
	return parser.parse_args()


def main():
	args = parse_args()

	splits = (0.7, 0.15, 0.15)
	if abs(sum(splits) - 1.0) > 1e-9:
		raise ValueError(f"Split-Ratios muessen 1.0 ergeben, aktuell: {splits}")

	neg_dir = resolve_from_script_dir(args.neg_dir)
	pos_dir = resolve_from_script_dir(args.pos_dir)
	output_dir = resolve_from_script_dir(args.output_dir)

	if not neg_dir.exists() or not neg_dir.is_dir():
		raise FileNotFoundError(f"Negativ-Ordner existiert nicht oder ist kein Ordner: {neg_dir}")
	if not pos_dir.exists() or not pos_dir.is_dir():
		raise FileNotFoundError(f"Positiv-Ordner existiert nicht oder ist kein Ordner: {pos_dir}")

	seed = 42

	output_dir.mkdir(parents=True, exist_ok=True)
	split_dataset(
		neg_dir=neg_dir,
		pos_dir=pos_dir,
		output_dir=output_dir,
		splits=splits,
		seed=seed,
	)


if __name__ == "__main__":
	main()
