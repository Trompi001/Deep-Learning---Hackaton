import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# ---------------------------
# Einstellungen
# ---------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

DATA_DIR = ROOT_DIR / "data" / "winterthur" / "n"
MODEL_PATH = SCRIPT_DIR / "models" / "cnn_zurich_optuna_best.pt"
OUTPUT_DIR = ROOT_DIR / "data" / "winterthur" / "positives"
OUTPUT_JSON = ROOT_DIR / "data" / "winterthur" / "positives.json"

IMAGE_SIZE = 128
THRESHOLD = 0.5
MOVE_FILES = True  # True = verschieben, False = kopieren


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


# ---------------------------
# Modell laden
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location=device)

if isinstance(checkpoint, nn.Module):
    model = checkpoint.to(device)
elif isinstance(checkpoint, dict):
    model = SimpleCNN(num_classes=2).to(device)
    model.load_state_dict(checkpoint)
else:
    raise TypeError(
        f"Unsupported checkpoint type at {MODEL_PATH}: {type(checkpoint).__name__}"
    )

model.eval()


# ---------------------------
# Bild-Transformation
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# ---------------------------
# Bilder sammeln
# ---------------------------
image_paths = list(DATA_DIR.rglob("*.*"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

positives = []


# ---------------------------
# Hauptloop
# ---------------------------
with torch.no_grad():
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            continue

        x = transform(img).unsqueeze(0).to(device)

        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0][1].item()

        if prob >= THRESHOLD:
            dst_path = OUTPUT_DIR / img_path.name

            # Duplikate vermeiden
            if dst_path.exists():
                dst_path = OUTPUT_DIR / f"{img_path.stem}_copy{img_path.suffix}"

            if MOVE_FILES:
                shutil.move(str(img_path), str(dst_path))
            else:
                shutil.copy2(str(img_path), str(dst_path))

            positives.append({
                "image": str(dst_path),
                "prob": round(prob, 4)
            })


# ---------------------------
# JSON speichern
# ---------------------------
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(positives, f, indent=2, ensure_ascii=False)


print(f"Fertig! {len(positives)} positive Bilder gefunden.")