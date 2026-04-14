# Deep-Learning---Hackaton

Klassifikation von Bildern mit PyTorch (binär: `n` vs. `y`) inklusive Datenaugmentation, Datensplit und CNN-Training.

## Projektüberblick

Die Pipeline besteht aus drei Schritten:

1. `project/00_data_aug.py`: Augmentiert positive Bilder (`y`) nach `aug_y`.
2. `project/01_data_split.py`: Erstellt `train/val/test` aus negativen (`n`) und augmentierten positiven Bildern (`aug_y`).
3. `project/02_train_model.py`: Trainiert ein CNN, speichert bestes Modell und Plots.

## Verzeichnisstruktur (relevant)

- `data/zürich/n/`: negative Bilder
- `data/zürich/y/`: originale positive Bilder
- `data/zürich/aug_y/`: augmentierte positive Bilder (Output Schritt 1)
- `data/zürich/split/{train,val,test}/{n,y}/`: Split-Daten (Output Schritt 2)
- `project/models/`: gespeicherte Modelle
- `project/plot/`: Lernkurven und Confusion Matrix

## Environment einrichten

Das Skript `torch_env_setup.sh` erstellt ein Conda-Environment `torch` und installiert die benötigten Pakete.

```bash
sbatch torch_env_setup.sh
```

Alternativ lokal:

```bash
bash torch_env_setup.sh
```

## Pipeline lokal ausführen

Die Skripte verwenden standardmäßig den Zürich-Datensatz unter `data/zürich/...`.

### 1) Positive Klasse augmentieren

```bash
cd project
conda run -n torch python3 00_data_aug.py
```

Optional mit eigenen Pfaden:

```bash
conda run -n torch python3 00_data_aug.py \
	--input-dir ../data/zürich/y/ \
	--output-dir ../data/zürich/aug_y/
```

### 2) Datensatz in train/val/test splitten

```bash
cd project
conda run -n torch python3 01_data_split.py
```

Optional mit eigenen Split-Ratios:

```bash
conda run -n torch python3 01_data_split.py --splits 0.7 0.15 0.15 --seed 42
```

### 3) Modell trainieren

```bash
cd project
conda run -n torch python3 02_train_model.py
```

Beispiel mit Parametern:

```bash
conda run -n torch python3 02_train_model.py \
	--epochs 10 \
	--batch-size 64 \
	--image-size 128 \
	--lr 1e-3
```

## Training per Slurm starten

`run_torch_script.sh` startet aktuell `project/02_train_model.py` auf GPU (`a100:1`).

```bash
sbatch run_torch_script.sh
```

Logs werden unter `logs/slurm-<jobid>.out` und `logs/slurm-<jobid>.err` geschrieben.

## Outputs

Nach erfolgreichem Training:

- Modell: `project/models/cnn_zurich_best.pt`
- Lernkurve: `project/plot/model_training_learning_curve.png`
- Confusion Matrix: `project/plot/test_confusion_matrix.png`

## Aktuelle Datensatzgrößen (Beispiel)

- Negativ (`n`): 44234 Bilder
- Positiv (`y`): 2166 Bilder
- Train: 32479 Bilder
- Val: 6959 Bilder
- Test: 6962 Bilder