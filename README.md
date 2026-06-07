# Deep-Learning-Pipeline (Hackathon)

Klassifikation von Bildern mit PyTorch (binär: `n` vs. `y`) mit Fokus auf Datenaufteilung (Train/Val/Test), Kompensation von Klassenimbalance (Oversampling & Weighted Sampling), Hyperparameter-Tuning (Optuna), Transfer Learning (ResNet18) und Active Learning / Dataset Improvement.

---

## 📂 Verzeichnisstruktur (Verzeichnisbaum)

```text
Deep-Learning---Hackaton/
├── data/                      # Lokale Bilddaten
│   └── zürich/
│       ├── n/                 # Negative Bilder (Klasse n)
│       ├── y/                 # Positiv-Bilder (Klasse y)
│       └── split/             # Train/Val/Test Splits
│           ├── train/ {n,y}
│           ├── val/ {n,y}
│           └── test/ {n,y}
├── project/                   # Quellcode & Pipeline-Skripte
│   ├── 01_data_split.py       # Schritt 1: Datensatz splitten
│   ├── 02.1_train_CNN_model.py # Schritt 2.1: Custom CNN trainieren (Oversampling)
│   ├── 02.2_train_pre-trained_model.py # Schritt 2.2: ResNet18 Transfer Learning (Weighted Sampling)
│   ├── 03_optuna_model_optimization.py # Automatisiertes Hyperparameter-Tuning mit Optuna
│   └── plot/                  # Lernkurven & Confusion Matrices (Git-tracked)
├── logs/                      # Slurm Logs (Stderr/Stdout)
├── torch_env_setup.sh         # Conda & Pip Environment Setup
├── run_torch_script.sh        # Slurm Job-Skript für Modelltraining
└── README.md                  # Projektdokumentation (dieses File)
```

---

## 🚀 Datenvorbereitung & Split

Das Skript `project/01_data_split.py` teilt die Rohbilder aus den Klassenordnern `n` (negativ) und `y` (positiv) in ein strukturiertes `train/val/test`-Verzeichnis auf.
* **Stratified Split:** Der Split erfolgt so, dass die ursprüngliche Verteilung der Klassen in allen drei Datensätzen (Training, Validierung und Test) proportional erhalten bleibt.
* **Seed-Fixierung:** Durch einen festen Zufalls-Seed (`--seed`) bleibt die Aufteilung über verschiedene Durchläufe hinweg deterministisch und somit reproduzierbar.
* **Aufteilung & Datensatzgröße:**
  Der Ausgangsdatensatz umfasst insgesamt **46'400 Bilder** (44'234 negative `n` und 2'166 originale positive `y`). Die Aufteilung erfolgt nach dem standardmäßigen 70 / 15 / 15-Verhältnis:
  * **Train (70%):** 32'479 Bilder
  * **Val (15%):** 6'959 Bilder
  * **Test (15%):** 6'962 Bilder


---

## 🧠 Modelltraining im Detail

Das Projekt untersucht zwei Trainingsansätze sowie ein automatisiertes Hyperparameter-Tuning.

### 1️⃣ Training des Custom CNN (`project/02.1_train_CNN_model.py`)

Dieses Modul trainiert ein einfaches, dreistufiges Faltungsnetzwerk (CNN) von Grund auf.

* **Modellarchitektur (`SimpleCNN`):**
  * **Feature Extractor:** Drei aufeinanderfolgende Convolutional-Blöcke. Jeder Block besteht aus einer Faltungsschicht (`nn.Conv2d` mit 3x3 Kernel, Stride=1 und Padding=1), einer Aktivierungsfunktion (`nn.ReLU`) und einer Pooling-Schicht (`nn.MaxPool2d` mit einer Fenstergröße von 2x2). Die Anzahl der Feature-Maps erhöht sich stufenweise von 3 (RGB-Kanäle) auf 16, 32 und schließlich 64 Kanäle.
  * **Klassifikationskopf:** Ein globales Mittelwert-Pooling (`nn.AdaptiveAvgPool2d((1, 1))`) reduziert die räumlichen Dimensionen der letzten Feature-Map auf 1x1, wodurch das Modell invariant gegenüber wechselnden Bildgrößen wird. Nach dem Glätten (`nn.Flatten`) folgt eine Dropout-Schicht (`p=0.3`) zur Regularisierung gegen Overfitting sowie ein linearer Layer (`nn.Linear`), der die finalen 2 Logits für die binäre Entscheidung ausgibt.
* **Kompensation der Klassenimbalance (Oversampling):**
  * Da die Klasse `y` stark unterrepräsentiert ist, wird ein **Oversampling** angewendet. Über einen Multiplikator (`positive_multiplier`) werden die Indizes der positiven Bilder im Trainingsdatensatz repliziert. Dadurch zieht der DataLoader diese Bilder häufiger, was das Ungleichgewicht ausgleicht.
* **Datenaugmentierung (nur im Training):**
  * Zur Erhöhung der Generalisierungsfähigkeit werden die Trainingsbilder bei jedem Batch on-the-fly transformiert:
    * Zufällige Rotation um $0^\circ, 90^\circ, 180^\circ$ oder $270^\circ$ (`transforms.RandomChoice`).
    * Zufällige horizontale Spiegelung (`transforms.RandomHorizontalFlip`, `p=0.5`).
* **Optimierung & Trainingssteuerung:**
  * **Verlustfunktion:** Standard `nn.CrossEntropyLoss` (ungewichtet, da die Balance bereits durch die Datenstruktur des Dataloaders hergestellt wird).
  * **Optimizer:** `Adam` mit einer Lernrate von $10^{-3}$.
  * **Early Stopping:** Das Training bricht vorzeitig ab, wenn sich der Validierungsverlust über einen Zeitraum von `early-stopping-patience` (Standard: 6 Epochen) nicht um mindestens `early-stopping-min-delta` ($10^{-4}$) verbessert. Der beste Modellzustand wird gespeichert.

---

### 2️⃣ Transfer Learning mit ResNet18 (`project/02.2_train_pre-trained_model.py`)

Dieses Skript nutzt ein auf ImageNet vor-trainiertes ResNet18-Modell und adaptiert es auf die Zielklassen.

* **Modellarchitektur & Fine-Tuning:**
  * Das ResNet18-Backbone wurde auf über einer Million Bildern trainiert und besitzt bereits universelle visuelle Repräsentationen (Kanten, Formen, Strukturen).
  * Der ursprüngliche vollvernetzte Layer (`model.fc`) des ResNets wird durch einen neuen linearen Klassifikationslayer (`nn.Linear`) mit 2 Ausgangskanälen ersetzt.
  * Da die Gewichte des Backbones bereits hochgradig optimiert sind, wird das gesamte Netzwerk mit einer sehr kleinen Lernrate ($5 \cdot 10^{-5}$) feingetunt, um das vortrainierte Wissen zu erhalten und nicht zu destabilisieren.
* **Kompensation der Klassenimbalance (WeightedRandomSampler):**
  * Statt Bilder explizit zu kopieren, wird hier PyTorchs `WeightedRandomSampler` im DataLoader verwendet. Jedes Bild erhält ein Gewicht, welches dem Kehrwert der Klassen-Häufigkeit entspricht (seltene positive Bilder erhalten ein ca. 20-fach höheres Gewicht als negative).
  * Der DataLoader zieht Bilder basierend auf diesen Gewichten, sodass jeder Batch im Schnitt eine ausgewogene 50/50-Klassenverteilung aufweist.
* **Datenaugmentierung & Normalisierung:**
  * Die Augmentierung erfolgt analog zum Custom CNN.
  * **Normalisierung:** Die Bildpixel werden mit dem ImageNet-Mittelwert `[0.485, 0.456, 0.406]` und der Standardabweichung `[0.229, 0.224, 0.225]` normalisiert, um sie an die Verteilung der Trainingsdaten des Originalmodells anzupassen.
* **Optimierung & Trainingssteuerung:**
  * **Optimizer:** `AdamW` (L2-Regularisierung mittels entkoppeltem Weight Decay von $10^{-3}$ zur Minimierung von Overfitting).
  * **Lernraten-Scheduler:** `CosineAnnealingLR` senkt die Lernrate über die Epochen hinweg sanft entlang einer Kosinuskurve ab. Dies ermöglicht eine präzise Konvergenz gegen Trainingsende.
  * **Early Stopping:** Geduldszeit von 5 Epochen mit einem sehr sensitiven Schwellenwert (`min_delta`) von $10^{-6}$.

---

### 3️⃣ Hyperparameter-Optimierung mit Optuna (`project/04_optuna_model_optimization.py`)

Führt eine automatisierte, intelligente Suche nach der optimalen Parameterkonfiguration für das Custom CNN durch.

* **Ablauf & Pruning:**
  * Optuna führt eine definierte Anzahl von Durchläufen (Trials) aus. In jedem Trial wird das Modell mit einer bestimmten Hyperparameter-Kombination für eine geringe Epochenanzahl (Standard: 8) trainiert.
  * **MedianPruner:** Nach jeder Epoche gleicht Optuna den aktuellen Validierungsverlust mit den Median-Verläufen früherer Trials ab. Liegt der Trial signifikant zurück, wird er sofort abgebrochen (gepruned), was immense Rechenzeit einspart.
  * **Transfer/Warm Start:** Jedes Trial startet mit dem Laden des vortrainierten Basismodells (`Simple_CNN_zurich.pt`) über `strict=False`, um eine kontinuierliche Verbesserung bestehender Gewichte zu erzielen.
* **Suchraum der Hyperparameter:**
  * Lernrate (`lr`): Logarithmisch-gleichmäßig verteilt zwischen $10^{-5}$ und $5 \cdot 10^{-3}$.
  * Optimizer: Kategorische Auswahl aus `adam`, `adamw` und `sgd` (mit Momentum 0.9).
  * Weight Decay: Logarithmisch-gleichmäßig verteilt zwischen $10^{-6}$ und $10^{-2}$.
  * Dropout-Rate: Schrittweise Optimierung von $0.0$ bis $0.6$.
  * Bildgröße: Skalierung der Eingabebilder auf $96$, $128$ oder $160$ Pixel.
  * Batch-Größe: Auswahl aus $32$, $64$, $128$ oder $256$ Bildern.
  * Oversampling-Faktor (`positive_multiplier`): Ganzzahlige Werte zwischen $1$ und $6$.
* **Finalisierung:**
  * Nach Beendigung der Suche wird das Modell mit den besten Parametern über die volle Epochenanzahl (Standard: 100) mit maximaler Patience trainiert und als finaler Checkpoint (`cnn_zurich_optuna_best.pt`) abgelegt.

---

## 🔍 Datensatz verbessern & Active Learning (`project/03_improve_dataset.py`)

Dieses Skript implementiert einen Active-Learning-Ansatz zur qualitativen Aufwertung und Vergrößerung des Datensatzes.

* **Modellgestützte Vorhersage:** Das Skript lädt das beste trainierte Modell (z. B. den Optuna-Best-Checkpoint) und klassifiziert einen ungelabelten oder bisher rein negativen Bildpool (z. B. Winterthur).
* **Wahrscheinlichkeitsfilter:** Mittels einer Softmax-Aktivierung auf den Ausgangs-Logits wird die Wahrscheinlichkeit berechnet, mit der ein Bild zur positiven Klasse `y` gehört.
* **Mining & Export:** Alle Bilder, deren Wahrscheinlichkeit den Schwellenwert (`THRESHOLD`, Standard: 0.5) erreicht, werden kopiert oder verschoben. Parallel dazu wird eine JSON-Datei (`positives.json`) mit den Dateipfaden und den exakten Konfidenzwerten angelegt. Diese gefilterten Bilder können manuell nachkontrolliert werden, um neue, seltene Positivbeispiele für zukünftige Trainingsläufe zu sichern.

---

## 📊 Visualisierungen & Evaluierung

Nach jedem Trainingslauf werden zur Analyse der Modellgüte automatisch Diagramme im Ordner `project/plot/` erzeugt:

* **Lernkurven (`learning_curve.png`):**
  * Enthält zwei Teil-Diagramme: Den Verlustverlauf (Loss) und den Genauigkeitsverlauf (Accuracy) jeweils für das Trainings- und das Validierungsset über den Verlauf der Epochen. Dient zur Identifikation von Over- oder Underfitting.
* **Confusion Matrix (`confusion_matrix.png`):**
  * Visualisiert die Richtig- und Falsch-Klassifikationen (True Positives, False Positives, True Negatives, False Negatives) des finalen Modells auf dem Testdatensatz. Liefert detaillierte Informationen darüber, wie verlässlich die seltene positive Klasse `y` erkannt wird.

---

## 🏆 5. Ergebnisse der Klassifikation

Die Evaluierung der trainierten Modelle auf dem unabhängigen **Test-Set** (Zürich-Datensatz) ergab folgende Ergebnisse:

| Modell | Test Loss | Accuracy | Recall (Sensitivität) | F1-Score | Imbalance-Kompensation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Pre-trained ResNet18** | 0.0207 | 99.35% | **95.21%** | 93.39% | `WeightedRandomSampler` |
| **Optuna-optimiertes Custom CNN** | **0.0158** | **99.55%** | 94.01% | **95.30%** | Keine (`positive_multiplier = 1`) |
| **Custom CNN (Standard)** | 0.0268 | 99.43% | 93.71% | 93.99% | Oversampling (`positive_multiplier = 4`) |

---

## 🎯 6. Fazit & Schlussfolgerung (Bestes Modell)

Obwohl das **Optuna-optimierte Custom CNN** die beste Gesamtgenauigkeit (99.55%) und den besten F1-Score (95.30%) erreicht, ist das **vortrainierte ResNet18-Modell** (Transfer Learning) für diese spezifische Aufgabenstellung als das **beste Modell** zu bewerten.

### Begründung:
* **Fokus auf Recall (Sensitivität):** Bei der Erkennung von Fussgängerstreifen handelt es sich um eine sicherheitsrelevante und kartografische Aufgabe. Ein **False Negative** (ein real existierender Fussgängerstreifen wird übersehen) ist weitaus gravierender als ein **False Positive** (ein fälschlicherweise markierter Streifen, der in einer manuellen Nachkontrolle leicht aussortiert werden kann).
* **Höchster Recall:** Das ResNet18-Modell erzielt mit **95.21%** den höchsten Recall-Wert aller Modelle auf den Testdaten und übersieht somit am wenigsten Fussgängerstreifen (nur 15 von 325 positiven Testbildern wurden verpasst).
* **Effektive Balancierung:** Der `WeightedRandomSampler` sorgt in Kombination mit dem Vortraining für eine extrem hohe Sensitivität gegenüber der seltenen positiven Klasse (`y`).