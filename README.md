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

* **Modellstruktur (`SimpleCNN`):**
  * **Feature Extractor (Faltungsbasis):** Drei aufeinanderfolgende Convolutional-Blöcke. Jeder Block besteht aus einer Faltungsschicht (`nn.Conv2d` mit 3x3-Kernel, Stride=1, Padding=1), einer ReLU-Aktivierungsfunktion (`nn.ReLU`) und einer Max-Pooling-Schicht (`nn.MaxPool2d` mit einer Fenstergröße von 2x2). Die Anzahl der Feature-Maps (Kanäle) erhöht sich stufenweise von 3 (RGB-Kanäle) auf 16, 32 und schließlich 64 Kanäle.
  * **Klassifikationskopf (Classifier):** Ein globales Mittelwert-Pooling (`nn.AdaptiveAvgPool2d((1, 1))`) reduziert die räumlichen Dimensionen der letzten Feature-Map auf 1x1, wodurch das Modell invariant gegenüber der Eingangsgröße wird. Nach dem Glätten (`nn.Flatten`) folgt eine Dropout-Schicht (`p=0.3`) zur Regularisierung gegen Overfitting sowie ein linearer Layer (`nn.Linear`), der die finalen 2 Logits für die binäre Klassifikation ausgibt.
* **Hyperparameter & Trainingssteuerung:**
  * **Optimizer:** `Adam`
  * **Lernrate (Learning Rate):** $10^{-3}$ (fest)
  * **Batch-Größe:** 128
  * **Maximale Epochen:** 100
  * **Eingangs-Bildgröße:** 128x128 Pixel
  * **Early Stopping:** Geduldszeit (Patience) von 6 Epochen mit einem Schwellenwert (`min_delta`) von $10^{-4}$ auf dem Validierungsverlust.
  * **Kompensation der Klassenimbalance:** Index-Oversampling der unterrepräsentierten positiven Klasse `y` über den Faktor `positive_multiplier = 4` im DataLoader.
  * **Verlustfunktion:** Standard `nn.CrossEntropyLoss` (ungewichtet, da die Klassenbalance bereits durch das Oversampling hergestellt wird).
* **Datenaugmentierung (nur im Training):**
  * Zur Erhöhung der Generalisierungsfähigkeit werden die Trainingsbilder bei jedem Batch on-the-fly transformiert:
    * Zufällige Rotation um $0^\circ, 90^\circ, 180^\circ$ oder $270^\circ$ (`transforms.RandomChoice`).
    * Zufällige horizontale Spiegelung (`transforms.RandomHorizontalFlip`, `p=0.5`).

---

### 2️⃣ Transfer Learning mit ResNet18 (`project/02.2_train_pre-trained_model.py`)

Dieses Skript nutzt ein auf ImageNet vor-trainiertes ResNet18-Modell und adaptiert es auf die Zielklassen.

* **Transfer-Learning-Ansatz (Fine-Tuning vs. Feature Extraction):**
  * **Methode:** **Fine-Tuning (Feinabstimmung)**.
  * **Begründung:** Im Gegensatz zur reinen *Feature Extraction*, bei der alle Gewichte des vor-trainierten Backbone-Netzwerks eingefroren werden (`requires_grad = False` für alle Layer außer dem neuen Klassifikationskopf), werden beim *Fine-Tuning* **alle Gewichte des gesamten Netzwerks** trainiert. Da das ResNet18 bereits ein exzellentes, universelles visuelles Verständnis besitzt (z. B. für Kanten, Kontraste und geometrische Muster), wird das gesamte Netzwerk mit einer sehr kleinen Lernrate ($5 \cdot 10^{-5}$) trainiert. Dadurch können sich die Filter des Backbones leicht an die spezifischen Texturen und geometrischen Eigenschaften von Fussgängerstreifen anpassen, ohne das erlernte Basiswissen zu zerstören. Dies führt zu einer tieferen Repräsentationsfähigkeit und maximiert die Generalisierung sowie den **Recall** auf der Zielaufgabe.
* **Modellstruktur:**
  * **Backbone:** ResNet18 (vortrainiert auf ImageNet mit den standardmäßig geladenen Gewichten `ResNet18_Weights.DEFAULT`).
  * **Klassifikationskopf:** Die ursprüngliche vollvernetzte Schicht (`model.fc`) des ResNets (512 Eingangsmerkmale) wird durch eine neu initialisierte lineare Schicht (`nn.Linear(512, 2)`) mit 2 Ausgangskanälen für die binäre Entscheidung ersetzt.
* **Hyperparameter & Trainingssteuerung:**
  * **Optimizer:** `AdamW` (L2-Regularisierung mittels entkoppeltem Weight Decay von $10^{-3}$ zur Minimierung von Overfitting).
  * **Lernrate (Learning Rate):** $5 \cdot 10^{-5}$ (sehr klein, um vortrainierte Filter schonend anzupassen).
  * **Lernraten-Scheduler:** `CosineAnnealingLR` senkt die Lernrate über die Epochen hinweg sanft entlang einer Kosinuskurve ab, was eine präzise Konvergenz ermöglicht.
  * **Batch-Größe:** 128
  * **Maximale Epochen:** 30 (aufgrund des Transfer-Learnings ist die Konvergenz deutlich schneller als bei Training von Grund auf).
  * **Eingangs-Bildgröße:** 128x128 Pixel
  * **Datenaugmentierung & Normalisierung:**
    * Augmentierung analog zum Custom CNN.
    * **Normalisierung:** Die Bildpixel werden an den ImageNet-Mittelwert `[0.485, 0.456, 0.406]` und die Standardabweichung `[0.229, 0.224, 0.225]` angepasst.
  * **Early Stopping:** Geduldszeit (Patience) von 5 Epochen mit einem sensitiven Schwellenwert (`min_delta`) von $10^{-6}$ auf dem Validierungsverlust.
  * **Kompensation der Klassenimbalance (WeightedRandomSampler):**
    * Statt Bilder explizit zu duplizieren, wird PyTorchs `WeightedRandomSampler` im DataLoader verwendet. Jedes Bild erhält ein gewicht, welches dem Kehrwert der Klassen-Häufigkeit entspricht (seltene positive Bilder erhalten ein ca. 20-fach höheres Gewicht als negative).
    * Der DataLoader zieht Bilder basierend auf diesen Gewichten, sodass jeder Batch im Schnitt eine ausgewogene 50/50-Klassenverteilung aufweist.

---

### 3️⃣ Hyperparameter-Optimierung mit Optuna (`project/03_optuna_model_optimization.py`)

Führt eine automatisierte, intelligente Suche nach der optimalen Parameterkonfiguration für das Custom CNN durch.

* **Ablauf & Pruning:**
  * Optuna führt eine definierte Anzahl von Durchläufen (Trials) aus. In jedem Trial wird das Modell mit einer bestimmten Hyperparameter-Kombination für eine geringe Epochenanzahl (Standard: 8) trainiert.
  * **MedianPruner:** Nach jeder Epoche gleicht Optuna den aktuellen Validierungsverlust mit den Median-Verläufen früherer Trials ab. Liegt der Trial signifikant zurück, wird er sofort abgebrochen (gepruned), was immense Rechenzeit einspart.
  * **Training from Scratch:** Um die volle Unabhängigkeit des Tunings und die direkte Übertragbarkeit der Ergebnisse zu gewährleisten, wird komplett auf einen Warm-Start verzichtet. Alle Trials starten mit einer zufälligen Gewichtsinitialisierung.
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