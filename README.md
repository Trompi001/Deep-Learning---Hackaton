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
* **Aufteilung & Datensatzgrösse:**
  Der Ausgangsdatensatz umfasst insgesamt **46'400 Bilder** (44'234 negative `n` und 2'166 originale positive `y`). Die Aufteilung erfolgt nach dem standardmässigen 70 / 15 / 15-Verhältnis:
  * **Train (70%):** 32'479 Bilder
  * **Val (15%):** 6'959 Bilder
  * **Test (15%):** 6'962 Bilder

---

## 🧠 Modelltraining im Detail

Das Projekt vergleicht zwei Haupt-Modellansätze: Ein von Grund auf trainiertes, optimiertes **Custom CNN** sowie ein Transfer-Learning-Modell basierend auf einem **vortrainierten ResNet18**.

---

### 1️⃣ Custom CNN: Zweistufiges Training mit Hyperparameter-Tuning

Das Custom CNN basiert auf einer einfachen, dreistufigen Faltungsstruktur (`SimpleCNN`). Das Training dieses Modells folgt einem zweistufigen Prozess: Erst werden die optimalen Hyperparameter mittels Optuna-Tuning gesucht, und anschliessend wird das finale Modell mit diesen optimalen Parametern trainiert.

#### 🔹 Schritt A: Hyperparameter-Tuning mit Optuna (`project/03_optuna_model_optimization.py`)
Vor dem eigentlichen Haupttraining wird eine automatisierte, intelligente Suche nach der optimalen Parameterkonfiguration für das Custom CNN durchgeführt.

* **Ablauf & Pruning:**
  * Optuna führt eine definierte Anzahl von Durchläufen (Trials, standardmässig 30) aus. In jedem Trial wird das Modell mit einer bestimmten Hyperparameter-Kombination für eine geringe Epochenanzahl (Standard: 8) trainiert.
  * **MedianPruner:** Nach jeder Epoche gleicht Optuna den aktuellen Validierungsverlust mit den Median-Verläufen früherer Trials ab. Liegt der Trial signifikant zurück, wird er sofort abgebrochen (gepruned), was immense Rechenzeit einspart.
  * **Training from Scratch:** Um die volle Unabhängigkeit des Tunings und die direkte Übertragbarkeit der Ergebnisse zu gewährleisten, wird komplett auf einen Warm-Start verzichtet. Alle Trials starten mit einer zufälligen Gewichtsinitialisierung.
* **Suchraum der Hyperparameter:**
  * Lernrate (`lr`): Logarithmisch-gleichmässig verteilt zwischen $10^{-5}$ und $5 \cdot 10^{-3}$.
  * Optimizer: Kategorische Auswahl aus `adam`, `adamw` und `sgd`.
  * Weight Decay: Logarithmisch-gleichmässig verteilt zwischen $10^{-6}$ und $10^{-2}$.
  * Dropout-Rate: Schrittweise Optimierung von $0.0$ bis $0.6$.
  * Bildgrösse: Skalierung der Eingabebilder auf $96$, $128$ oder $160$ Pixel.
  * Batch-Grösse: Auswahl aus $32$, $64$, $128$ oder $256$ Bildern.
  * Oversampling-Faktor (`positive_multiplier`): Ganzzahlige Werte zwischen $1$ und $6$.

#### 🔹 Schritt B: Finales Training mit den ermittelten Parametern (`project/02.1_train_CNN_model.py`)
Nach Beendigung der Suche werden die gefundenen Optimalparameter direkt in das Haupttrainingsskript des Custom CNN übertragen, um das Modell über die volle Epochenanzahl mit maximaler Patience zu trainieren und den finalen Checkpoint (`CNN_model.pt`) zu erstellen.

* **Modellstruktur (`SimpleCNN`):**
  * **Feature Extractor (Faltungsbasis):** Drei aufeinanderfolgende Convolutional-Blöcke. Jeder Block besteht aus einer Faltungsschicht (`nn.Conv2d` mit 3x3-Kernel, Stride=1, Padding=1), einer ReLU-Aktivierungsfunktion (`nn.ReLU`) und einer Max-Pooling-Schicht (`nn.MaxPool2d` mit einer Fenstergrösse von 2x2). Die Kanaltiefe erhöht sich stufenweise von 3 (RGB-Kanäle) auf 16, 32 und schliesslich 64 Kanäle.
  * **Klassifikationskopf (Classifier):** Ein globales Mittelwert-Pooling (`nn.AdaptiveAvgPool2d((1, 1))`) reduziert die räumlichen Dimensionen der letzten Feature-Map auf 1x1. Nach dem Glätten (`nn.Flatten`) folgt eine Dropout-Schicht zur Regularisierung (durch Optuna auf `p=0.2` optimiert) sowie ein linearer Layer (`nn.Linear`), der die finalen 2 Logits für die binäre Klassifikation ausgibt.
* **Die ermittelten Optimal-Hyperparameter:**
  * **Optimizer:** `AdamW` (entkoppeltes Weight Decay)
  * **Lernrate (Learning Rate):** $0.001239789430250594$ (~ $1.24 \cdot 10^{-3}$)
  * **Weight Decay:** $0.0018452087946386937$ (~ $1.85 \cdot 10^{-3}$)
  * **Dropout-Wahrscheinlichkeit:** $0.2$
  * **Batch-Grösse:** 32
  * **Oversampling-Faktor (`positive_multiplier`):** 5 (Index-Oversampling der unterrepräsentierten positiven Klasse `y` zur Balancierung).
  * **Eingangs-Bildgrösse:** 128x128 Pixel
* **Optimierung & Trainingssteuerung:**
  * **Maximale Epochen:** 100
  * **Verlustfunktion:** Standard `nn.CrossEntropyLoss` (ungewichtet, da die Balance bereits durch das Oversampling hergestellt wird).
  * **Early Stopping:** Geduldszeit (Patience) von 6 Epochen mit einem Schwellenwert (`min_delta`) von $10^{-4}$ auf dem Validierungsverlust.
* **Datenaugmentierung (nur im Training):**
  * Zur Erhöhung der Generalisierungsfähigkeit werden die Trainingsbilder bei jedem Batch on-the-fly transformiert:
    * Zufällige Rotation um $0^\circ, 90^\circ, 180^\circ$ oder $270^\circ$ (`transforms.RandomChoice`).
    * Zufällige horizontale Spiegelung (`transforms.RandomHorizontalFlip`, `p=0.5`).

---

### 2️⃣ Transfer Learning mit ResNet18 (`project/02.2_train_pre-trained_model.py`)

Dieses Skript nutzt ein auf ImageNet vor-trainiertes ResNet18-Modell und adaptiert es auf die Zielklassen.

* **Transfer-Learning-Ansatz (Fine-Tuning vs. Feature Extraction):**
  * **Methode:** **Fine-Tuning (Feinabstimmung)**.
  * **Begründung:** Im Gegensatz zur reinen *Feature Extraction*, bei der alle Gewichte des vor-trainierten Backbone-Netzwerks eingefroren werden (`requires_grad = False` für alle Layer ausser dem neuen Klassifikationskopf), werden beim *Fine-Tuning* **alle Gewichte des gesamten Netzwerks** trainiert. Da das ResNet18 bereits ein exzellentes, universelles visuelles Verständnis besitzt (z. B. für Kanten, Kontraste und geometrische Muster), wird das gesamte Netzwerk mit einer sehr kleinen Lernrate ($5 \cdot 10^{-5}$) trainiert. Dadurch können sich die Filter des Backbones leicht an die spezifischen Texturen und geometrischen Eigenschaften von Fussgängerstreifen anpassen, ohne das erlernte Basiswissen zu zerstören. Dies führt zu einer tieferen Repräsentationsfähigkeit und maximiert die Generalisierung sowie den **Recall** auf der Zielaufgabe.
* **Modellstruktur:**
  * **Backbone:** ResNet18 (vortrainiert auf ImageNet mit den standardmässig geladenen Gewichten `ResNet18_Weights.DEFAULT`).
  * **Klassifikationskopf:** Die ursprüngliche vollvernetzte Schicht (`model.fc`) des ResNets (512 Eingangsmerkmale) wird durch eine neu initialisierte lineare Schicht (`nn.Linear(512, 2)`) mit 2 Ausgangskanälen für die binäre Entscheidung ersetzt.
* **Hyperparameter & Trainingssteuerung:**
  * **Optimizer:** `AdamW` (L2-Regularisierung mittels entkoppeltem Weight Decay von $10^{-3}$ zur Minimierung von Overfitting).
  * **Lernrate (Learning Rate):** $5 \cdot 10^{-5}$ (sehr klein, um vortrainierte Filter schonend anzupassen).
  * **Lernraten-Scheduler:** `CosineAnnealingLR` senkt die Lernrate über die Epochen hinweg sanft entlang einer Kosinuskurve ab, was eine präzise Konvergenz ermöglicht.
  * **Batch-Grösse:** 128
  * **Maximale Epochen:** 30 (aufgrund des Transfer-Learnings ist die Konvergenz deutlich schneller als bei Training von Grund auf).
  * **Eingangs-Bildgrösse:** 128x128 Pixel
  * **Datenaugmentierung & Normalisierung:**
    * Augmentierung analog zum Custom CNN.
    * **Normalisierung:** Die Bildpixel werden an den ImageNet-Mittelwert `[0.485, 0.456, 0.406]` und die Standardabweichung `[0.229, 0.224, 0.225]` angepasst.
  * **Early Stopping:** Geduldszeit (Patience) von 5 Epochen mit einem sensitiven Schwellenwert (`min_delta`) von $10^{-6}$ auf dem Validierungsverlust.
  * **Kompensation der Klassenimbalance (WeightedRandomSampler):**
    * Statt Bilder explizit zu duplizieren, wird PyTorchs `WeightedRandomSampler` im DataLoader verwendet. Jedes Bild erhält ein gewicht, welches dem Kehrwert der Klassen-Häufigkeit entspricht (seltene positive Bilder erhalten ein ca. 20-fach höheres Gewicht als negative).
    * Der DataLoader zieht Bilder basierend auf diesen Gewichten, sodass jeder Batch im Schnitt eine ausgewogene 50/50-Klassenverteilung aufweist.

---

## 📊 Visualisierungen & Evaluierung

Nach jedem Trainingslauf werden zur Analyse der Modellgüte automatisch Diagramme im Ordner `project/plot/` erzeugt:

* **Lernkurven (`learning_curve.png`):**
  * Enthält zwei Teil-Diagramme: Den Verlustverlauf (Loss) und den Genauigkeitsverlauf (Accuracy) jeweils für das Trainings- und das Validierungsset über den Verlauf der Epochen. Dient zur Identifikation von Over- oder Underfitting.
* **Confusion Matrix (`confusion_matrix.png`):**
  * Visualisiert die Richtig- und Falsch-Klassifikationen (True Positives, False Positives, True Negatives, False Negatives) des finalen Modells auf dem Testdatensatz. Liefert detaillierte Informationen darüber, wie verlässlich die seltene positive Klasse `y` erkannt wird.

---

## 🏆 5. Ergebnisse der Klassifikation

Die Evaluierung der trainierten Modelle auf dem unabhängigen **Test-Set**  ergab folgende Ergebnisse:

| Modell | Test Loss | Accuracy | Recall (Sensitivität) | F1-Score | Imbalance-Kompensation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Optuna-optimiertes Custom CNN (Finale)** | **0.0128** | **99.67%** | **96.71%** | **96.56%** | Oversampling (`positive_multiplier = 5`) |
| **Pre-trained ResNet18** | 0.0207 | 99.35% | 95.21% | 93.39% | `WeightedRandomSampler` |

---

## 🎯 6. Fazit & Schlussfolgerung (Bestes Modell)

Nach der Hyperparameter-Optimierung mittels Optuna und dem anschliessenden Training des Custom CNN mit den gefundenen optimalen Parametern ist das **Custom CNN (Optimiert)** das mit Abstand **beste Modell** in allen gemessenen Metriken.

### Begründung:
* **Höchster Recall (Sensitivität):** Da die Erkennung von Fussgängerstreifen eine sicherheitsrelevante und kartografische Aufgabe darstellt, liegt das Hauptaugenmerk auf dem **Recall** (um so wenig existierende Fussgängerstreifen wie möglich zu übersehen). Das optimierte Custom CNN erzielt mit **96.71%** den besten Recall und übertrifft somit auch das vortrainierte ResNet18 (95.21%).
* **Beste Gesamtmetriken:** Mit einem F1-Score von **96.56%**, einer Accuracy von **99.67%** und einem minimalen Test Loss von **0.0128** liefert das Modell die präzisesten Vorhersagen bei gleichzeitig sehr geringen Falsch-Klassifikationsraten.
* **Effektive Regularisierung & Balance:** Die Optimierung des Oversampling-Faktors auf `positive_multiplier = 5` in Kombination mit der Wahl des `AdamW`-Optimizers und dem Weight Decay ($1.85 \cdot 10^{-3}$) hat das Klassenungleichgewicht exzellent gelöst und Overfitting wirksam unterbunden.