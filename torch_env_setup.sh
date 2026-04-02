#!/bin/bash

#SBATCH --output="slurm-%j.out"  ## Im Verzeichnis aus dem sbatch aufgerufen wird, wird ein Logfile mit dem Namen slurm-[Jobid].out erstellt.
#SBATCH --error="slurm-%j.err"   ## Ähnlich wie --output. Jedoch ein Log für Fehlermeldungen.
#SBATCH --time=1:00:00           ## Zeitlimite. Diese sollte gleich oder kleiner der Partitions Zeitlimite sein.
#SBATCH --job-name="Tensorflow-Keras Installation"   ## Job Name.
#SBATCH --partition=students	 ## Partitionsname. Die zur Verfügung stehenden Partitionen können mit dem Befehl sinfo angezeigt werden
#SBATCH --mem=20G               ## Der Arbeitsspeicher, welcher für den Job reserviert wird
#SBATCH --cpus-per-task=4        ## Die Anzahl virtueller Cores, die für den Job reserviert werden

# Initialisierung von Conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Erstellung des Conda Environments basierend auf Python 3.13
conda create -n torch python=3.13 -y

# Aktivieren der erstellten Umgebung
conda activate torch

# Upgraden von pip zur Sicherheit bei allfälligen Channel Änderungen 
pip install --upgrade pip

echo "Torch Installation"
# Separate PyTorch Installation zur Sicherheit
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "Other Libraries"
# Installation der im Kurs verwendeten Bibliotheken
pip install numpy pandas scikit-learn plotly matplotlib

# "Erfolgsmeldung" :)
echo "Installation abgeschlossen"
