#!/bin/bash


#SBATCH --output="logs/slurm-%j.out"  ## Im Verzeichnis aus dem sbatch aufgerufen wird, wird ein Logfile mit dem Namen slurm-[Jobid].out erstellt.
#SBATCH --error="logs/slurm-%j.err"   ## Ähnlich wie --output. Jedoch ein Log für Fehlermeldungen.
#SBATCH --time=1:30:00           ## Zeitlimite. Diese sollte gleich oder kleiner der Partitions Zeitlimite sein. In diesem Fall ist diese auf 1 Stunde und 30 Minuten gesetzt.
#SBATCH --job-name="Pytorch Skript"   ## Job Name.
#SBATCH --partition=students	 ## Partitionsname. Die zur Verfügung stehenden Partitionen können mit dem Befehl sinfo angezeigt werden
#SBATCH --mem=50G		## Der Arbeitsspeicher, welcher für den Job reserviert wird	
#SBATCH --cpus-per-task=16  ## Die Anzahl virtueller Cores, die für den Job reserviert werden
#SBATCH --gpus=a100:1	 ## Die Anzahl GPUs (hier eine GPU, mit der Syntax :1)


script_name="project/02_train_model.py" # Hier kann der Name des Skriptes ausgetauscht werden. Alternativ kann hier auch ein absoluter Pfad zum Skript angegeben werden.

# Live-Logs: conda darf nicht capturen, Python schreibt ungepuffert.
export PYTHONUNBUFFERED=1
conda run --no-capture-output -n torch python3 -u "$script_name"
