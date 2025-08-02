# Mario AI - Reinforcement Learning Project

Ein objektorientiertes Python-Projekt, das eine AI mittels Deep Reinforcement Learning trainiert, um 2D-Plattformspiele zu spielen.

## Features

- **Screen Capture**: Automatische Erfassung von beliebigen Windows-Anwendungen
- **Deep Q-Network (DQN)**: Fortgeschrittenes Reinforcement Learning mit Experience Replay
- **Objektorientierten Architektur**: Saubere Trennung der Komponenten
- **GUI Interface**: Benutzerfreundliche Oberfläche zur Steuerung
- **Live-Monitoring**: Echtzeit-Anzeige des Trainingsfortschritts
- **Modell-Persistierung**: Automatisches Speichern und Laden von trainierten Modellen

## Projektstruktur

```
MarioAI/
├── main.py                 # Hauptanwendung mit GUI
├── screen_capture.py       # Bildschirmaufnahme-System
├── game_controller.py      # Spielsteuerung über Pfeiltasten
├── ai_agent.py            # Deep Q-Network Agent
├── game_environment.py    # Reinforcement Learning Umgebung
├── training_manager.py    # Training-Management
├── requirements.txt       # Python-Abhängigkeiten
└── models/               # Gespeicherte AI-Modelle
```

## Klassen-Architektur

### ScreenCapture
- Erfassung von Windows-Anwendungen
- Live-Frame-Bereitstellung
- Bildvorverarbeitung für AI

### GameController
- Automatische Tasteneingaben (Pfeiltasten + A-Taste zum Springen)
- Action-Mapping für AI-Aktionen
- Manuelle Spielsteuerung für Tests

### DQNAgent
- Deep Q-Network Implementation
- Experience Replay Buffer
- Epsilon-Greedy Exploration
- Modell-Persistierung

### GameEnvironment
- Reward-System für Reinforcement Learning
- Episode-Management
- Zustandsrepräsentation
- Fortschrittserkennung

### TrainingManager
- Koordination des Trainingsprozesses
- Statistik-Tracking
- Modell-Verwaltung

## Installation

1. **Repository klonen oder Dateien kopieren**
2. **Python-Umgebung einrichten**:
   ```bash
   cd MarioAI
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. **Abhängigkeiten installieren**:
   ```bash
   pip install -r requirements.txt
   ```

## Verwendung

### Anwendung starten
```bash
python main.py
```

### Grundlegende Schritte:

1. **Fenster auswählen**: Wählen Sie das Spiel-Fenster aus der Liste
2. **Screen Capture starten**: Beginnen Sie mit der Bildschirmaufnahme
3. **Training starten**: Konfigurieren Sie die Anzahl der Episoden und starten Sie das Training

### Manuelle Steuerung (zum Testen):
- Verwenden Sie den "Manuell spielen" Button
- Steuerung mit Pfeiltasten + **A-Taste zum Springen**
- ESC zum Beenden

## Reinforcement Learning Details

### Reward-System:
- **+10 Punkte** für Fortschritt nach rechts
- **+0.1 Punkte** für Rechtsbewegung
- **+0.05 Punkte** für Springen (kann strategisch nützlich sein)
- **-0.1 Punkte** für Linksbewegung
- **-0.5 Punkte** für längeren Stillstand
- **+0.01 Punkte** kontinuierlich für Überleben

### Netzwerk-Architektur:
- **Input**: 4 gestapelte Grayscale-Frames (84x84)
- **Conv Layers**: 32, 64, 64 Filter
- **FC Layers**: 512 Hidden Units
- **Output**: 5 Aktionen (Nichts, Links, Rechts, Springen, Runter)

### Hyperparameter:
- Learning Rate: 0.0001
- Gamma (Discount): 0.99
- Epsilon Decay: 100,000 Steps
- Batch Size: 32
- Replay Buffer: 100,000 Erfahrungen

## Erweiterte Features

### Modell-Verwaltung:
- Automatisches Speichern alle 5 Minuten
- Laden des zuletzt gespeicherten Modells
- Export/Import von Trainingsfortschritt

### Monitoring:
- Live-Statistiken während des Trainings
- Reward- und Loss-Plots
- Performance-Metriken

### Anpassbarkeit:
- Einfache Erweiterung für andere Spiele
- Konfigurierbare Reward-Funktionen
- Modulare Architektur für neue Features

## Systemanforderungen

- **OS**: Windows 10/11
- **Python**: 3.8+
- **RAM**: 8GB+ (empfohlen)
- **GPU**: CUDA-kompatible GPU (optional, aber empfohlen)

## Troubleshooting

### Häufige Probleme:

1. **"Kein Fenster gefunden"**:
   - Stellen Sie sicher, dass das Spiel läuft
   - Aktualisieren Sie die Fensterliste

2. **Training startet nicht**:
   - Überprüfen Sie, ob Screen Capture aktiv ist
   - Stellen Sie sicher, dass ein Fenster ausgewählt ist

3. **Schlechte Performance**:
   - Reduzieren Sie die Fenstergröße des Spiels
   - Verwenden Sie eine GPU für das Training

## Lizenz

Dieses Projekt ist für Bildungszwecke erstellt. Bitte beachten Sie die Lizenzbedingungen der verwendeten Bibliotheken.

## Autor

Entwickelt als objektorientiertes Reinforcement Learning System für 2D-Plattformspiele.
