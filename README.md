# Agent Resonance Simulation Platform

Eine interaktive Plattform zur Simulation und Analyse von Bewusstsein, Resonanz und Beziehungsdynamiken zwischen "bewussten" Agenten in einer Grid-World-Umgebung.

## Beschreibung

Diese Anwendung bietet ein fortschrittliches Framework zur Modellierung und Erforschung von Bewusstseins-ähnlichen Prozessen und Interaktionen zwischen autonomen Agenten. Die Simulation basiert auf theoretischen Konzepten wie:

- Phi (Integration von Informationen)
- Delta (Entscheidungsentropie)
- Sigma (Selbstprädiktive Genauigkeit)
- Resonanz zwischen Agenten
- Bindungsstärke und Beziehungsdynamik

## Hauptfunktionen

- **Interaktive Simulationen**: Beobachten Sie Agenten in Echtzeit und analysieren Sie ihre Interaktionen
- **Experiment Lab**: Führen Sie kontrollierte Experimente mit Versuchs- und Kontrollgruppen durch
- **Batch-Simulationen**: Führen Sie mehrere Simulationen mit variierenden Parametern zur systematischen Untersuchung durch
- **Datenanalyse**: Analysieren Sie Simulationsergebnisse mit fortschrittlichen statistischen Methoden
- **Datenpersistenz**: Speichern und laden Sie Simulationen und Experimente in/aus der PostgreSQL-Datenbank

## Komponenten

- Streamlit für die interaktive Web-Oberfläche
- Python-basierte Simulationsengine
- PostgreSQL-Datenbank für Datenpersistenz
- Fortgeschrittene Visualisierungen mit Matplotlib und Plotly
- Statistische Analyse mit SciPy

## Installation

### Standard-Installation

```bash
# Repository klonen
git clone https://github.com/relstar911/AgentResonance.git
cd AgentResonance

# Abhängigkeiten installieren
pip install streamlit matplotlib numpy pandas plotly scikit-learn scipy seaborn sqlalchemy psycopg2-binary

# Optional: Für die .env-Datei-Unterstützung
# pip install python-dotenv

# Streamlit-App starten
streamlit run app.py
```

### Lokale Entwicklung mit SQLite

Für die lokale Entwicklung verwendet das System automatisch eine SQLite-Datenbank, wenn keine PostgreSQL-Verbindung konfiguriert ist. Es sind keine weiteren Schritte erforderlich - die Anwendung erstellt automatisch eine Datei 'local_data.db' im Projektverzeichnis.

### Lokale Entwicklung mit PostgreSQL

Wenn Sie stattdessen eine lokale PostgreSQL-Datenbank verwenden möchten:

1. PostgreSQL installieren und einrichten
2. Eine neue Datenbank erstellen
3. Die Verbindung über eine Umgebungsvariable konfigurieren:

```bash
# Umgebungsvariable setzen (Linux/macOS)
export DATABASE_URL="postgresql://benutzername:passwort@localhost:5432/datenbankname"

# Für Windows (PowerShell)
$env:DATABASE_URL="postgresql://benutzername:passwort@localhost:5432/datenbankname"

# Für Windows (CMD)
set DATABASE_URL=postgresql://benutzername:passwort@localhost:5432/datenbankname
```

Alternativ können Sie eine `.env`-Datei im Projektverzeichnis erstellen:

```
DATABASE_URL=postgresql://benutzername:passwort@localhost:5432/datenbankname
```

Und dann mit python-dotenv laden:

```python
from dotenv import load_dotenv
load_dotenv()  # Diese Zeile am Anfang von app.py oder database.py hinzufügen
```

## Abhängigkeiten

- streamlit>=1.30.0
- matplotlib>=3.7.0
- numpy>=1.24.0
- pandas>=2.0.0
- plotly>=5.14.0
- scikit-learn>=1.2.0
- scipy>=1.10.0
- seaborn>=0.12.0
- sqlalchemy>=2.0.0
- psycopg2-binary>=2.9.0

Für die .env-Datei-Unterstützung (optional):
- python-dotenv>=1.0.0

## Benutzung

1. **Simulation**: Führen Sie einzelne Simulationen mit benutzerdefinierten Parametern durch
2. **Experiment Lab**: Entwerfen und führen Sie empirische Experimente durch
3. **Batch-Experimente**: Führen Sie systematische Untersuchungen mit mehreren Parameterkombinationen durch
4. **Gespeicherte Simulationen/Experimente**: Greifen Sie auf frühere Ergebnisse zu und analysieren Sie diese

## Konzept

Die Plattform ist darauf ausgerichtet, theoretische Konzepte des Bewusstseins und der zwischenmenschlichen Resonanz empirisch zu erforschen. Die Agenten verfügen über primitive Formen von "Bewusstsein" mit folgenden Eigenschaften:

- Umgebungswahrnehmung
- Selbstmodellierung
- Entscheidungsfindung basierend auf internen Zuständen
- Bindungsentwicklung zu anderen Agenten
- Resonanzfähigkeit mit anderen Agenten

## Zukunftsentwicklung

- Integration von Lernalgorithmen für adaptive Agenten
- Erweitertes Theoriemodell für Bewusstseinssimulation
- Verbesserte statistische Analysetools
- 3D-Visualisierung für komplexe Simulationen