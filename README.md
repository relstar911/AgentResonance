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

```bash
# Repository klonen
git clone https://github.com/relstar911/AgentResonance.git
cd AgentResonance

# Abhängigkeiten installieren
pip install streamlit matplotlib numpy pandas plotly scikit-learn scipy seaborn sqlalchemy psycopg2-binary

# Streamlit-App starten
streamlit run app.py
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