# AURA NEXUS — AI Acoustic Fault Detection

Interactive Streamlit web application for the team's machine-sound classification project.

## Deploy
This repository is structured for Streamlit Community Cloud.
- Entrypoint: `streamlit_app.py`
- Dependencies: `requirements.txt`

## Included
- Team-trained Random Forest model and exact feature schema
- Feature dataset (derived features and labels)
- Normal and anomalous WAV demo samples
- Signal Explorer, State Comparison, Sensitivity Lab, Feature Intelligence, and Model & Dataset views

## Scientific scope
The classifier predicts only `Normal (0)` or `Anomalous (1)`. It does not identify a fault subtype.
Amplitude sensitivity is a digital signal-scaling experiment, not a physical fault simulation.
The displayed anomaly probability is the model's class probability, not a calibrated real-world fault probability.
The bundled examples are demonstrations, not a held-out evaluation or field validation.
