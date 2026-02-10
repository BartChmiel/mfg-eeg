# MFG-EEG: Time-Delay Multi-Feature Correlation Analysis

Implementacja metody TD-MFCA do analizy dynamicznych sieci funkcjonalnych mózgu (na podstawie danych EEG Kaggle Grasp-and-Lift).

System pozwala na:
1. Ekstrakcję nieliniowych cech sygnału (HCR - Hierarchical Correlation Reconstruction).
2. Redukcję wymiarowości za pomocą PCA na globalnej, zbalansowanej bazie.
3. Wizualizację przepływu informacji (Granger-like) między regionami mózgu w różnych fazach ruchu.

---

## Setup

### 1) Create virtual environment (Python 3.10+)
```bash
uv venv --python 3.12
# lub standardowo: python -m venv .venv