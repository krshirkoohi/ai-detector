# ai-detector
Open source application written in Python to check for AI content in paragraph input.

***

## How to get started

### Install dependencies

python -m venv venv && source venv/bin/activate
pip install transformers torch scikit-learn numpy textstat joblib datasets

### Download the small backbone models

python ai_humaniser.py --download-models

### Run interactively

python ai_humaniser.py

Paste a paragraph, hit ↵ twice, and you’ll get an AI-likelihood score plus a friendly verdict.

### Benchmark against GPTZero

If you have a CSV with columns `text,gptzero_score`, load it in a notebook:

```python
import pandas as pd
from ai_humaniser import Detector
det = Detector.load()
df = pd.read_csv("gptzero_eval.csv")
df["our_prob"] = df["text"].apply(det.predict_proba)
mape = det.mape_vs_gptzero(df["text"].tolist(), df["gptzero_score"].tolist())
print("MAPE vs GPTZero:", mape)
```

***

## Tuning & extending
* Retrain with your own labelled data:

python ai_humaniser.py --retrain

The script automatically builds a new dataset (10 k human + 10 k AI paragraphs) and trains a logistic-regression classifier on the richer feature set in the code.
* Add extra features or swap in a different classifier – the Detector and _FeatureExtractor classes are cleanly separated; modify _FeatureExtractor.extract() or switch to an XGBoost model for potentially higher accuracy.
* API integration – wrap Detector.load() inside a Flask/FastAPI route to turn the script into a lightweight web service.