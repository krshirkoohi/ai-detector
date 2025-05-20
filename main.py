#!/usr/bin/env python3
"""
Simplified AI Text Detector

A streamlined version of the AI Humaniser that detects AI-generated text
using a smaller set of features and pre-trained models.
"""
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from textstat import textstat

# Create cache directory
data_dir = Path.home() / ".cache/ai_detector"
data_dir.mkdir(parents=True, exist_ok=True)
model_path = data_dir / "simple_detector.joblib"

class SimpleDetector:
    """A simplified detector for AI-generated text using surface-level features."""
    
    feature_names = [
        "avg_word_len", "TTR", "punct_density", "dup_word_ratio",
        "flesch", "smog", "lix"
    ]
    
    def __init__(self, clf=None):
        """Initialize with an optional classifier."""
        self.clf = clf
        self._word_pattern = re.compile(r"[\w'‑-]+", re.UNICODE)
    
    def extract_features(self, text):
        """Extract surface-level features from text."""
        # Handle empty text
        if not text or not text.strip():
            return np.zeros(len(self.feature_names))
        
        # Normalize line endings and whitespace
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'\n{3,}', '\n\n', text)  # Replace excessive newlines
        
        # Surface statistics
        words = self._word_pattern.findall(text)
        if not words or len(words) < 5:  # Require at least 5 words for meaningful analysis
            return np.zeros(len(self.feature_names))
            
        chars = sum(len(w) for w in words)
        vocab = len(set(w.lower() for w in words))
        punct = sum(1 for ch in text if ch in ",.;:!?")
        
        # Count repeated words, but ignore common repeats like "very very"
        redund = 0
        for i in range(len(words) - 1):
            if words[i].lower() == words[i + 1].lower() and len(words[i]) > 2:  # Ignore short words
                redund += 1
        
        # Readability metrics - handle potential errors with textstat
        try:
            # Make sure we have enough text for readability metrics
            if len(text.split()) < 30:  # Most readability metrics need ~30 words
                flesch, smog, lix = 50.0, 10.0, 30.0  # Neutral defaults
            else:
                flesch = textstat.flesch_reading_ease(text)
                smog = textstat.smog_index(text)
                lix = textstat.lix(text)
        except Exception:
            flesch, smog, lix = 50.0, 10.0, 30.0  # Neutral defaults
        
        features = [
            chars / max(len(words), 1),      # avg word length
            vocab / max(len(words), 1),      # type-token ratio
            punct / max(len(text), 1),       # punctuation density
            redund / max(len(words), 1),     # repeated word ratio
            flesch,
            smog,
            lix
        ]
        
        return np.array(features, dtype=np.float32)
    
    def predict_proba(self, text):
        """Predict probability that text is AI-generated."""
        if not self.clf:
            raise ValueError("Detector has no trained classifier")
        
        features = self.extract_features(text).reshape(1, -1)
        return float(self.clf.predict_proba(features)[0, 1])
    
    def label(self, text, threshold=0.5):
        """Return 'ai' or 'human' label based on probability."""
        return "ai" if self.predict_proba(text) >= threshold else "human"
    
    @classmethod
    def load(cls, path=None):
        """Load a pre-trained detector from disk."""
        path = Path(path or model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"No model found at {path}. Please train a model first."
            )
        
        clf = joblib.load(path)
        return cls(clf)
    
    def save(self, path=None):
        """Save the trained detector to disk."""
        if not self.clf:
            raise ValueError("Cannot save untrained detector")
            
        path = Path(path or model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.clf, path)
    
    @classmethod
    def train_default(cls, train_data=None):
        """Train a simple detector with default parameters."""
        # Use provided data or load example data
        if train_data is None:
            # Example data with more variety including multi-line examples
            ai_examples = [
                "The utilization of artificial intelligence in modern society presents numerous advantages and challenges that must be carefully considered.",
                "Upon analyzing the data, it becomes evident that there exists a correlation between the variables, suggesting a causal relationship.",
                "The implementation of sustainable practices in corporate environments can lead to enhanced efficiency and reduced environmental impact.",
                "In conclusion, the aforementioned evidence clearly demonstrates that the hypothesis is valid within the specified parameters.\n\nFurthermore, additional research in this domain would likely yield even more compelling results.",
                "The quantum mechanical properties of subatomic particles reveal fascinating insights into the nature of reality.\n\nThese properties include superposition, entanglement, and wave-particle duality."
            ]
            
            human_examples = [
                "I went to the store yesterday and bought some milk and eggs for breakfast.",
                "My friend told me about this great movie she saw last weekend. I can't wait to see it!",
                "The sunset was beautiful tonight. I sat on my porch and watched the colors change.",
                "I've been thinking about changing careers lately.\n\nIt's scary but exciting at the same time. My current job isn't fulfilling anymore.",
                "The concert was amazing! The band played all their hits and the crowd went wild.\n\nI lost my voice from singing along so much."
            ]
            
            texts = human_examples + ai_examples
            labels = [0] * len(human_examples) + [1] * len(ai_examples)
        else:
            texts, labels = train_data
        
        # Create detector and extract features
        detector = cls()
        features = np.stack([detector.extract_features(t) for t in texts])
        
        # Train logistic regression
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf.fit(features, labels)
        
        detector.clf = clf
        detector.save()
        return detector

def cli():
    """Command-line interface for the detector."""
    parser = argparse.ArgumentParser(
        description="Simplified AI text detector"
    )
    parser.add_argument(
        "--train", action="store_true", 
        help="Train a new model with example data"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output JSON instead of friendly text"
    )
    parser.add_argument(
        "--file", type=str,
        help="Analyze text from a file instead of stdin"
    )
    args = parser.parse_args()
    
    if args.train:
        print("Training new model with example data...")
        SimpleDetector.train_default()
        print(f"Model saved to {model_path}")
        return
    
    # Try to load the model
    try:
        detector = SimpleDetector.load()
    except FileNotFoundError:
        print("No trained model found. Training a new model...")
        detector = SimpleDetector.train_default()
    
    # Get input text
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    elif sys.stdin.isatty():
        print("Enter text (finish with TWO consecutive empty lines or Ctrl+D):")
        print("You can paste multiple paragraphs - just keep typing or pasting until done.")
        lines = []
        empty_line_count = 0
        try:
            while True:
                line = input()
                if line == "":
                    empty_line_count += 1
                    if empty_line_count >= 2:
                        # Two consecutive empty lines means we're done
                        break
                else:
                    empty_line_count = 0  # Reset counter when non-empty line is entered
                lines.append(line)
        except EOFError:
            # Handle Ctrl+D
            pass
        text = "\n".join(lines)
        
        # If we still have no text, provide a helpful message
        if not text.strip():
            print("No text provided. Please enter some text to analyze.")
            return
    else:
        # Reading from pipe or redirection
        text = sys.stdin.read()
    
    # Make prediction
    try:
        # Skip analysis for very short texts
        if len(text.strip()) < 10:
            print("Text is too short for reliable analysis. Please provide a longer sample.")
            return
            
        prob = detector.predict_proba(text)
        label = "ai" if prob >= 0.5 else "human"
        
        if args.json:
            print(json.dumps({"probability_ai": prob, "label": label}))
        else:
            pct = f"{prob*100:.1f}%"
            if prob > 0.85:
                verdict = "very likely AI"
            elif prob > 0.6:
                verdict = "likely AI"
            elif prob >= 0.4:
                verdict = "uncertain"
            elif prob > 0.15:
                verdict = "likely human"
            else:
                verdict = "very likely human"
                
            print(f"AI-likelihood: {pct} → {verdict}")
    except Exception as e:
        print(f"Error analyzing text: {e}")

if __name__ == "__main__":
    cli()
