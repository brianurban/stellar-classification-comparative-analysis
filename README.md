# Stellar Classification Comparative Analysis

A comparative study evaluating how different machine learning model classes
respond to measurement noise when classifying stars into six astrophysical
categories. The project uses a multilayer perceptron (MLP) implemented in
PyTorch alongside logistic regression and two decision tree variants,
evaluated across six Gaussian noise levels using 5-fold stratified
cross-validation.

## Key Findings

- **Peak clean-data accuracy is misleading for model selection.** The
  unconstrained decision tree achieved 99.6% accuracy on clean data but
  collapsed to 60.4% under maximum noise — a 39-point degradation.
- The regularized MLP (dropout=0.3) held a narrow advantage at moderate
  noise levels most relevant to real astronomical observations.
- Logistic regression edged ahead under extreme perturbation, suggesting
  simpler linear boundaries are more resilient to heavy noise.
- A dropout ablation confirmed that lighter regularization (0.1–0.3)
  outperforms aggressive dropout (0.5) on small datasets.

## Models Compared

| Model | Library |
|---|---|
| Multilayer Perceptron (64→32→6, ReLU, BatchNorm, Dropout=0.3) | PyTorch |
| Logistic Regression | scikit-learn |
| Decision Tree (unconstrained) | scikit-learn |
| Decision Tree (max_depth=4) | scikit-learn |

## Dataset

240 samples from [Kaggle](https://www.kaggle.com/datasets/deepu1109/star-dataset)
with 6 features (4 numeric, 2 categorical) and 6 balanced target classes
(Brown Dwarf, Red Dwarf, White Dwarf, Main Sequence, Supergiant, Hypergiant).

## Methodology

- **Primary evaluation:** 5-fold stratified cross-validation
- **Noise injection:** Gaussian perturbation on numeric features, scaled
  proportionally to each feature's standard deviation
- **Two experiments:**
  1. Noise robustness sweep (6 levels × 4 models)
  2. Dropout ablation (4 rates × 1 noise level)
- **Diagnostics:** Representative 70-15-15 split for training curves,
  classification report, and confusion matrix

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

## Getting Started

```bash
# Clone the repository
git clone https://github.com/brianurban/stellar-classification-noise-robustness.git

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook notebooks/stellar_classification_comparative_analysis.ipynb
