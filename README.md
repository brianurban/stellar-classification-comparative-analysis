# Stellar Classification Comparative Analysis

A comparative study evaluating how different machine learning model classes respond to measurement noise when classifying stars into six astrophysical categories. The project uses a multilayer perceptron (MLP) implemented in PyTorch alongside logistic regression, two decision tree variants, and two ensemble methods (Random Forest and Gradient Boosting), evaluated across six Gaussian noise levels using 5-fold stratified cross-validation.

## Key Findings

- **Peak clean-data accuracy is misleading for model selection.** Random Forest achieved perfect accuracy (100%) on clean data, while the unconstrained decision tree reached 99.6%—but both collapsed under noise, with the decision tree dropping to 60.4% at maximum noise (a 39-point degradation).
- **Three distinct performance regimes emerged:** tree-based methods dominated clean data (Random Forest: 100%), the MLP and Random Forest co-led at moderate noise (91.7% at noise=0.25), and logistic regression prevailed under extreme perturbation (75.4% at maximum noise).
- The unconstrained decision tree's 39-point collapse (99.6% to 60.4%) confirmed that threshold-based models are particularly fragile under measurement uncertainty, while ensemble averaging partially mitigated this sensitivity.
- A dropout ablation revealed that the highest dropout rate (0.5) achieved marginally better accuracy (93.3%) than lighter regularization (91.7%), contrary to the original hypothesis—an artifact of checkpoint methodology resolved through principled final-epoch evaluation.
- No pairwise differences reached statistical significance (p ≥ 0.0625), reflecting limited power with 5-fold cross-validation.

## Models Compared

| Model | Library |
|---|---|
| Multilayer Perceptron (64→32→6, ReLU, BatchNorm, Dropout=0.3) | PyTorch |
| Logistic Regression | scikit-learn |
| Decision Tree (unconstrained) | scikit-learn |
| Decision Tree (max_depth=4) | scikit-learn |
| Random Forest | scikit-learn |
| Gradient Boosting | scikit-learn |

## Dataset

240 samples from [Kaggle](https://www.kaggle.com/datasets/deepu1109/star-dataset) with 6 features (4 numeric, 2 categorical) and 6 balanced target classes (Brown Dwarf, Red Dwarf, White Dwarf, Main Sequence, Supergiant, Hypergiant).

## Methodology

- **Primary evaluation:** 5-fold stratified cross-validation
- **Noise injection:** Gaussian perturbation on numeric features, scaled proportionally to each feature's standard deviation
- **Two experiments:**
  1. Noise robustness sweep (6 levels × 6 models)
  2. Dropout ablation (4 rates × 1 noise level)
- **Diagnostics:** Representative 70-15-15 split for training curves, classification report, and confusion matrix

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=fff)
![NumPy](https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff)
![Scikit-learn](https://custom-icon-badges.demolab.com/badge/-scikit--learn-%23F7931E?logo=scikit-learn&logoColor=white)
![Matplotlib](https://custom-icon-badges.demolab.com/badge/Matplotlib-71D291?logo=matplotlib&logoColor=fff)
![Seaborn](https://img.shields.io/badge/Seaborn-4EAEAA?logo=python&logoColor=fff)
![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)
![SciPy](https://custom-icon-badges.demolab.com/badge/SciPy-54A6FF?logo=scipy&logoColor=fff)

## Getting Started

```bash
# Clone the repository
git clone https://github.com/brianurban/stellar-classification-comparative-analysis.git

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook stellar_classification_comparative_analysis.ipynb
