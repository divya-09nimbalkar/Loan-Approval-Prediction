# Loan Approval Predictor

A professional-grade machine learning system for predicting loan approval decisions using advanced classification models and comprehensive data preprocessing pipelines.

## Overview

This project implements a complete ML pipeline for binary loan approval classification, demonstrating industry-standard practices including:
- **Data Preprocessing**: Feature scaling, encoding, and normalization
- **Model Training**: Multiple algorithms (Random Forest, Gradient Boosting, Logistic Regression)
- **Cross-Validation**: 5-fold cross-validation with stratified splits
- **Evaluation**: Comprehensive metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- **Feature Engineering**: Synthetic data generation with realistic relationships
- **Model Persistence**: Save/load trained models with preprocessor state
- **Production-Ready**: Logging, error handling, and type hints

## Features

✓ **Multiple Classification Models**
- Random Forest (default)
- Gradient Boosting Classifier
- Logistic Regression

✓ **Comprehensive ML Pipeline**
- Data generation and preprocessing
- Feature encoding and scaling
- Train-test split with stratification
- Cross-validation with multiple folds

✓ **Advanced Evaluation**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix
- Feature Importance Analysis
- Classification Reports

✓ **Production Features**
- Model serialization (pickle)
- Structured logging
- Type hints for better code quality
- Error handling
- Modular class-based design

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Run the Complete Pipeline

```bash
python loan_approval_predictor.py
```

This will:
1. Generate synthetic loan data (1500 samples)
2. Preprocess and split into train/test sets
3. Train Random Forest model with 5-fold cross-validation
4. Evaluate on test set with comprehensive metrics
5. Display feature importance rankings
6. Make a sample prediction
7. Save the trained model

### Expected Output

```
============================================================
LOAN APPROVAL PREDICTION SYSTEM
============================================================
...
Test Set Performance Metrics:
  ACCURACY: 0.8500
  PRECISION: 0.7937
  RECALL: 0.8403
  F1: 0.8163
  ROC_AUC: 0.8653
...
Feature Importance:
  income: 0.3098
  credit_score: 0.3013
  loan_amount: 0.1516
...
```

## Usage Examples

### As a Library

```python
from loan_approval_predictor import LoanApprovalPredictor

# Initialize predictor
predictor = LoanApprovalPredictor(model_type='random_forest')

# Generate data and train
df = predictor.generate_synthetic_data(n_samples=1000)
X = df.drop('loan_approved', axis=1)
y = df['loan_approved'].values

X_processed = predictor.preprocess_data(X, fit=True)
predictor.train(X_processed, y)

# Make predictions
applicant = {
    'age': 35,
    'income': 75000,
    'credit_score': 720,
    'employment_years': 8,
    'loan_amount': 250000,
    'num_dependents': 2,
    'existing_debts': 50000,
    'employment_type': 'Salaried',
    'education': 'Bachelor',
    'home_ownership': 'Mortgage'
}

prediction, probability = predictor.predict(applicant)
print(f"Loan Decision: {'APPROVED' if prediction == 1 else 'REJECTED'}")
print(f"Confidence: {probability:.2%}")
```

### Save and Load Models

```python
# Save trained model
predictor.save_model('loan_model.pkl')

# Load for later use
predictor = LoanApprovalPredictor.load_model('loan_model.pkl')
prediction, prob = predictor.predict(applicant_data)
```

## Architecture

### Core Classes

**LoanApprovalPredictor**
- Main orchestrator for the ML pipeline
- Handles model training, evaluation, and inference
- Manages data preprocessing and feature encoding

### Key Methods

| Method | Purpose |
|--------|---------|
| `generate_synthetic_data()` | Create realistic loan dataset |
| `preprocess_data()` | Scale and encode features |
| `build_model()` | Initialize ML model |
| `train()` | Train with cross-validation |
| `evaluate()` | Compute performance metrics |
| `predict()` | Make predictions on new applicants |
| `get_feature_importance()` | Analyze feature contributions |
| `save_model()` | Serialize trained model |
| `load_model()` | Deserialize saved model |

## Dataset Features

The synthetic dataset includes:
- **Demographic**: Age, Number of Dependents
- **Financial**: Income, Existing Debts, Loan Amount
- **Credit**: Credit Score
- **Employment**: Employment Type, Years of Experience, Education
- **Housing**: Home Ownership Status

## Performance Metrics

The current model achieves:
- **Accuracy**: 85%
- **Precision**: 79.4%
- **Recall**: 84.0%
- **F1-Score**: 81.6%
- **ROC-AUC**: 86.5%

## Model Comparison

Run with different model types:

```python
# Gradient Boosting
predictor = LoanApprovalPredictor(model_type='gradient_boost')

# Logistic Regression
predictor = LoanApprovalPredictor(model_type='logistic_regression')
```

## Requirements

- Python 3.7+
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Scikit-learn >= 1.0.0

## Project Structure

```
loan_approval_predictor/
├── loan_approval_predictor.py    # Main ML pipeline
├── loan_approval_model.pkl       # Saved model (generated)
├── requirements.txt              # Dependencies
└── README.md                      # Documentation
```

## Key Design Patterns

✓ **Object-Oriented Design**: Class-based architecture for maintainability
✓ **Separation of Concerns**: Data, model, and evaluation logic separated
✓ **Type Hints**: Full type annotations for IDE support
✓ **Logging**: Structured logging for debugging and monitoring
✓ **Cross-Validation**: Prevents overfitting through multiple folds
✓ **Stratified Splits**: Maintains class distribution in subsets

## Future Enhancements

- [ ] Feature selection using information gain
- [ ] Hyperparameter grid search optimization
- [ ] ROC curve visualization
- [ ] Confusion matrix heatmap
- [ ] Feature interaction analysis
- [ ] Model explainability (SHAP values)
- [ ] API endpoint (Flask/FastAPI)
- [ ] Real dataset integration

## Author

Divya Nimbalkar

## License

MIT License
