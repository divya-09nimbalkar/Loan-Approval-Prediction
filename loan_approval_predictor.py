"""
Loan Approval Prediction
Professional ML Pipeline for Binary Classification

This module provides a complete loan approval prediction system including:
- Data preprocessing and feature engineering
- Multiple classification models
- Model evaluation and comparison
- Cross-validation and hyperparameter tuning
- Prediction interface for new loan applications
"""

import logging
import pickle
import warnings
from typing import Tuple, Dict, Any, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LoanApprovalPredictor:
    """Professional Loan Approval Prediction System"""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the predictor with specified model type.
        
        Args:
            model_type: 'random_forest', 'gradient_boost', or 'logistic_regression'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        logger.info(f"Initialized LoanApprovalPredictor with {model_type}")
    
    def generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic loan application dataset for demonstration.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with loan application features and approval status
        """
        logger.info(f"Generating synthetic dataset with {n_samples} samples")
        
        np.random.seed(42)
        data = {
            'age': np.random.randint(22, 65, n_samples),
            'income': np.random.normal(50000, 25000, n_samples).astype(int),
            'credit_score': np.random.normal(700, 100, n_samples).astype(int),
            'employment_years': np.random.randint(0, 40, n_samples),
            'loan_amount': np.random.normal(200000, 100000, n_samples).astype(int),
            'num_dependents': np.random.randint(0, 5, n_samples),
            'existing_debts': np.random.normal(50000, 30000, n_samples).astype(int),
            'employment_type': np.random.choice(['Salaried', 'Self-Employed', 'Unemployed'], n_samples),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
            'home_ownership': np.random.choice(['Owned', 'Rented', 'Mortgage'], n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable based on logical rules (for realistic labeling)
        df['loan_approved'] = (
            (df['credit_score'] > 650) &
            (df['income'] > 30000) &
            (df['loan_amount'] < df['income'] * 5) &
            (df['existing_debts'] < df['income'] * 2) &
            (df['age'] > 21) &
            (df['age'] < 65)
        ).astype(int)
        
        # Add some noise
        noise_indices = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
        df.loc[noise_indices, 'loan_approved'] = 1 - df.loc[noise_indices, 'loan_approved']
        
        logger.info(f"Approval rate: {df['loan_approved'].mean():.2%}")
        return df
    
    def preprocess_data(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Preprocess and encode features.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit encoders (True for training, False for prediction)
            
        Returns:
            Processed feature array
        """
        logger.info("Preprocessing data")
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_features = df_processed.select_dtypes(include=['object']).columns
        
        for feature in categorical_features:
            if fit:
                self.label_encoders[feature] = LabelEncoder()
                df_processed[feature] = self.label_encoders[feature].fit_transform(df_processed[feature])
            else:
                df_processed[feature] = self.label_encoders[feature].transform(df_processed[feature])
        
        # Store feature names
        if fit:
            self.feature_names = df_processed.columns.tolist()
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(df_processed)
        else:
            X_scaled = self.scaler.transform(df_processed)
        
        return X_scaled
    
    def build_model(self, hyperparameters: Dict[str, Any] = None):
        """
        Build the specified model with optional hyperparameter tuning.
        
        Args:
            hyperparameters: Custom hyperparameters for the model
        """
        logger.info(f"Building {self.model_type} model")
        
        if self.model_type == 'random_forest':
            params = hyperparameters or {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'random_state': 42,
                'n_jobs': -1
            }
            self.model = RandomForestClassifier(**params)
            
        elif self.model_type == 'gradient_boost':
            params = hyperparameters or {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42
            }
            self.model = GradientBoostingClassifier(**params)
            
        elif self.model_type == 'logistic_regression':
            params = hyperparameters or {
                'max_iter': 1000,
                'random_state': 42
            }
            self.model = LogisticRegression(**params)
        
        logger.info("Model built successfully")
    
    def train(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Dict[str, float]:
        """
        Train the model with cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
            
        Returns:
            Cross-validation scores
        """
        logger.info("Training model with cross-validation")
        self.build_model()
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=cv_folds, scoring='f1')
        logger.info(f"CV Scores: {cv_scores}")
        logger.info(f"Mean CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Train on full dataset
        self.model.fit(X, y)
        logger.info("Model training completed")
        
        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores
        }
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary of performance metrics
        """
        logger.info("Evaluating model")
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }
        
        logger.info("Classification Report:")
        logger.info("\n" + classification_report(y, y_pred))
        
        return metrics
    
    def predict(self, applicant_data: Dict[str, Any]) -> Tuple[int, float]:
        """
        Predict loan approval for a new applicant.
        
        Args:
            applicant_data: Dictionary with applicant features
            
        Returns:
            Tuple of (prediction: 0/1, probability: 0.0-1.0)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        logger.info(f"Making prediction for applicant")
        
        # Create DataFrame from input
        df_input = pd.DataFrame([applicant_data])
        
        # Preprocess
        X_processed = self.preprocess_data(df_input, fit=False)
        
        # Predict
        prediction = self.model.predict(X_processed)[0]
        probability = self.model.predict_proba(X_processed)[0][1]
        
        status = "APPROVED" if prediction == 1 else "REJECTED"
        logger.info(f"Prediction: {status} (Confidence: {probability:.2%})")
        
        return prediction, probability
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance if supported by model."""
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str = 'loan_approval_model.pkl'):
        """Save trained model and preprocessor."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str = 'loan_approval_model.pkl'):
        """Load trained model and preprocessor."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        predictor = cls(model_type=model_data['model_type'])
        predictor.model = model_data['model']
        predictor.scaler = model_data['scaler']
        predictor.label_encoders = model_data['label_encoders']
        predictor.feature_names = model_data['feature_names']
        
        logger.info(f"Model loaded from {filepath}")
        return predictor


def main():
    """Main execution pipeline."""
    
    logger.info("=" * 60)
    logger.info("LOAN APPROVAL PREDICTION SYSTEM")
    logger.info("=" * 60)
    
    # Initialize predictor
    predictor = LoanApprovalPredictor(model_type='random_forest')
    
    # Generate synthetic data
    df = predictor.generate_synthetic_data(n_samples=1500)
    
    # Split features and target
    X = df.drop('loan_approved', axis=1)
    y = df['loan_approved'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocess training data
    X_train_processed = predictor.preprocess_data(X_train, fit=True)
    X_test_processed = predictor.preprocess_data(X_test, fit=False)
    
    # Train model
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING PHASE")
    logger.info("=" * 60)
    predictor.train(X_train_processed, y_train, cv_folds=5)
    
    # Evaluate model
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION PHASE")
    logger.info("=" * 60)
    metrics = predictor.evaluate(X_test_processed, y_test)
    
    print("\nTest Set Performance Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name.upper()}: {metric_value:.4f}")
    
    # Feature importance
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE IMPORTANCE")
    logger.info("=" * 60)
    importance_df = predictor.get_feature_importance()
    if importance_df is not None:
        print(importance_df.to_string(index=False))
    
    # Test prediction
    logger.info("\n" + "=" * 60)
    logger.info("PREDICTION EXAMPLE")
    logger.info("=" * 60)
    
    test_applicant = {
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
    
    prediction, probability = predictor.predict(test_applicant)
    print(f"\nApplicant Profile:")
    for key, value in test_applicant.items():
        print(f"  {key}: {value}")
    print(f"\nLoan Decision: {'APPROVED ✓' if prediction == 1 else 'REJECTED ✗'}")
    print(f"Confidence Score: {probability:.2%}")
    
    # Save model
    predictor.save_model()
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
