"""
Model Training and Evaluation Module

Trains multiple classification models for support ticket categorization and priority prediction.
Includes model evaluation, hyperparameter tuning, and performance metrics.
"""

import pickle
import json
from typing import Dict, List, Tuple, Any
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import warnings

warnings.filterwarnings('ignore')


class ModelTrainer:
    """Train and evaluate classification models."""

    def __init__(self, random_state: int = 42):
        """Initialize trainer."""
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None

    def create_models(self) -> Dict[str, Any]:
        """
        Create multiple classification models.

        Returns:
            Dictionary of model name to model instance
        """
        models = {
            "Naive Bayes": MultinomialNB(alpha=1.0),
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                solver='lbfgs'
            ),
            "Linear SVM": LinearSVC(
                max_iter=2000,
                random_state=self.random_state,
                dual=False
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=self.random_state,
                n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state
            )
        }
        self.models = models
        return models

    def train_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train all models.

        Args:
            X_train: Training features
            y_train: Training labels
            verbose: Print training progress

        Returns:
            Dictionary of trained models
        """
        if not self.models:
            self.create_models()

        trained_models = {}

        for name, model in self.models.items():
            if verbose:
                print(f"Training {name}...")

            model.fit(X_train, y_train)
            trained_models[name] = model

            if verbose:
                print(f"  ✓ {name} trained")

        self.models = trained_models
        return trained_models

    def evaluate_models(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        label_mapping: Dict[int, str] = None,
        verbose: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models.

        Args:
            X_test: Test features
            y_test: Test labels
            label_mapping: Mapping of label indices to names
            verbose: Print evaluation progress

        Returns:
            Dictionary of model evaluation results
        """
        results = {}

        for name, model in self.models.items():
            if verbose:
                print(f"Evaluating {name}...")

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            # Get confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            # Get classification report
            class_report = classification_report(
                y_test, y_pred,
                target_names=list(label_mapping.values()) if label_mapping else None,
                output_dict=True,
                zero_division=0
            )

            results[name] = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "confusion_matrix": cm.tolist(),
                "classification_report": class_report,
                "predictions": y_pred.tolist()
            }

            if verbose:
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  F1-Score: {f1:.4f}")

        self.results = results

        # Find best model
        best_model_name = max(results, key=lambda x: results[x]['f1_score'])
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]

        if verbose:
            print(f"\n✓ Best Model: {best_model_name} (F1: {results[best_model_name]['f1_score']:.4f})")

        return results

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Perform cross-validation on all models.

        Args:
            X: Features
            y: Labels
            cv: Number of folds
            verbose: Print progress

        Returns:
            Dictionary of cross-validation scores
        """
        cv_results = {}

        for name, model in self.models.items():
            if verbose:
                print(f"Cross-validating {name}...")

            scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
            cv_results[name] = {
                "scores": scores.tolist(),
                "mean": float(scores.mean()),
                "std": float(scores.std())
            }

            if verbose:
                print(f"  Mean F1: {scores.mean():.4f} (+/- {scores.std():.4f})")

        return cv_results

    def get_feature_importance(self, model_name: str = None, top_n: int = 20) -> Dict[str, float]:
        """
        Get feature importance from tree-based models.

        Args:
            model_name: Name of model (uses best model if None)
            top_n: Number of top features

        Returns:
            Dictionary of feature names and importance scores
        """
        if model_name is None:
            model_name = self.best_model_name

        model = self.models.get(model_name)

        if model is None:
            raise ValueError(f"Model {model_name} not found")

        if not hasattr(model, 'feature_importances_'):
            raise ValueError(f"Model {model_name} does not support feature importance")

        importances = model.feature_importances_
        top_indices = np.argsort(importances)[-top_n:][::-1]

        return {
            f"feature_{i}": float(importances[i])
            for i in top_indices
        }

    def save_model(self, model_name: str, filepath: str) -> None:
        """Save trained model to file."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        with open(filepath, 'wb') as f:
            pickle.dump(self.models[model_name], f)

    def load_model(self, model_name: str, filepath: str) -> None:
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            self.models[model_name] = pickle.load(f)

    def predict(self, X: np.ndarray, model_name: str = None) -> np.ndarray:
        """
        Make predictions using specified or best model.

        Args:
            X: Features
            model_name: Model to use (uses best model if None)

        Returns:
            Predictions
        """
        if model_name is None:
            model = self.best_model
        else:
            model = self.models.get(model_name)

        if model is None:
            raise ValueError("No model available for prediction")

        return model.predict(X)

    def predict_proba(self, X: np.ndarray, model_name: str = None) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: Features
            model_name: Model to use

        Returns:
            Prediction probabilities
        """
        if model_name is None:
            model = self.best_model
        else:
            model = self.models.get(model_name)

        if not hasattr(model, 'predict_proba'):
            raise ValueError(f"Model does not support probability predictions")

        return model.predict_proba(X)

    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluation results."""
        if not self.results:
            raise ValueError("No results available. Run evaluate_models first.")

        summary = {
            "timestamp": datetime.now().isoformat(),
            "best_model": self.best_model_name,
            "best_f1_score": self.results[self.best_model_name]['f1_score'],
            "all_models": {}
        }

        for model_name, metrics in self.results.items():
            summary["all_models"][model_name] = {
                "accuracy": metrics['accuracy'],
                "precision": metrics['precision'],
                "recall": metrics['recall'],
                "f1_score": metrics['f1_score']
            }

        return summary


class HyperparameterTuner:
    """Tune hyperparameters for models."""

    @staticmethod
    def tune_logistic_regression(
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 5
    ) -> Dict[str, Any]:
        """Tune Logistic Regression hyperparameters."""
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [1000, 2000]
        }

        model = LogisticRegression(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        return {
            "best_params": grid_search.best_params_,
            "best_score": float(grid_search.best_score_),
            "model": grid_search.best_estimator_
        }

    @staticmethod
    def tune_random_forest(
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 5
    ) -> Dict[str, Any]:
        """Tune Random Forest hyperparameters."""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5, 10]
        }

        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        return {
            "best_params": grid_search.best_params_,
            "best_score": float(grid_search.best_score_),
            "model": grid_search.best_estimator_
        }


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification

    # Create sample data
    X, y = make_classification(n_samples=200, n_features=100, n_informative=50, n_classes=4, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    trainer = ModelTrainer()
    trainer.create_models()
    trainer.train_models(X_train, y_train)

    # Evaluate models
    label_mapping = {0: "Class 0", 1: "Class 1", 2: "Class 2", 3: "Class 3"}
    results = trainer.evaluate_models(X_test, y_test, label_mapping)

    # Print results
    print("\nModel Evaluation Results:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
