"""
MLflow-enabled Training Script for Iris Classification
This script trains a Random Forest model on the Iris dataset with MLflow tracking.
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IrisModelTrainer:
    def __init__(self, mlflow_tracking_uri="http://127.0.0.1:5000", experiment_name="iris_classification"):
        """
        Initialize the model trainer with MLflow configuration.
        
        Args:
            mlflow_tracking_uri (str): MLflow tracking server URI
            experiment_name (str): Name of the MLflow experiment
        """
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.experiment_name = experiment_name
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        # Create or get experiment
        try:
            experiment_id = mlflow.create_experiment(self.experiment_name)
        except Exception as e:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"Using MLflow experiment: {self.experiment_name} (ID: {experiment_id})")

    def load_and_prepare_data(self, test_size=0.2, random_state=42):
        """
        Load and prepare the Iris dataset.
        
        Args:
            test_size (float): Proportion of dataset to include in test split
            random_state (int): Random state for reproducibility
        """
        logger.info("Loading Iris dataset...")
        
        # Load the iris dataset
        iris = load_iris()
        X = iris.data
        y = iris.target
        
        # Create DataFrame for better handling
        feature_names = iris.feature_names
        target_names = iris.target_names
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Training set: {self.X_train.shape[0]} samples")
        logger.info(f"Test set: {self.X_test.shape[0]} samples")
        logger.info(f"Classes: {target_names}")
        
        return {
            'feature_names': feature_names,
            'target_names': target_names,
            'dataset_info': {
                'total_samples': X.shape[0],
                'features': X.shape[1],
                'classes': len(target_names)
            }
        }

    def train_model(self, n_estimators=100, max_depth=None, random_state=42):
        """
        Train the Random Forest model with MLflow tracking.
        
        Args:
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of the trees
            random_state (int): Random state for reproducibility
        """
        logger.info("Starting model training...")
        
        with mlflow.start_run(run_name="iris_rf_training") as run:
            # Log parameters
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth if max_depth else "None")
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("test_size", 0.2)
            
            # Create and train the model
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
            
            self.model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred_train = self.model.predict(self.X_train)
            y_pred_test = self.model.predict(self.X_test)
            
            # Calculate metrics
            train_accuracy = accuracy_score(self.y_train, y_pred_train)
            test_accuracy = accuracy_score(self.y_test, y_pred_test)
            
            # Log metrics
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("overfitting", train_accuracy - test_accuracy)
            
            # Log feature importance
            feature_importance = dict(zip(
                ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
                self.model.feature_importances_
            ))
            
            for feature, importance in feature_importance.items():
                mlflow.log_metric(f"feature_importance_{feature}", importance)
            
            # Create and log model signature
            signature = infer_signature(self.X_train, y_pred_train)
            
            # Log the model
            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="model",
                signature=signature,
                registered_model_name="iris_random_forest"
            )
            
            # Log classification report as text artifact
            report = classification_report(self.y_test, y_pred_test, 
                                         target_names=['setosa', 'versicolor', 'virginica'])
            
            with open("classification_report.txt", "w") as f:
                f.write(report)
            mlflow.log_artifact("classification_report.txt")
            
            # Clean up temporary file
            os.remove("classification_report.txt")
            
            logger.info(f"Model training completed!")
            logger.info(f"Train Accuracy: {train_accuracy:.4f}")
            logger.info(f"Test Accuracy: {test_accuracy:.4f}")
            logger.info(f"MLflow Run ID: {run.info.run_id}")
            
            return {
                'run_id': run.info.run_id,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'feature_importance': feature_importance
            }

    def save_model_locally(self, model_path="models/iris_rf_model.joblib"):
        """
        Save the trained model locally using joblib.
        
        Args:
            model_path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved locally at: {model_path}")
        
        return model_path

def main():
    """
    Main function to run the training pipeline.
    """
    # Initialize trainer
    trainer = IrisModelTrainer()
    
    # Load and prepare data
    data_info = trainer.load_and_prepare_data()
    
    # Train model with different hyperparameters
    hyperparameters = [
        {'n_estimators': 50, 'max_depth': 3},
        {'n_estimators': 100, 'max_depth': 5},
        {'n_estimators': 200, 'max_depth': None}
    ]
    
    best_accuracy = 0
    best_run_id = None
    
    for params in hyperparameters:
        logger.info(f"Training with parameters: {params}")
        result = trainer.train_model(**params)
        
        if result['test_accuracy'] > best_accuracy:
            best_accuracy = result['test_accuracy']
            best_run_id = result['run_id']
    
    logger.info(f"Best model achieved test accuracy: {best_accuracy:.4f}")
    logger.info(f"Best run ID: {best_run_id}")
    
    # Save the final model locally
    trainer.save_model_locally()

if __name__ == "__main__":
    main()


