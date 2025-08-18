"""
Kubeflow Pipeline Component for Model Training
This component handles the training phase of the ML pipeline in Kubeflow.
"""

from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics, Artifact
from typing import NamedTuple
import os

@component(
    base_image="python:3.9-slim",
    packages_to_install=[
        "scikit-learn==1.3.0",
        "pandas==2.0.3",
        "numpy==1.24.3",
        "mlflow==2.7.1",
        "joblib==1.3.2"
    ]
)
def train_iris_model(
    mlflow_tracking_uri: str,
    experiment_name: str,
    n_estimators: int = 100,
    max_depth: int = 5,
    random_state: int = 42,
    test_size: float = 0.2,
    model_output: Output[Model],
    metrics_output: Output[Metrics],
    dataset_output: Output[Dataset]
) -> NamedTuple('TrainingOutput', [
    ('run_id', str),
    ('accuracy', float),
    ('model_uri', str)
]):
    """
    Train an Iris classification model using Random Forest.
    
    Args:
        mlflow_tracking_uri: MLflow tracking server URI
        experiment_name: Name of the MLflow experiment
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees
        random_state: Random state for reproducibility
        test_size: Proportion of dataset for testing
        model_output: Output model artifact
        metrics_output: Output metrics artifact
        dataset_output: Output dataset artifact
        
    Returns:
        NamedTuple with run_id, accuracy, and model_uri
    """
    import pandas as pd
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
    import joblib
    import json
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Iris model training component...")
    
    # Set MLflow tracking
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Create or get experiment
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except Exception:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(experiment_name)
    
    # Load and prepare data
    logger.info("Loading Iris dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Save dataset info
    dataset_info = {
        'feature_names': iris.feature_names.tolist(),
        'target_names': iris.target_names.tolist(),
        'total_samples': int(X.shape[0]),
        'features': int(X.shape[1]),
        'train_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0])
    }
    
    with open(dataset_output.path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # Train model with MLflow tracking
    with mlflow.start_run(run_name="kubeflow_iris_training") as run:
        # Log parameters
        mlflow.log_param("component", "kubeflow_training")
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_size", test_size)
        
        # Create and train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        
        logger.info("Training model...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        overfitting = train_accuracy - test_accuracy
        
        # Log metrics to MLflow
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("overfitting", overfitting)
        
        # Log feature importance
        feature_importance = dict(zip(iris.feature_names, model.feature_importances_))
        for feature, importance in feature_importance.items():
            mlflow.log_metric(f"feature_importance_{feature.replace(' ', '_')}", importance)
        
        # Create model signature
        signature = infer_signature(X_train, y_pred_train)
        
        # Log model to MLflow
        model_uri = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="iris_random_forest_kubeflow"
        ).model_uri
        
        # Save model locally for Kubeflow
        joblib.dump(model, model_output.path)
        
        # Save metrics for Kubeflow
        metrics_data = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "overfitting": overfitting,
            "feature_importance": feature_importance,
            "model_params": {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "random_state": random_state
            }
        }
        
        with open(metrics_output.path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        # Log classification report
        report = classification_report(y_test, y_pred_test, target_names=iris.target_names)
        mlflow.log_text(report, "classification_report.txt")
        
        logger.info(f"Training completed!")
        logger.info(f"Train Accuracy: {train_accuracy:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        logger.info(f"Model URI: {model_uri}")
        
        # Return output
        from collections import namedtuple
        TrainingOutput = namedtuple('TrainingOutput', ['run_id', 'accuracy', 'model_uri'])
        return TrainingOutput(
            run_id=run.info.run_id,
            accuracy=test_accuracy,
            model_uri=model_uri
        )

@component(
    base_image="python:3.9-slim",
    packages_to_install=[
        "scikit-learn==1.3.0",
        "pandas==2.0.3",
        "numpy==1.24.3",
        "mlflow==2.7.1",
        "joblib==1.3.2"
    ]
)
def hyperparameter_tuning(
    mlflow_tracking_uri: str,
    experiment_name: str,
    param_grid: dict,
    cv_folds: int = 5,
    random_state: int = 42,
    best_model_output: Output[Model],
    tuning_results_output: Output[Metrics]
) -> NamedTuple('TuningOutput', [
    ('best_run_id', str),
    ('best_accuracy', float),
    ('best_params', dict)
]):
    """
    Perform hyperparameter tuning for the Iris model.
    
    Args:
        mlflow_tracking_uri: MLflow tracking server URI
        experiment_name: Name of the MLflow experiment
        param_grid: Dictionary of hyperparameters to tune
        cv_folds: Number of cross-validation folds
        random_state: Random state for reproducibility
        best_model_output: Output best model artifact
        tuning_results_output: Output tuning results
        
    Returns:
        NamedTuple with best_run_id, best_accuracy, and best_params
    """
    import pandas as pd
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import mlflow
    import mlflow.sklearn
    import joblib
    import json
    import logging
    from itertools import product
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting hyperparameter tuning component...")
    
    # Set MLflow tracking
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    # Load and prepare data
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # Create parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    best_accuracy = 0
    best_params = {}
    best_run_id = None
    best_model = None
    all_results = []
    
    logger.info(f"Testing {len(param_combinations)} parameter combinations...")
    
    for i, param_combo in enumerate(param_combinations):
        params = dict(zip(param_names, param_combo))
        
        with mlflow.start_run(run_name=f"hypertuning_combo_{i+1}") as run:
            # Log parameters
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            mlflow.log_param("component", "hyperparameter_tuning")
            mlflow.log_param("cv_folds", cv_folds)
            
            # Create and train model
            model = RandomForestClassifier(random_state=random_state, **params)
            
            # Perform cross-validation
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Train on full training set and evaluate on test set
            model.fit(X_train, y_train)
            test_accuracy = accuracy_score(y_test, model.predict(X_test))
            
            # Log metrics
            mlflow.log_metric("cv_accuracy_mean", cv_mean)
            mlflow.log_metric("cv_accuracy_std", cv_std)
            mlflow.log_metric("test_accuracy", test_accuracy)
            
            # Track results
            result = {
                'run_id': run.info.run_id,
                'params': params,
                'cv_accuracy_mean': cv_mean,
                'cv_accuracy_std': cv_std,
                'test_accuracy': test_accuracy
            }
            all_results.append(result)
            
            # Update best model if this is better
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_params = params
                best_run_id = run.info.run_id
                best_model = model
            
            logger.info(f"Combo {i+1}/{len(param_combinations)}: "
                       f"CV={cv_mean:.4f}±{cv_std:.4f}, Test={test_accuracy:.4f}")
    
    # Save best model
    if best_model is not None:
        joblib.dump(best_model, best_model_output.path)
    
    # Save tuning results
    tuning_results = {
        'best_params': best_params,
        'best_accuracy': best_accuracy,
        'best_run_id': best_run_id,
        'total_combinations': len(param_combinations),
        'all_results': all_results
    }
    
    with open(tuning_results_output.path, 'w') as f:
        json.dump(tuning_results, f, indent=2, default=str)
    
    logger.info(f"Hyperparameter tuning completed!")
    logger.info(f"Best accuracy: {best_accuracy:.4f}")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best run ID: {best_run_id}")
    
    # Return output
    from collections import namedtuple
    TuningOutput = namedtuple('TuningOutput', ['best_run_id', 'best_accuracy', 'best_params'])
    return TuningOutput(
        best_run_id=best_run_id,
        best_accuracy=best_accuracy,
        best_params=best_params
    )

# Example component factory function
def create_training_pipeline_components():
    """
    Factory function to create training pipeline components with default configurations.
    
    Returns:
        Dict of configured components
    """
    return {
        'train_model': train_iris_model,
        'hyperparameter_tuning': hyperparameter_tuning
    }


