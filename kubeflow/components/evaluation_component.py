"""
Kubeflow Pipeline Component for Model Evaluation
This component handles the evaluation phase of the ML pipeline in Kubeflow.
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
        "joblib==1.3.2",
        "matplotlib==3.7.2",
        "seaborn==0.12.2"
    ]
)
def evaluate_model(
    model_input: Input[Model],
    mlflow_tracking_uri: str,
    experiment_name: str,
    model_run_id: str = "",
    evaluation_metrics: Output[Metrics],
    evaluation_report: Output[Artifact]
) -> NamedTuple('EvaluationOutput', [
    ('accuracy', float),
    ('precision', float),
    ('recall', float),
    ('f1_score', float)
]):
    """
    Evaluate a trained model on test data.
    
    Args:
        model_input: Input trained model
        mlflow_tracking_uri: MLflow tracking server URI
        experiment_name: Name of the MLflow experiment
        model_run_id: Run ID of the model being evaluated
        evaluation_metrics: Output evaluation metrics
        evaluation_report: Output evaluation report
        
    Returns:
        NamedTuple with accuracy, precision, recall, and f1_score
    """
    import pandas as pd
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix, roc_auc_score
    )
    from sklearn.preprocessing import label_binarize
    import mlflow
    import joblib
    import json
    import logging
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting model evaluation component...")
    
    # Set MLflow tracking
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    # Load the model
    logger.info(f"Loading model from {model_input.path}")
    model = joblib.load(model_input.path)
    
    # Load test data (same split as training for consistency)
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    with mlflow.start_run(run_name="kubeflow_model_evaluation") as run:
        # Log component info
        mlflow.log_param("component", "kubeflow_evaluation")
        if model_run_id:
            mlflow.log_param("evaluated_model_run_id", model_run_id)
        
        # Make predictions
        logger.info("Making predictions...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Calculate AUC for multiclass
        y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
        auc_score = roc_auc_score(y_test_binarized, y_proba, multi_class='ovr', average='weighted')
        
        # Log metrics to MLflow
        mlflow.log_metric("evaluation_accuracy", accuracy)
        mlflow.log_metric("evaluation_precision", precision)
        mlflow.log_metric("evaluation_recall", recall)
        mlflow.log_metric("evaluation_f1", f1)
        mlflow.log_metric("evaluation_auc", auc_score)
        
        # Calculate per-class metrics
        precision_per_class = precision_score(y_test, y_pred, average=None)
        recall_per_class = recall_score(y_test, y_pred, average=None)
        f1_per_class = f1_score(y_test, y_pred, average=None)
        
        class_names = iris.target_names
        class_metrics = {}
        for i, class_name in enumerate(class_names):
            class_metrics[f"precision_{class_name}"] = precision_per_class[i]
            class_metrics[f"recall_{class_name}"] = recall_per_class[i]
            class_metrics[f"f1_{class_name}"] = f1_per_class[i]
            
            # Log to MLflow
            mlflow.log_metric(f"precision_{class_name}", precision_per_class[i])
            mlflow.log_metric(f"recall_{class_name}", recall_per_class[i])
            mlflow.log_metric(f"f1_{class_name}", f1_per_class[i])
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Model Evaluation')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        # Save confusion matrix plot
        confusion_matrix_path = '/tmp/confusion_matrix.png'
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log confusion matrix to MLflow
        mlflow.log_artifact(confusion_matrix_path, "evaluation_plots")
        
        # Generate classification report
        report = classification_report(y_test, y_pred, target_names=class_names)
        
        # Create detailed evaluation report
        evaluation_summary = {
            "model_evaluation_summary": {
                "test_samples": int(len(y_test)),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "auc_score": float(auc_score)
            },
            "per_class_metrics": class_metrics,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "predictions_sample": {
                "actual": y_test[:10].tolist(),
                "predicted": y_pred[:10].tolist(),
                "probabilities": y_proba[:10].tolist()
            }
        }
        
        # Save metrics for Kubeflow
        with open(evaluation_metrics.path, 'w') as f:
            json.dump(evaluation_summary, f, indent=2)
        
        # Create detailed report text
        report_text = f"""
MODEL EVALUATION REPORT
{'='*50}

Model Performance Summary:
- Test Samples: {len(y_test)}
- Accuracy: {accuracy:.4f}
- Precision: {precision:.4f}
- Recall: {recall:.4f}
- F1-Score: {f1:.4f}
- AUC Score: {auc_score:.4f}

Per-Class Performance:
{'-'*30}
"""
        
        for i, class_name in enumerate(class_names):
            report_text += f"""
{class_name.capitalize()}:
  - Precision: {precision_per_class[i]:.4f}
  - Recall: {recall_per_class[i]:.4f}
  - F1-Score: {f1_per_class[i]:.4f}
"""
        
        report_text += f"""

Classification Report:
{'-'*30}
{report}

Confusion Matrix:
{'-'*30}
{cm}
"""
        
        # Save report to artifact
        with open(evaluation_report.path, 'w') as f:
            f.write(report_text)
        
        # Log report to MLflow
        mlflow.log_text(report_text, "evaluation_report.txt")
        
        logger.info("Model evaluation completed!")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        
        # Return output
        from collections import namedtuple
        EvaluationOutput = namedtuple('EvaluationOutput', ['accuracy', 'precision', 'recall', 'f1_score'])
        return EvaluationOutput(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1
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
def model_validation(
    model_input: Input[Model],
    accuracy_threshold: float = 0.8,
    precision_threshold: float = 0.8,
    validation_result: Output[Artifact]
) -> NamedTuple('ValidationOutput', [
    ('is_valid', bool),
    ('validation_score', float),
    ('recommendations', str)
]):
    """
    Validate model performance against predefined thresholds.
    
    Args:
        model_input: Input trained model
        accuracy_threshold: Minimum required accuracy
        precision_threshold: Minimum required precision
        validation_result: Output validation result
        
    Returns:
        NamedTuple with validation status, score, and recommendations
    """
    import pandas as pd
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import joblib
    import json
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting model validation component...")
    
    # Load the model
    model = joblib.load(model_input.path)
    
    # Load test data
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Validation checks
    validation_checks = {
        'accuracy_check': accuracy >= accuracy_threshold,
        'precision_check': precision >= precision_threshold,
        'recall_check': recall >= 0.7,  # Additional check
        'f1_check': f1 >= 0.75,  # Additional check
        'no_class_bias': min(precision_score(y_test, y_pred, average=None)) >= 0.6  # Check for class bias
    }
    
    # Calculate validation score
    passed_checks = sum(validation_checks.values())
    total_checks = len(validation_checks)
    validation_score = passed_checks / total_checks
    
    # Overall validation status
    is_valid = validation_score >= 0.8  # At least 80% of checks must pass
    
    # Generate recommendations
    recommendations = []
    if not validation_checks['accuracy_check']:
        recommendations.append(f"Accuracy ({accuracy:.3f}) below threshold ({accuracy_threshold}). Consider more training data or feature engineering.")
    if not validation_checks['precision_check']:
        recommendations.append(f"Precision ({precision:.3f}) below threshold ({precision_threshold}). Review false positive rate.")
    if not validation_checks['recall_check']:
        recommendations.append(f"Recall ({recall:.3f}) below 0.7. Model may be missing important cases.")
    if not validation_checks['f1_check']:
        recommendations.append(f"F1-score ({f1:.3f}) below 0.75. Balance between precision and recall needs improvement.")
    if not validation_checks['no_class_bias']:
        recommendations.append("Class bias detected. Some classes have very low precision. Consider class balancing techniques.")
    
    if is_valid:
        recommendations.append("Model passes validation criteria and is ready for deployment.")
    
    recommendations_text = "; ".join(recommendations) if recommendations else "No specific recommendations."
    
    # Create validation report
    validation_report = {
        'validation_summary': {
            'is_valid': is_valid,
            'validation_score': validation_score,
            'passed_checks': passed_checks,
            'total_checks': total_checks
        },
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        },
        'thresholds': {
            'accuracy_threshold': accuracy_threshold,
            'precision_threshold': precision_threshold,
            'recall_threshold': 0.7,
            'f1_threshold': 0.75
        },
        'validation_checks': validation_checks,
        'recommendations': recommendations
    }
    
    # Save validation result
    with open(validation_result.path, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    logger.info(f"Model validation completed!")
    logger.info(f"Validation status: {'PASSED' if is_valid else 'FAILED'}")
    logger.info(f"Validation score: {validation_score:.2f}")
    logger.info(f"Accuracy: {accuracy:.4f} (threshold: {accuracy_threshold})")
    logger.info(f"Precision: {precision:.4f} (threshold: {precision_threshold})")
    
    # Return output
    from collections import namedtuple
    ValidationOutput = namedtuple('ValidationOutput', ['is_valid', 'validation_score', 'recommendations'])
    return ValidationOutput(
        is_valid=is_valid,
        validation_score=validation_score,
        recommendations=recommendations_text
    )

@component(
    base_image="python:3.9-slim",
    packages_to_install=[
        "mlflow==2.7.1"
    ]
)
def register_model_if_valid(
    mlflow_tracking_uri: str,
    model_run_id: str,
    model_name: str,
    is_model_valid: bool,
    validation_score: float,
    model_stage: str = "Staging",
    registration_result: Output[Artifact]
) -> NamedTuple('RegistrationOutput', [
    ('registered', bool),
    ('model_version', str),
    ('model_stage', str)
]):
    """
    Register model in MLflow registry if validation passes.
    
    Args:
        mlflow_tracking_uri: MLflow tracking server URI
        model_run_id: Run ID of the model to register
        model_name: Name for the registered model
        is_model_valid: Whether the model passed validation
        validation_score: Validation score from validation component
        model_stage: Stage to assign to the model
        registration_result: Output registration result
        
    Returns:
        NamedTuple with registration status, version, and stage
    """
    import mlflow
    from mlflow.tracking import MlflowClient
    import json
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting model registration component...")
    
    # Set MLflow tracking
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient(mlflow_tracking_uri)
    
    registration_info = {
        'model_run_id': model_run_id,
        'model_name': model_name,
        'is_model_valid': is_model_valid,
        'validation_score': validation_score,
        'requested_stage': model_stage,
        'registered': False,
        'model_version': None,
        'actual_stage': None,
        'registration_timestamp': None
    }
    
    if is_model_valid:
        try:
            # Register the model
            model_uri = f"runs:/{model_run_id}/model"
            
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags={
                    "validation_score": str(validation_score),
                    "registered_via": "kubeflow_pipeline",
                    "validation_passed": "true"
                }
            )
            
            # Update model version description
            description = f"Model registered via Kubeflow pipeline. Validation score: {validation_score:.3f}"
            client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=description
            )
            
            # Transition to specified stage
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage=model_stage
            )
            
            registration_info.update({
                'registered': True,
                'model_version': model_version.version,
                'actual_stage': model_stage,
                'registration_timestamp': model_version.creation_timestamp
            })
            
            logger.info(f"Model registered successfully!")
            logger.info(f"Model name: {model_name}")
            logger.info(f"Version: {model_version.version}")
            logger.info(f"Stage: {model_stage}")
            
            registered = True
            version = model_version.version
            stage = model_stage
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            registration_info['error'] = str(e)
            registered = False
            version = ""
            stage = ""
    else:
        logger.info("Model validation failed. Skipping registration.")
        registration_info['skip_reason'] = "Model failed validation checks"
        registered = False
        version = ""
        stage = ""
    
    # Save registration result
    with open(registration_result.path, 'w') as f:
        json.dump(registration_info, f, indent=2, default=str)
    
    # Return output
    from collections import namedtuple
    RegistrationOutput = namedtuple('RegistrationOutput', ['registered', 'model_version', 'model_stage'])
    return RegistrationOutput(
        registered=registered,
        model_version=version,
        model_stage=stage
    )

# Example component factory function
def create_evaluation_pipeline_components():
    """
    Factory function to create evaluation pipeline components with default configurations.
    
    Returns:
        Dict of configured components
    """
    return {
        'evaluate_model': evaluate_model,
        'validate_model': model_validation,
        'register_model': register_model_if_valid
    }
