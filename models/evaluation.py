"""
MLflow-enabled Model Evaluation Script
This script evaluates trained models and logs metrics to MLflow.
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import mlflow
import mlflow.sklearn
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, mlflow_tracking_uri="http://127.0.0.1:5000"):
        """
        Initialize the model evaluator with MLflow configuration.
        
        Args:
            mlflow_tracking_uri (str): MLflow tracking server URI
        """
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.model = None
        self.X_test = None
        self.y_test = None
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

    def load_test_data(self, test_size=0.2, random_state=42):
        """
        Load and prepare test data from Iris dataset.
        
        Args:
            test_size (float): Proportion of dataset to include in test split
            random_state (int): Random state for reproducibility
        """
        logger.info("Loading test data...")
        
        # Load the iris dataset
        iris = load_iris()
        X = iris.data
        y = iris.target
        
        # Split the data (same as training to ensure consistency)
        _, self.X_test, _, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Test data loaded: {self.X_test.shape[0]} samples")
        
        return {
            'feature_names': iris.feature_names,
            'target_names': iris.target_names
        }

    def load_model_from_mlflow(self, model_name="iris_random_forest", model_version="latest"):
        """
        Load a registered model from MLflow.
        
        Args:
            model_name (str): Name of the registered model
            model_version (str): Version of the model ("latest" or specific version number)
        """
        logger.info(f"Loading model {model_name} version {model_version} from MLflow...")
        
        try:
            if model_version == "latest":
                model_uri = f"models:/{model_name}/Latest"
            else:
                model_uri = f"models:/{model_name}/{model_version}"
            
            self.model = mlflow.sklearn.load_model(model_uri)
            logger.info("Model loaded successfully from MLflow")
            
        except Exception as e:
            logger.error(f"Failed to load model from MLflow: {e}")
            raise

    def load_model_from_file(self, model_path):
        """
        Load a model from a local file.
        
        Args:
            model_path (str): Path to the model file
        """
        logger.info(f"Loading model from {model_path}...")
        
        try:
            self.model = joblib.load(model_path)
            logger.info("Model loaded successfully from file")
            
        except Exception as e:
            logger.error(f"Failed to load model from file: {e}")
            raise

    def evaluate_model(self, run_name="model_evaluation"):
        """
        Evaluate the loaded model and log metrics to MLflow.
        
        Args:
            run_name (str): Name for the MLflow run
        """
        if self.model is None:
            raise ValueError("No model loaded for evaluation!")
        
        if self.X_test is None or self.y_test is None:
            raise ValueError("No test data loaded!")
        
        logger.info("Starting model evaluation...")
        
        with mlflow.start_run(run_name=run_name) as run:
            # Make predictions
            y_pred = self.model.predict(self.X_test)
            y_proba = self.model.predict_proba(self.X_test)
            
            # Calculate basic metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            # Calculate AUC for multiclass (one-vs-rest)
            y_test_binarized = label_binarize(self.y_test, classes=[0, 1, 2])
            auc_score = roc_auc_score(y_test_binarized, y_proba, multi_class='ovr', average='weighted')
            
            # Log metrics
            mlflow.log_metric("evaluation_accuracy", accuracy)
            mlflow.log_metric("evaluation_precision", precision)
            mlflow.log_metric("evaluation_recall", recall)
            mlflow.log_metric("evaluation_f1", f1)
            mlflow.log_metric("evaluation_auc", auc_score)
            
            # Calculate per-class metrics
            precision_per_class = precision_score(self.y_test, y_pred, average=None)
            recall_per_class = recall_score(self.y_test, y_pred, average=None)
            f1_per_class = f1_score(self.y_test, y_pred, average=None)
            
            class_names = ['setosa', 'versicolor', 'virginica']
            for i, class_name in enumerate(class_names):
                mlflow.log_metric(f"precision_{class_name}", precision_per_class[i])
                mlflow.log_metric(f"recall_{class_name}", recall_per_class[i])
                mlflow.log_metric(f"f1_{class_name}", f1_per_class[i])
            
            # Generate and log confusion matrix plot
            self._plot_confusion_matrix(self.y_test, y_pred, class_names)
            
            # Generate and log classification report
            report = classification_report(self.y_test, y_pred, target_names=class_names)
            
            with open("evaluation_report.txt", "w") as f:
                f.write("Model Evaluation Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Test Samples: {len(self.y_test)}\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"F1-Score: {f1:.4f}\n")
                f.write(f"AUC Score: {auc_score:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(report)
            
            mlflow.log_artifact("evaluation_report.txt")
            
            # Clean up temporary files
            for file in ["evaluation_report.txt", "confusion_matrix.png"]:
                if os.path.exists(file):
                    os.remove(file)
            
            logger.info("Model evaluation completed!")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"F1-Score: {f1:.4f}")
            logger.info(f"MLflow Run ID: {run.info.run_id}")
            
            return {
                'run_id': run.info.run_id,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_score': auc_score
            }

    def _plot_confusion_matrix(self, y_true, y_pred, class_names):
        """
        Generate and save confusion matrix plot.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        mlflow.log_artifact('confusion_matrix.png')

    def compare_models(self, model_name="iris_random_forest", versions=None):
        """
        Compare multiple versions of a model.
        
        Args:
            model_name (str): Name of the registered model
            versions (list): List of version numbers to compare (None for all versions)
        """
        logger.info(f"Comparing versions of model: {model_name}")
        
        client = mlflow.tracking.MlflowClient()
        
        try:
            model_versions = client.get_latest_versions(model_name)
            
            if not model_versions:
                logger.warning(f"No versions found for model: {model_name}")
                return
            
            comparison_results = []
            
            for version in model_versions:
                if versions is None or int(version.version) in versions:
                    logger.info(f"Evaluating version {version.version}...")
                    
                    # Load model version
                    model_uri = f"models:/{model_name}/{version.version}"
                    self.model = mlflow.sklearn.load_model(model_uri)
                    
                    # Evaluate
                    result = self.evaluate_model(f"comparison_v{version.version}")
                    result['version'] = version.version
                    result['stage'] = version.current_stage
                    
                    comparison_results.append(result)
            
            # Create comparison report
            comparison_df = pd.DataFrame(comparison_results)
            comparison_df.to_csv("model_comparison.csv", index=False)
            
            with mlflow.start_run(run_name="model_comparison_summary"):
                mlflow.log_artifact("model_comparison.csv")
                
                # Log best model metrics
                best_model = comparison_df.loc[comparison_df['accuracy'].idxmax()]
                mlflow.log_metric("best_model_version", float(best_model['version']))
                mlflow.log_metric("best_model_accuracy", best_model['accuracy'])
            
            os.remove("model_comparison.csv")
            
            logger.info("Model comparison completed!")
            return comparison_results
            
        except Exception as e:
            logger.error(f"Error during model comparison: {e}")
            raise

def main():
    """
    Main function to run model evaluation.
    """
    evaluator = ModelEvaluator()
    
    # Load test data
    evaluator.load_test_data()
    
    try:
        # Try to load from MLflow first
        evaluator.load_model_from_mlflow("iris_random_forest", "latest")
        logger.info("Loaded model from MLflow")
    except Exception as e:
        logger.warning(f"Could not load from MLflow: {e}")
        # Fallback to local file
        try:
            evaluator.load_model_from_file("models/iris_rf_model.joblib")
            logger.info("Loaded model from local file")
        except Exception as e:
            logger.error(f"Could not load model from file either: {e}")
            return
    
    # Evaluate the model
    results = evaluator.evaluate_model()
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for metric, value in results.items():
        if metric != 'run_id':
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")

if __name__ == "__main__":
    main()


