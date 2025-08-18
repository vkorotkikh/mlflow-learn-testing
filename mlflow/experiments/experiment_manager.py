"""
MLflow Experiment Management Utilities
This module provides utilities for managing MLflow experiments, models, and runs.
"""

import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import pandas as pd
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLflowExperimentManager:
    def __init__(self, tracking_uri="http://127.0.0.1:5000"):
        """
        Initialize MLflow experiment manager.
        
        Args:
            tracking_uri (str): MLflow tracking server URI
        """
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri)
        
        logger.info(f"MLflow client initialized with tracking URI: {tracking_uri}")

    def create_experiment(self, experiment_name: str, artifact_location: Optional[str] = None, 
                         tags: Optional[Dict[str, str]] = None) -> str:
        """
        Create a new MLflow experiment.
        
        Args:
            experiment_name (str): Name of the experiment
            artifact_location (str, optional): Artifact storage location
            tags (dict, optional): Experiment tags
            
        Returns:
            str: Experiment ID
        """
        try:
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=artifact_location,
                tags=tags
            )
            logger.info(f"Created experiment '{experiment_name}' with ID: {experiment_id}")
            return experiment_id
            
        except Exception as e:
            if "already exists" in str(e):
                experiment = mlflow.get_experiment_by_name(experiment_name)
                logger.info(f"Experiment '{experiment_name}' already exists with ID: {experiment.experiment_id}")
                return experiment.experiment_id
            else:
                logger.error(f"Failed to create experiment: {e}")
                raise

    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all MLflow experiments.
        
        Returns:
            List[Dict]: List of experiment information
        """
        experiments = self.client.search_experiments()
        
        experiment_list = []
        for exp in experiments:
            experiment_list.append({
                'experiment_id': exp.experiment_id,
                'name': exp.name,
                'lifecycle_stage': exp.lifecycle_stage,
                'artifact_location': exp.artifact_location,
                'tags': exp.tags,
                'creation_time': datetime.fromtimestamp(exp.creation_time / 1000.0)
            })
        
        return experiment_list

    def get_experiment_runs(self, experiment_name: str, 
                           max_results: int = 100) -> pd.DataFrame:
        """
        Get all runs for an experiment.
        
        Args:
            experiment_name (str): Name of the experiment
            max_results (int): Maximum number of runs to retrieve
            
        Returns:
            pd.DataFrame: DataFrame with run information
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.error(f"Experiment '{experiment_name}' not found")
            return pd.DataFrame()
        
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=max_results
        )
        
        run_data = []
        for run in runs:
            run_info = {
                'run_id': run.info.run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'Unnamed'),
                'status': run.info.status,
                'start_time': datetime.fromtimestamp(run.info.start_time / 1000.0),
                'end_time': datetime.fromtimestamp(run.info.end_time / 1000.0) if run.info.end_time else None,
                'artifact_uri': run.info.artifact_uri
            }
            
            # Add metrics
            for metric_name, metric_value in run.data.metrics.items():
                run_info[f"metric_{metric_name}"] = metric_value
            
            # Add parameters
            for param_name, param_value in run.data.params.items():
                run_info[f"param_{param_name}"] = param_value
                
            run_data.append(run_info)
        
        return pd.DataFrame(run_data)

    def register_model(self, model_uri: str, model_name: str, 
                      description: Optional[str] = None, 
                      tags: Optional[Dict[str, str]] = None) -> str:
        """
        Register a model in MLflow Model Registry.
        
        Args:
            model_uri (str): URI of the model (e.g., runs:/<run_id>/model)
            model_name (str): Name for the registered model
            description (str, optional): Model description
            tags (dict, optional): Model tags
            
        Returns:
            str: Model version
        """
        try:
            # Register the model
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags=tags
            )
            
            # Add description if provided
            if description:
                self.client.update_model_version(
                    name=model_name,
                    version=model_version.version,
                    description=description
                )
            
            logger.info(f"Registered model '{model_name}' version {model_version.version}")
            return model_version.version
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise

    def promote_model(self, model_name: str, version: str, stage: str) -> None:
        """
        Promote a model version to a specific stage.
        
        Args:
            model_name (str): Name of the registered model
            version (str): Version number
            stage (str): Target stage (Staging, Production, Archived)
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            logger.info(f"Promoted model '{model_name}' version {version} to {stage}")
            
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            raise

    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Get all versions of a registered model.
        
        Args:
            model_name (str): Name of the registered model
            
        Returns:
            List[Dict]: List of model version information
        """
        try:
            versions = self.client.get_latest_versions(model_name, stages=["None", "Staging", "Production", "Archived"])
            
            version_list = []
            for version in versions:
                version_list.append({
                    'version': version.version,
                    'stage': version.current_stage,
                    'description': version.description,
                    'creation_time': datetime.fromtimestamp(version.creation_timestamp / 1000.0),
                    'last_updated': datetime.fromtimestamp(version.last_updated_timestamp / 1000.0),
                    'run_id': version.run_id,
                    'status': version.status,
                    'source': version.source
                })
            
            return version_list
            
        except Exception as e:
            logger.error(f"Failed to get model versions: {e}")
            raise

    def compare_model_metrics(self, experiment_name: str, 
                             metric_names: List[str]) -> pd.DataFrame:
        """
        Compare metrics across runs in an experiment.
        
        Args:
            experiment_name (str): Name of the experiment
            metric_names (List[str]): List of metric names to compare
            
        Returns:
            pd.DataFrame: Comparison table
        """
        runs_df = self.get_experiment_runs(experiment_name)
        
        if runs_df.empty:
            logger.warning(f"No runs found for experiment '{experiment_name}'")
            return pd.DataFrame()
        
        # Extract specified metrics
        comparison_data = {
            'run_id': runs_df['run_id'],
            'run_name': runs_df['run_name'],
            'status': runs_df['status'],
            'start_time': runs_df['start_time']
        }
        
        for metric in metric_names:
            metric_col = f"metric_{metric}"
            if metric_col in runs_df.columns:
                comparison_data[metric] = runs_df[metric_col]
            else:
                comparison_data[metric] = None
                logger.warning(f"Metric '{metric}' not found in runs")
        
        return pd.DataFrame(comparison_data)

    def export_experiment_data(self, experiment_name: str, 
                              output_path: str = "experiment_data.json") -> None:
        """
        Export experiment data to JSON file.
        
        Args:
            experiment_name (str): Name of the experiment
            output_path (str): Output file path
        """
        try:
            # Get experiment info
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                raise ValueError(f"Experiment '{experiment_name}' not found")
            
            # Get runs data
            runs_df = self.get_experiment_runs(experiment_name)
            
            export_data = {
                'experiment_info': {
                    'experiment_id': experiment.experiment_id,
                    'name': experiment.name,
                    'lifecycle_stage': experiment.lifecycle_stage,
                    'artifact_location': experiment.artifact_location,
                    'creation_time': experiment.creation_time,
                    'last_update_time': experiment.last_update_time
                },
                'runs': runs_df.to_dict('records') if not runs_df.empty else [],
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Experiment data exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export experiment data: {e}")
            raise

    def cleanup_experiments(self, experiment_names: List[str], 
                           keep_best_n: int = 5, 
                           metric_name: str = "test_accuracy") -> None:
        """
        Clean up old runs in experiments, keeping only the best N runs.
        
        Args:
            experiment_names (List[str]): List of experiment names to clean
            keep_best_n (int): Number of best runs to keep
            metric_name (str): Metric to use for ranking runs
        """
        for exp_name in experiment_names:
            try:
                logger.info(f"Cleaning up experiment: {exp_name}")
                
                runs_df = self.get_experiment_runs(exp_name)
                metric_col = f"metric_{metric_name}"
                
                if runs_df.empty or metric_col not in runs_df.columns:
                    logger.warning(f"No runs or metric '{metric_name}' found for experiment '{exp_name}'")
                    continue
                
                # Sort by metric (descending) and keep top N
                sorted_runs = runs_df.sort_values(metric_col, ascending=False)
                runs_to_delete = sorted_runs.iloc[keep_best_n:]
                
                for _, run in runs_to_delete.iterrows():
                    self.client.delete_run(run['run_id'])
                    logger.info(f"Deleted run: {run['run_id']}")
                
                logger.info(f"Cleanup completed for '{exp_name}'. Kept {keep_best_n} best runs.")
                
            except Exception as e:
                logger.error(f"Failed to cleanup experiment '{exp_name}': {e}")

def main():
    """
    Main function to demonstrate experiment management capabilities.
    """
    manager = MLflowExperimentManager()
    
    # Create or get experiment
    exp_id = manager.create_experiment(
        "iris_classification",
        tags={"project": "mlops_pipeline", "dataset": "iris"}
    )
    
    # List all experiments
    experiments = manager.list_experiments()
    print("\nAvailable Experiments:")
    print("-" * 50)
    for exp in experiments:
        print(f"Name: {exp['name']}")
        print(f"ID: {exp['experiment_id']}")
        print(f"Stage: {exp['lifecycle_stage']}")
        print("-" * 50)
    
    # Get runs for iris experiment
    runs_df = manager.get_experiment_runs("iris_classification")
    if not runs_df.empty:
        print(f"\nFound {len(runs_df)} runs in iris_classification experiment")
        print("\nRun Summary:")
        print(runs_df[['run_name', 'status', 'metric_test_accuracy']].head())
    
    # Compare metrics if runs exist
    if not runs_df.empty:
        comparison = manager.compare_model_metrics(
            "iris_classification",
            ["test_accuracy", "train_accuracy", "overfitting"]
        )
        print("\nMetric Comparison:")
        print(comparison)

if __name__ == "__main__":
    main()


