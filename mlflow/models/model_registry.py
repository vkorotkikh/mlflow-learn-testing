"""
MLflow Model Registry Management
This module provides utilities for managing models in the MLflow Model Registry.
"""

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
import pandas as pd
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRegistryManager:
    def __init__(self, tracking_uri="http://127.0.0.1:5000"):
        """
        Initialize MLflow Model Registry manager.
        
        Args:
            tracking_uri (str): MLflow tracking server URI
        """
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri)
        
        logger.info(f"Model Registry client initialized with tracking URI: {tracking_uri}")

    def register_model_from_run(self, run_id: str, model_name: str, 
                               description: Optional[str] = None,
                               tags: Optional[Dict[str, str]] = None) -> ModelVersion:
        """
        Register a model from a specific MLflow run.
        
        Args:
            run_id (str): MLflow run ID
            model_name (str): Name for the registered model
            description (str, optional): Model description
            tags (dict, optional): Model tags
            
        Returns:
            ModelVersion: Registered model version
        """
        try:
            model_uri = f"runs:/{run_id}/model"
            
            # Register the model
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags=tags
            )
            
            # Update description if provided
            if description:
                self.client.update_model_version(
                    name=model_name,
                    version=model_version.version,
                    description=description
                )
            
            logger.info(f"Registered model '{model_name}' version {model_version.version} from run {run_id}")
            return model_version
            
        except Exception as e:
            logger.error(f"Failed to register model from run {run_id}: {e}")
            raise

    def create_registered_model(self, model_name: str, 
                               description: Optional[str] = None,
                               tags: Optional[Dict[str, str]] = None) -> None:
        """
        Create a new registered model in the registry.
        
        Args:
            model_name (str): Name of the model
            description (str, optional): Model description
            tags (dict, optional): Model tags
        """
        try:
            self.client.create_registered_model(
                name=model_name,
                description=description,
                tags=tags
            )
            logger.info(f"Created registered model: {model_name}")
            
        except Exception as e:
            if "already exists" in str(e):
                logger.info(f"Registered model '{model_name}' already exists")
            else:
                logger.error(f"Failed to create registered model: {e}")
                raise

    def list_registered_models(self) -> List[Dict[str, Any]]:
        """
        List all registered models.
        
        Returns:
            List[Dict]: List of registered model information
        """
        try:
            models = self.client.search_registered_models()
            
            model_list = []
            for model in models:
                model_info = {
                    'name': model.name,
                    'description': model.description,
                    'creation_time': datetime.fromtimestamp(model.creation_timestamp / 1000.0),
                    'last_updated': datetime.fromtimestamp(model.last_updated_timestamp / 1000.0),
                    'latest_versions': []
                }
                
                # Get latest versions for each stage
                latest_versions = self.client.get_latest_versions(model.name)
                for version in latest_versions:
                    model_info['latest_versions'].append({
                        'version': version.version,
                        'stage': version.current_stage,
                        'run_id': version.run_id
                    })
                
                model_list.append(model_info)
            
            return model_list
            
        except Exception as e:
            logger.error(f"Failed to list registered models: {e}")
            raise

    def get_model_version_details(self, model_name: str, 
                                 version: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get detailed information about model versions.
        
        Args:
            model_name (str): Name of the registered model
            version (str, optional): Specific version number (None for all versions)
            
        Returns:
            List[Dict]: List of model version details
        """
        try:
            if version:
                versions = [self.client.get_model_version(model_name, version)]
            else:
                # Get all versions
                versions = self.client.search_model_versions(f"name='{model_name}'")
            
            version_details = []
            for v in versions:
                # Get run details
                run = self.client.get_run(v.run_id)
                
                detail = {
                    'version': v.version,
                    'stage': v.current_stage,
                    'description': v.description,
                    'creation_time': datetime.fromtimestamp(v.creation_timestamp / 1000.0),
                    'last_updated': datetime.fromtimestamp(v.last_updated_timestamp / 1000.0),
                    'run_id': v.run_id,
                    'status': v.status,
                    'source': v.source,
                    'run_name': run.data.tags.get('mlflow.runName', 'Unnamed'),
                    'metrics': dict(run.data.metrics),
                    'params': dict(run.data.params)
                }
                
                version_details.append(detail)
            
            return version_details
            
        except Exception as e:
            logger.error(f"Failed to get model version details: {e}")
            raise

    def transition_model_stage(self, model_name: str, version: str, 
                              stage: str, archive_existing: bool = True) -> None:
        """
        Transition a model version to a new stage.
        
        Args:
            model_name (str): Name of the registered model
            version (str): Version number
            stage (str): Target stage (None, Staging, Production, Archived)
            archive_existing (bool): Whether to archive existing models in target stage
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing
            )
            logger.info(f"Transitioned model '{model_name}' version {version} to {stage}")
            
        except Exception as e:
            logger.error(f"Failed to transition model stage: {e}")
            raise

    def delete_model_version(self, model_name: str, version: str) -> None:
        """
        Delete a specific model version.
        
        Args:
            model_name (str): Name of the registered model
            version (str): Version number to delete
        """
        try:
            self.client.delete_model_version(model_name, version)
            logger.info(f"Deleted model '{model_name}' version {version}")
            
        except Exception as e:
            logger.error(f"Failed to delete model version: {e}")
            raise

    def delete_registered_model(self, model_name: str) -> None:
        """
        Delete a registered model and all its versions.
        
        Args:
            model_name (str): Name of the registered model
        """
        try:
            self.client.delete_registered_model(model_name)
            logger.info(f"Deleted registered model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to delete registered model: {e}")
            raise

    def compare_model_versions(self, model_name: str, 
                              metric_names: List[str]) -> pd.DataFrame:
        """
        Compare metrics across different versions of a model.
        
        Args:
            model_name (str): Name of the registered model
            metric_names (List[str]): List of metrics to compare
            
        Returns:
            pd.DataFrame: Comparison table
        """
        try:
            version_details = self.get_model_version_details(model_name)
            
            if not version_details:
                logger.warning(f"No versions found for model '{model_name}'")
                return pd.DataFrame()
            
            comparison_data = []
            for detail in version_details:
                row = {
                    'version': detail['version'],
                    'stage': detail['stage'],
                    'run_name': detail['run_name'],
                    'creation_time': detail['creation_time']
                }
                
                # Add requested metrics
                for metric in metric_names:
                    row[metric] = detail['metrics'].get(metric, None)
                
                comparison_data.append(row)
            
            df = pd.DataFrame(comparison_data)
            df = df.sort_values('version', ascending=False)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to compare model versions: {e}")
            raise

    def get_production_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the current production model version.
        
        Args:
            model_name (str): Name of the registered model
            
        Returns:
            Dict or None: Production model details
        """
        try:
            production_versions = self.client.get_latest_versions(
                model_name, 
                stages=["Production"]
            )
            
            if not production_versions:
                logger.info(f"No production version found for model '{model_name}'")
                return None
            
            # Should only be one production version
            prod_version = production_versions[0]
            details = self.get_model_version_details(model_name, prod_version.version)
            
            return details[0] if details else None
            
        except Exception as e:
            logger.error(f"Failed to get production model: {e}")
            raise

    def promote_best_model(self, model_name: str, metric_name: str, 
                          higher_is_better: bool = True) -> str:
        """
        Automatically promote the best performing model to Production.
        
        Args:
            model_name (str): Name of the registered model
            metric_name (str): Metric to use for comparison
            higher_is_better (bool): Whether higher metric values are better
            
        Returns:
            str: Version number of promoted model
        """
        try:
            # Get all non-archived versions
            all_versions = self.client.search_model_versions(f"name='{model_name}'")
            active_versions = [v for v in all_versions if v.current_stage != "Archived"]
            
            if not active_versions:
                raise ValueError(f"No active versions found for model '{model_name}'")
            
            # Get version details and find best performing
            version_details = []
            for version in active_versions:
                details = self.get_model_version_details(model_name, version.version)[0]
                if metric_name in details['metrics']:
                    version_details.append(details)
            
            if not version_details:
                raise ValueError(f"No versions found with metric '{metric_name}'")
            
            # Sort by metric
            version_details.sort(
                key=lambda x: x['metrics'][metric_name], 
                reverse=higher_is_better
            )
            
            best_version = version_details[0]
            
            # Promote to production
            self.transition_model_stage(
                model_name, 
                best_version['version'], 
                "Production"
            )
            
            logger.info(f"Promoted model '{model_name}' version {best_version['version']} "
                       f"to Production ({metric_name}: {best_version['metrics'][metric_name]})")
            
            return best_version['version']
            
        except Exception as e:
            logger.error(f"Failed to promote best model: {e}")
            raise

    def model_deployment_readiness(self, model_name: str, version: str) -> Dict[str, Any]:
        """
        Check if a model version is ready for deployment.
        
        Args:
            model_name (str): Name of the registered model
            version (str): Version number
            
        Returns:
            Dict: Readiness assessment
        """
        try:
            details = self.get_model_version_details(model_name, version)[0]
            
            # Define readiness criteria
            readiness_checks = {
                'has_description': bool(details['description']),
                'has_metrics': bool(details['metrics']),
                'has_accuracy_metric': 'test_accuracy' in details['metrics'],
                'accuracy_threshold': details['metrics'].get('test_accuracy', 0) >= 0.8,
                'is_not_archived': details['stage'] != 'Archived',
                'has_recent_training': (datetime.now() - details['creation_time']).days <= 30
            }
            
            # Calculate overall readiness score
            passed_checks = sum(readiness_checks.values())
            total_checks = len(readiness_checks)
            readiness_score = passed_checks / total_checks
            
            assessment = {
                'model_name': model_name,
                'version': version,
                'readiness_score': readiness_score,
                'is_ready': readiness_score >= 0.8,
                'checks': readiness_checks,
                'recommendations': []
            }
            
            # Add recommendations for failed checks
            if not readiness_checks['has_description']:
                assessment['recommendations'].append("Add model description")
            if not readiness_checks['has_accuracy_metric']:
                assessment['recommendations'].append("Ensure test_accuracy metric is logged")
            if not readiness_checks['accuracy_threshold']:
                assessment['recommendations'].append("Improve model accuracy (>= 80%)")
            
            return assessment
            
        except Exception as e:
            logger.error(f"Failed to assess model readiness: {e}")
            raise

def main():
    """
    Main function to demonstrate model registry management.
    """
    registry = ModelRegistryManager()
    
    # List all registered models
    models = registry.list_registered_models()
    print("Registered Models:")
    print("-" * 50)
    for model in models:
        print(f"Name: {model['name']}")
        print(f"Description: {model['description']}")
        print(f"Latest Versions: {model['latest_versions']}")
        print("-" * 50)
    
    # If iris model exists, show version comparison
    iris_models = [m for m in models if m['name'] == 'iris_random_forest']
    if iris_models:
        print("\nIris Model Version Comparison:")
        comparison = registry.compare_model_versions(
            'iris_random_forest',
            ['test_accuracy', 'train_accuracy', 'overfitting']
        )
        print(comparison)
        
        # Check production model readiness
        if not comparison.empty:
            latest_version = comparison.iloc[0]['version']
            readiness = registry.model_deployment_readiness('iris_random_forest', latest_version)
            print(f"\nModel Readiness Assessment (v{latest_version}):")
            print(f"Ready for deployment: {readiness['is_ready']}")
            print(f"Readiness score: {readiness['readiness_score']:.2f}")
            if readiness['recommendations']:
                print("Recommendations:")
                for rec in readiness['recommendations']:
                    print(f"  - {rec}")

if __name__ == "__main__":
    main()


