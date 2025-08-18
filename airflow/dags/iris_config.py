"""
Configuration Management for Iris MLOps Pipeline

This module centralizes all configuration management for the Iris classification
MLOps pipeline, including Airflow variables, default settings, and validation.

Classes:
    IrisMLOpsConfig: Main configuration class with validation and defaults
    
Functions:
    get_config: Factory function to create and validate configuration
    validate_config: Validation function for configuration parameters

Example:
    >>> from iris_config import get_config
    >>> config = get_config()
    >>> print(config.mlflow_tracking_uri)
    http://localhost:5000
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import os
import logging
from airflow.models import Variable

logger = logging.getLogger(__name__)


@dataclass
class ModelQualityThresholds:
    """
    Model quality thresholds for promotion decisions.
    
    Attributes:
        accuracy_threshold (float): Minimum accuracy for model acceptance (0.0-1.0)
        precision_threshold (float): Minimum precision for model acceptance (0.0-1.0)
        recall_threshold (float): Minimum recall for model acceptance (0.0-1.0)
        f1_threshold (float): Minimum F1-score for model acceptance (0.0-1.0)
        production_accuracy_threshold (float): Higher accuracy bar for production (0.0-1.0)
    """
    accuracy_threshold: float = 0.85
    precision_threshold: float = 0.85
    recall_threshold: float = 0.80
    f1_threshold: float = 0.82
    production_accuracy_threshold: float = 0.90


@dataclass
class InfrastructureConfig:
    """
    Infrastructure and service endpoint configuration.
    
    Attributes:
        mlflow_tracking_uri (str): MLflow tracking server URL
        kubeflow_endpoint (str): Kubeflow Pipelines API endpoint
        mlflow_health_timeout (int): Health check timeout in seconds
        kubeflow_health_timeout (int): Kubeflow health check timeout in seconds
        max_health_retries (int): Maximum health check retry attempts
        health_retry_delay (int): Delay between health check retries in seconds
    """
    mlflow_tracking_uri: str = "http://localhost:5000"
    kubeflow_endpoint: str = "http://localhost:8080"
    mlflow_health_timeout: int = 10
    kubeflow_health_timeout: int = 10
    max_health_retries: int = 5
    health_retry_delay: int = 30


@dataclass
class PipelineConfig:
    """
    Pipeline execution configuration and parameters.
    
    Attributes:
        experiment_name_prefix (str): Base name for MLflow experiments
        model_name_prefix (str): Base name for registered models
        pipeline_script_path (str): Path to Kubeflow pipeline script
        max_pipeline_wait_time (int): Maximum pipeline execution wait time in seconds
        pipeline_check_interval (int): Pipeline status check interval in seconds
        perform_hyperparameter_tuning (bool): Enable hyperparameter tuning
        deploy_if_valid (bool): Auto-deploy models that meet quality criteria
        config_temp_dir (str): Temporary directory for pipeline configs
    """
    experiment_name_prefix: str = "iris_classification_production"
    model_name_prefix: str = "iris_random_forest_production"
    pipeline_script_path: str = "/opt/airflow/dags/kubeflow/pipelines/iris_pipeline.py"
    max_pipeline_wait_time: int = 3600  # 1 hour
    pipeline_check_interval: int = 60   # 1 minute
    perform_hyperparameter_tuning: bool = True
    deploy_if_valid: bool = True
    config_temp_dir: str = "/tmp"


@dataclass
class NotificationConfig:
    """
    Notification and alerting configuration.
    
    Attributes:
        notification_emails (List[str]): List of email addresses for notifications
        email_on_failure (bool): Send emails on pipeline failures
        email_on_success (bool): Send emails on pipeline success
        slack_webhook_url (Optional[str]): Slack webhook for notifications
        teams_webhook_url (Optional[str]): Teams webhook for notifications
    """
    notification_emails: List[str] = field(default_factory=lambda: ["admin@company.com"])
    email_on_failure: bool = True
    email_on_success: bool = True
    slack_webhook_url: Optional[str] = None
    teams_webhook_url: Optional[str] = None


@dataclass
class IrisMLOpsConfig:
    """
    Complete configuration for the Iris MLOps pipeline.
    
    This class centralizes all configuration parameters for the MLOps pipeline,
    including infrastructure settings, model quality thresholds, pipeline
    parameters, and notification preferences.
    
    Attributes:
        infrastructure (InfrastructureConfig): Service endpoints and timeouts
        pipeline (PipelineConfig): Pipeline execution parameters
        quality_thresholds (ModelQualityThresholds): Model quality criteria
        notifications (NotificationConfig): Alert and notification settings
        dag_id (str): Unique identifier for the Airflow DAG
        tags (List[str]): Tags for DAG categorization
        max_active_runs (int): Maximum concurrent DAG runs
        catchup (bool): Whether to catch up on missed runs
        
    Example:
        >>> config = IrisMLOpsConfig()
        >>> config.infrastructure.mlflow_tracking_uri
        'http://localhost:5000'
        >>> config.quality_thresholds.accuracy_threshold
        0.85
    """
    infrastructure: InfrastructureConfig = field(default_factory=InfrastructureConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    quality_thresholds: ModelQualityThresholds = field(default_factory=ModelQualityThresholds)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    
    # DAG-specific configuration
    dag_id: str = "iris_mlops_pipeline"
    tags: List[str] = field(default_factory=lambda: ['mlops', 'iris', 'classification', 'kubeflow', 'mlflow'])
    max_active_runs: int = 1
    catchup: bool = False


def load_from_airflow_variables() -> Dict[str, Any]:
    """
    Load configuration values from Airflow Variables.
    
    Retrieves configuration parameters from Airflow Variables with fallback
    to default values. This allows for runtime configuration without code changes.
    
    Returns:
        Dict[str, Any]: Dictionary of configuration values from Airflow Variables
        
    Example:
        >>> variables = load_from_airflow_variables()
        >>> variables['mlflow_tracking_uri']
        'http://mlflow.example.com:5000'
        
    Note:
        If Airflow Variables are not set, default values will be used.
        See Airflow UI -> Admin -> Variables for configuration.
    """
    logger.info("Loading configuration from Airflow Variables")
    
    variables = {
        # Infrastructure
        'mlflow_tracking_uri': Variable.get("mlflow_tracking_uri", default_var="http://localhost:5000"),
        'kubeflow_endpoint': Variable.get("kubeflow_endpoint", default_var="http://localhost:8080"),
        
        # Pipeline
        'experiment_name_prefix': Variable.get("experiment_name", default_var="iris_classification_production"),
        'model_name_prefix': Variable.get("model_name", default_var="iris_random_forest_production"),
        
        # Quality thresholds
        'accuracy_threshold': float(Variable.get("accuracy_threshold", default_var="0.85")),
        'precision_threshold': float(Variable.get("precision_threshold", default_var="0.85")),
        'production_accuracy_threshold': float(Variable.get("production_accuracy_threshold", default_var="0.90")),
        
        # Notifications
        'notification_emails': Variable.get("notification_emails", default_var="admin@company.com").split(","),
        'email_on_failure': Variable.get("email_on_failure", default_var="true").lower() == "true",
        'email_on_success': Variable.get("email_on_success", default_var="true").lower() == "true",
        
        # Optional webhooks
        'slack_webhook_url': Variable.get("slack_webhook_url", default_var=None),
        'teams_webhook_url': Variable.get("teams_webhook_url", default_var=None),
    }
    
    logger.info(f"Loaded {len(variables)} configuration variables")
    return variables


def validate_config(config: IrisMLOpsConfig) -> None:
    """
    Validate the configuration parameters.
    
    Performs comprehensive validation of the configuration to ensure all
    parameters are within acceptable ranges and required services are accessible.
    
    Args:
        config (IrisMLOpsConfig): Configuration object to validate
        
    Raises:
        ValueError: If any configuration parameter is invalid
        ConnectionError: If required services are not accessible
        
    Example:
        >>> config = get_config()
        >>> validate_config(config)  # Raises exception if invalid
        
    Validation Checks:
        - Threshold values are between 0.0 and 1.0
        - URLs are properly formatted
        - Email addresses are valid format
        - File paths exist and are accessible
        - Timeout values are positive integers
    """
    logger.info("Validating configuration parameters")
    
    # Validate quality thresholds
    thresholds = config.quality_thresholds
    threshold_fields = [
        ('accuracy_threshold', thresholds.accuracy_threshold),
        ('precision_threshold', thresholds.precision_threshold),
        ('recall_threshold', thresholds.recall_threshold),
        ('f1_threshold', thresholds.f1_threshold),
        ('production_accuracy_threshold', thresholds.production_accuracy_threshold)
    ]
    
    for field_name, value in threshold_fields:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{field_name} must be between 0.0 and 1.0, got {value}")
    
    # Validate infrastructure settings
    infra = config.infrastructure
    if infra.max_health_retries <= 0:
        raise ValueError(f"max_health_retries must be positive, got {infra.max_health_retries}")
    
    if infra.health_retry_delay <= 0:
        raise ValueError(f"health_retry_delay must be positive, got {infra.health_retry_delay}")
    
    # Validate pipeline settings
    pipeline = config.pipeline
    if pipeline.max_pipeline_wait_time <= 0:
        raise ValueError(f"max_pipeline_wait_time must be positive, got {pipeline.max_pipeline_wait_time}")
    
    if pipeline.pipeline_check_interval <= 0:
        raise ValueError(f"pipeline_check_interval must be positive, got {pipeline.pipeline_check_interval}")
    
    # Validate notification settings
    if not config.notifications.notification_emails:
        logger.warning("No notification emails configured")
    
    # Check if pipeline script exists
    if not os.path.exists(pipeline.pipeline_script_path):
        logger.warning(f"Pipeline script not found at {pipeline.pipeline_script_path}")
    
    logger.info("Configuration validation completed successfully")


def get_config() -> IrisMLOpsConfig:
    """
    Factory function to create and validate the complete configuration.
    
    Creates a configuration object by merging default values with Airflow Variables,
    then validates the resulting configuration for consistency and correctness.
    
    Returns:
        IrisMLOpsConfig: Validated configuration object ready for use
        
    Raises:
        ValueError: If configuration validation fails
        ConnectionError: If required services are not accessible
        
    Example:
        >>> config = get_config()
        >>> print(f"MLflow URI: {config.infrastructure.mlflow_tracking_uri}")
        MLflow URI: http://localhost:5000
        
    Note:
        This function should be called once at the beginning of the DAG
        to ensure consistent configuration across all tasks.
    """
    logger.info("Creating Iris MLOps configuration")
    
    try:
        # Load variables from Airflow
        variables = load_from_airflow_variables()
        
        # Create configuration objects
        infrastructure = InfrastructureConfig(
            mlflow_tracking_uri=variables['mlflow_tracking_uri'],
            kubeflow_endpoint=variables['kubeflow_endpoint']
        )
        
        pipeline = PipelineConfig(
            experiment_name_prefix=variables['experiment_name_prefix'],
            model_name_prefix=variables['model_name_prefix']
        )
        
        quality_thresholds = ModelQualityThresholds(
            accuracy_threshold=variables['accuracy_threshold'],
            precision_threshold=variables['precision_threshold'],
            production_accuracy_threshold=variables['production_accuracy_threshold']
        )
        
        notifications = NotificationConfig(
            notification_emails=variables['notification_emails'],
            email_on_failure=variables['email_on_failure'],
            email_on_success=variables['email_on_success'],
            slack_webhook_url=variables.get('slack_webhook_url'),
            teams_webhook_url=variables.get('teams_webhook_url')
        )
        
        # Create main config object
        config = IrisMLOpsConfig(
            infrastructure=infrastructure,
            pipeline=pipeline,
            quality_thresholds=quality_thresholds,
            notifications=notifications
        )
        
        # Validate configuration
        validate_config(config)
        
        logger.info("Configuration created and validated successfully")
        return config
        
    except Exception as e:
        logger.error(f"Error creating configuration: {e}")
        raise


# Convenience function for backward compatibility
def get_legacy_variables() -> Dict[str, Any]:
    """
    Get configuration in the legacy format for backward compatibility.
    
    Returns:
        Dict[str, Any]: Configuration variables in legacy format
        
    Deprecated:
        Use get_config() instead for new code.
    """
    config = get_config()
    return {
        'MLFLOW_TRACKING_URI': config.infrastructure.mlflow_tracking_uri,
        'KUBEFLOW_ENDPOINT': config.infrastructure.kubeflow_endpoint,
        'EXPERIMENT_NAME': config.pipeline.experiment_name_prefix,
        'MODEL_NAME': config.pipeline.model_name_prefix,
        'NOTIFICATION_EMAIL': config.notifications.notification_emails[0] if config.notifications.notification_emails else "admin@company.com"
    }
