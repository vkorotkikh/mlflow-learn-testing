"""
Grafana Integration for MLOps Pipeline

This module provides integration with Grafana for monitoring, alerting, and
dashboard management within the MLOps pipeline.

Classes:
    GrafanaManager: Main class for Grafana operations
    DashboardManager: Dashboard creation and management
    AlertManager: Alert rule management and notification

Functions:
    Grafana API operations
    Dashboard provisioning
    Metric annotation and tagging
    Alert configuration

Example:
    >>> from grafana_integration import GrafanaManager
    >>> from iris_config import get_config
    >>> 
    >>> config = get_config()
    >>> grafana = GrafanaManager(config)
    >>> grafana.send_pipeline_annotation("Pipeline completed successfully")
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging
import requests
from requests.auth import HTTPBasicAuth
import time

from iris_config import IrisMLOpsConfig
from iris_ml_operations import ModelAnalysisResults, PromotionDecision

logger = logging.getLogger(__name__)


class GrafanaAPIError(Exception):
    """Exception raised for Grafana API errors."""
    pass


class DashboardNotFoundError(Exception):
    """Exception raised when dashboard is not found."""
    pass


class GrafanaManager:
    """
    Main Grafana manager for MLOps pipeline integration.
    
    Provides high-level interface for Grafana operations including
    annotations, dashboard management, and alerting integration.
    
    Attributes:
        config (IrisMLOpsConfig): Configuration object
        grafana_url (str): Grafana server URL
        auth (HTTPBasicAuth): Authentication for Grafana API
        
    Example:
        >>> config = get_config()
        >>> grafana = GrafanaManager(config)
        >>> grafana.send_pipeline_annotation("Model promoted to production")
    """
    
    def __init__(self, config: IrisMLOpsConfig):
        """
        Initialize Grafana manager.
        
        Args:
            config (IrisMLOpsConfig): Complete configuration object
        """
        self.config = config
        self.grafana_url = config.monitoring.grafana['url']
        self.auth = HTTPBasicAuth(
            config.monitoring.grafana['admin_user'],
            config.monitoring.grafana['admin_password']
        )
        self.session = requests.Session()
        self.session.auth = self.auth
        self._validate_connection()
    
    def _validate_connection(self) -> None:
        """
        Validate connection to Grafana server.
        
        Raises:
            GrafanaAPIError: If connection fails
        """
        try:
            response = self.session.get(f"{self.grafana_url}/api/health")
            if response.status_code != 200:
                raise GrafanaAPIError(f"Grafana health check failed: {response.status_code}")
            logger.info("Grafana connection validated successfully")
        except requests.exceptions.RequestException as e:
            raise GrafanaAPIError(f"Cannot connect to Grafana: {e}")
    
    def send_pipeline_annotation(
        self,
        text: str,
        tags: Optional[List[str]] = None,
        dashboard_id: Optional[int] = None,
        panel_id: Optional[int] = None,
        time_start: Optional[datetime] = None,
        time_end: Optional[datetime] = None
    ) -> bool:
        """
        Send annotation to Grafana for pipeline events.
        
        Annotations provide context for events in Grafana dashboards,
        helping correlate pipeline activities with metrics changes.
        
        Args:
            text (str): Annotation text/description
            tags (Optional[List[str]]): Tags for categorization
            dashboard_id (Optional[int]): Specific dashboard ID
            panel_id (Optional[int]): Specific panel ID
            time_start (Optional[datetime]): Event start time
            time_end (Optional[datetime]): Event end time (for regions)
            
        Returns:
            bool: True if annotation was sent successfully
            
        Example:
            >>> grafana.send_pipeline_annotation(
            ...     "Model training started",
            ...     tags=["training", "iris-model"],
            ...     time_start=datetime.now()
            ... )
        """
        if tags is None:
            tags = ["mlops", "pipeline"]
        
        if time_start is None:
            time_start = datetime.now()
        
        annotation_data = {
            "text": text,
            "tags": tags,
            "time": int(time_start.timestamp() * 1000)  # Grafana expects milliseconds
        }
        
        if time_end:
            annotation_data["timeEnd"] = int(time_end.timestamp() * 1000)
            annotation_data["isRegion"] = True
        
        if dashboard_id:
            annotation_data["dashboardId"] = dashboard_id
        
        if panel_id:
            annotation_data["panelId"] = panel_id
        
        try:
            response = self.session.post(
                f"{self.grafana_url}/api/annotations",
                json=annotation_data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code in [200, 201]:
                logger.info(f"Annotation sent successfully: {text}")
                return True
            else:
                logger.error(f"Failed to send annotation: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending annotation: {e}")
            return False
    
    def send_model_performance_annotation(
        self,
        analysis_results: ModelAnalysisResults,
        promotion_decision: Optional[PromotionDecision] = None
    ) -> bool:
        """
        Send model performance annotation with detailed metrics.
        
        Args:
            analysis_results (ModelAnalysisResults): Model analysis results
            promotion_decision (Optional[PromotionDecision]): Promotion decision
            
        Returns:
            bool: True if annotation was sent successfully
        """
        # Create detailed annotation text
        text = f"Model Analysis: {analysis_results.model_type}\n"
        text += f"Accuracy: {analysis_results.accuracy:.3f}, "
        text += f"Precision: {analysis_results.precision:.3f}, "
        text += f"Recall: {analysis_results.recall:.3f}, "
        text += f"F1: {analysis_results.f1_score:.3f}\n"
        
        if promotion_decision:
            text += f"Promotion: {'✅ Promoted' if promotion_decision.should_promote else '❌ Not Promoted'}"
        
        tags = [
            "model-performance",
            "analysis",
            analysis_results.experiment_name,
            analysis_results.model_type.lower()
        ]
        
        if analysis_results.meets_quality_thresholds:
            tags.append("quality-passed")
        else:
            tags.append("quality-failed")
        
        if promotion_decision and promotion_decision.should_promote:
            tags.append("promoted")
        
        return self.send_pipeline_annotation(text, tags)
    
    def create_alert_for_model_degradation(
        self,
        model_name: str,
        accuracy_threshold: float = 0.85
    ) -> bool:
        """
        Create alert rule for model performance degradation.
        
        Args:
            model_name (str): Name of the model to monitor
            accuracy_threshold (float): Accuracy threshold for alerts
            
        Returns:
            bool: True if alert was created successfully
        """
        alert_rule = {
            "alert": {
                "name": f"Model Degradation - {model_name}",
                "message": f"Model {model_name} accuracy has dropped below {accuracy_threshold}",
                "frequency": "1m",
                "conditions": [
                    {
                        "query": {
                            "queryType": "",
                            "refId": "A",
                            "model": {
                                "expr": f'mlflow_model_accuracy{{model_name="{model_name}",model_stage="production"}}',
                                "intervalMs": 1000,
                                "maxDataPoints": 43200
                            }
                        },
                        "reducer": {
                            "params": ["last"],
                            "type": "last"
                        },
                        "evaluator": {
                            "params": [accuracy_threshold],
                            "type": "lt"
                        }
                    }
                ],
                "executionErrorState": "alerting",
                "noDataState": "no_data",
                "for": "5m"
            }
        }
        
        try:
            response = self.session.post(
                f"{self.grafana_url}/api/alerts",
                json=alert_rule,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code in [200, 201]:
                logger.info(f"Alert rule created for model {model_name}")
                return True
            else:
                logger.error(f"Failed to create alert rule: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating alert rule: {e}")
            return False
    
    def get_dashboard_by_uid(self, dashboard_uid: str) -> Optional[Dict[str, Any]]:
        """
        Get dashboard by UID.
        
        Args:
            dashboard_uid (str): Dashboard UID
            
        Returns:
            Optional[Dict[str, Any]]: Dashboard data or None if not found
        """
        try:
            response = self.session.get(f"{self.grafana_url}/api/dashboards/uid/{dashboard_uid}")
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return None
            else:
                raise GrafanaAPIError(f"Error fetching dashboard: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error getting dashboard {dashboard_uid}: {e}")
            return None
    
    def send_pipeline_start_annotation(self, pipeline_name: str, run_id: str) -> bool:
        """
        Send annotation for pipeline start.
        
        Args:
            pipeline_name (str): Name of the pipeline
            run_id (str): Pipeline run identifier
            
        Returns:
            bool: True if annotation was sent successfully
        """
        text = f"🚀 Pipeline Started: {pipeline_name}"
        if run_id:
            text += f"\nRun ID: {run_id}"
        
        return self.send_pipeline_annotation(
            text,
            tags=["pipeline-start", pipeline_name.lower(), run_id]
        )
    
    def send_pipeline_completion_annotation(
        self,
        pipeline_name: str,
        run_id: str,
        success: bool,
        duration: Optional[float] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Send annotation for pipeline completion.
        
        Args:
            pipeline_name (str): Name of the pipeline
            run_id (str): Pipeline run identifier
            success (bool): Whether pipeline completed successfully
            duration (Optional[float]): Pipeline duration in seconds
            error_message (Optional[str]): Error message if failed
            
        Returns:
            bool: True if annotation was sent successfully
        """
        status_emoji = "✅" if success else "❌"
        status_text = "Completed Successfully" if success else "Failed"
        
        text = f"{status_emoji} Pipeline {status_text}: {pipeline_name}"
        if run_id:
            text += f"\nRun ID: {run_id}"
        if duration:
            text += f"\nDuration: {duration:.1f}s"
        if error_message:
            text += f"\nError: {error_message}"
        
        tags = [
            "pipeline-completion",
            pipeline_name.lower(),
            run_id,
            "success" if success else "failure"
        ]
        
        return self.send_pipeline_annotation(text, tags)


class DashboardManager:
    """
    Manager for Grafana dashboard operations.
    
    Handles dashboard creation, updates, and management for MLOps monitoring.
    """
    
    def __init__(self, grafana_manager: GrafanaManager):
        """
        Initialize dashboard manager.
        
        Args:
            grafana_manager (GrafanaManager): Grafana manager instance
        """
        self.grafana = grafana_manager
        self.session = grafana_manager.session
        self.grafana_url = grafana_manager.grafana_url
    
    def create_model_specific_dashboard(
        self,
        model_name: str,
        experiment_name: str
    ) -> Optional[str]:
        """
        Create a dashboard specific to a model.
        
        Args:
            model_name (str): Name of the model
            experiment_name (str): Name of the experiment
            
        Returns:
            Optional[str]: Dashboard UID if created successfully
        """
        dashboard_data = {
            "dashboard": {
                "id": None,
                "uid": f"model-{model_name.lower().replace('_', '-')}",
                "title": f"Model Dashboard - {model_name}",
                "description": f"Performance monitoring for {model_name} model",
                "tags": ["mlops", "model", model_name.lower()],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "Model Accuracy",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": f'mlflow_model_accuracy{{model_name="{model_name}"}}',
                                "legendFormat": "{{model_stage}}"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "Model Performance Metrics",
                        "type": "timeseries",
                        "targets": [
                            {
                                "expr": f'mlflow_model_accuracy{{model_name="{model_name}"}}',
                                "legendFormat": "Accuracy"
                            },
                            {
                                "expr": f'mlflow_model_precision{{model_name="{model_name}"}}',
                                "legendFormat": "Precision"
                            },
                            {
                                "expr": f'mlflow_model_recall{{model_name="{model_name}"}}',
                                "legendFormat": "Recall"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    }
                ],
                "time": {"from": "now-24h", "to": "now"},
                "refresh": "30s"
            },
            "folderId": 0,
            "overwrite": True
        }
        
        try:
            response = self.session.post(
                f"{self.grafana_url}/api/dashboards/db",
                json=dashboard_data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                dashboard_uid = result.get('uid')
                logger.info(f"Dashboard created for model {model_name}: {dashboard_uid}")
                return dashboard_uid
            else:
                logger.error(f"Failed to create dashboard: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            return None


def send_grafana_annotation_task(**context: Any) -> bool:
    """
    Airflow task to send Grafana annotation.
    
    Args:
        **context: Airflow context dictionary
        
    Returns:
        bool: True if annotation was sent successfully
    """
    from iris_config import get_config
    
    try:
        config = get_config()
        grafana = GrafanaManager(config)
        
        # Get task information from context
        task_id = context['task_instance'].task_id
        dag_id = context['dag'].dag_id
        execution_date = context['execution_date']
        
        # Send annotation
        text = f"Airflow Task: {dag_id}.{task_id}"
        tags = ["airflow", dag_id, task_id]
        
        return grafana.send_pipeline_annotation(
            text=text,
            tags=tags,
            time_start=execution_date
        )
        
    except Exception as e:
        logger.error(f"Failed to send Grafana annotation: {e}")
        return False


def send_model_performance_annotation_task(**context: Any) -> bool:
    """
    Airflow task to send model performance annotation to Grafana.
    
    Args:
        **context: Airflow context dictionary
        
    Returns:
        bool: True if annotation was sent successfully
    """
    from iris_config import get_config
    
    try:
        config = get_config()
        grafana = GrafanaManager(config)
        
        # Get model analysis results from XCom
        analysis_dict = context['task_instance'].xcom_pull(key='analysis_results')
        promotion_dict = context['task_instance'].xcom_pull(key='promotion_decision')
        
        if not analysis_dict:
            logger.warning("No analysis results found in XCom")
            return False
        
        # Reconstruct objects
        from iris_ml_operations import ModelAnalysisResults, PromotionDecision
        analysis_results = ModelAnalysisResults(**analysis_dict)
        
        promotion_decision = None
        if promotion_dict:
            promotion_decision = PromotionDecision(**promotion_dict)
        
        # Send annotation
        return grafana.send_model_performance_annotation(
            analysis_results=analysis_results,
            promotion_decision=promotion_decision
        )
        
    except Exception as e:
        logger.error(f"Failed to send model performance annotation: {e}")
        return False


def send_pipeline_status_annotation_task(**context: Any) -> bool:
    """
    Airflow task to send pipeline status annotation to Grafana.
    
    Args:
        **context: Airflow context dictionary
        
    Returns:
        bool: True if annotation was sent successfully
    """
    from iris_config import get_config
    
    try:
        config = get_config()
        grafana = GrafanaManager(config)
        
        # Get pipeline information from context
        dag_id = context['dag'].dag_id
        execution_date = context['execution_date']
        run_id = context['run_id']
        
        # Check if pipeline was successful (no failed tasks)
        dag_run = context['dag_run']
        task_instances = dag_run.get_task_instances()
        failed_tasks = [ti for ti in task_instances if ti.state == 'failed']
        
        success = len(failed_tasks) == 0
        
        # Calculate duration if available
        duration = None
        if dag_run.end_date and dag_run.start_date:
            duration = (dag_run.end_date - dag_run.start_date).total_seconds()
        
        # Get error message if failed
        error_message = None
        if failed_tasks:
            error_message = f"Failed tasks: {', '.join([ti.task_id for ti in failed_tasks])}"
        
        # Send completion annotation
        return grafana.send_pipeline_completion_annotation(
            pipeline_name=dag_id,
            run_id=run_id,
            success=success,
            duration=duration,
            error_message=error_message
        )
        
    except Exception as e:
        logger.error(f"Failed to send pipeline status annotation: {e}")
        return False

