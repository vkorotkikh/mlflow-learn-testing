"""
Machine Learning Operations for Iris Classification Pipeline

This module contains all ML-specific operations including MLflow integration,
model analysis, promotion decisions, and pipeline execution monitoring.

Classes:
    IrisMLOpsManager: Main class for managing ML operations
    ModelAnalysisResults: Data class for model analysis results
    PromotionDecision: Data class for model promotion decisions

Functions:
    Health check operations
    Pipeline execution and monitoring
    Model analysis and promotion
    MLflow integration utilities

Example:
    >>> from iris_ml_operations import IrisMLOpsManager
    >>> from iris_config import get_config
    >>> 
    >>> config = get_config()
    >>> ml_ops = IrisMLOpsManager(config)
    >>> results = ml_ops.analyze_pipeline_results(experiment_name="test_exp")
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import json
import logging
import time
import subprocess
import requests
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Run
import pandas as pd

from iris_config import IrisMLOpsConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelAnalysisResults:
    """
    Results from model performance analysis.
    
    Attributes:
        run_id (str): MLflow run identifier
        status (str): Run execution status
        accuracy (float): Model accuracy score (0.0-1.0)
        precision (float): Model precision score (0.0-1.0)
        recall (float): Model recall score (0.0-1.0)
        f1_score (float): Model F1-score (0.0-1.0)
        model_type (str): Type/algorithm of the model
        experiment_name (str): Name of the MLflow experiment
        meets_quality_thresholds (bool): Whether model meets quality criteria
        analysis_timestamp (str): When analysis was performed
        additional_metrics (Dict[str, float]): Any additional metrics
    """
    run_id: str
    status: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    model_type: str
    experiment_name: str
    meets_quality_thresholds: bool
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    additional_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PromotionDecision:
    """
    Decision data for model promotion to production.
    
    Attributes:
        should_promote (bool): Whether model should be promoted
        criteria (Dict[str, bool]): Individual promotion criteria results
        model_metrics (ModelAnalysisResults): Model performance metrics
        decision_timestamp (str): When decision was made
        promotion_reason (str): Explanation for promotion decision
        risk_assessment (str): Risk level assessment (low/medium/high)
    """
    should_promote: bool
    criteria: Dict[str, bool]
    model_metrics: ModelAnalysisResults
    decision_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    promotion_reason: str = ""
    risk_assessment: str = "medium"


class HealthCheckError(Exception):
    """Exception raised when health checks fail."""
    pass


class PipelineExecutionError(Exception):
    """Exception raised when pipeline execution fails."""
    pass


class ModelAnalysisError(Exception):
    """Exception raised when model analysis fails."""
    pass


class IrisMLOpsManager:
    """
    Main manager class for Iris ML Operations.
    
    This class provides a high-level interface for all ML operations in the
    Iris classification pipeline, including health checks, pipeline execution,
    model analysis, and promotion decisions.
    
    Attributes:
        config (IrisMLOpsConfig): Configuration object
        mlflow_client (MlflowClient): MLflow tracking client
        
    Example:
        >>> from iris_config import get_config
        >>> config = get_config()
        >>> ml_ops = IrisMLOpsManager(config)
        >>> ml_ops.check_services_health()
    """
    
    def __init__(self, config: IrisMLOpsConfig):
        """
        Initialize the ML Operations Manager.
        
        Args:
            config (IrisMLOpsConfig): Complete configuration object
        """
        self.config = config
        self.mlflow_client = None
        self._setup_mlflow()
    
    def _setup_mlflow(self) -> None:
        """
        Setup MLflow tracking client with configured URI.
        
        Raises:
            ConnectionError: If MLflow server is not accessible
        """
        try:
            mlflow.set_tracking_uri(self.config.infrastructure.mlflow_tracking_uri)
            self.mlflow_client = MlflowClient(self.config.infrastructure.mlflow_tracking_uri)
            logger.info(f"MLflow client initialized with URI: {self.config.infrastructure.mlflow_tracking_uri}")
        except Exception as e:
            logger.error(f"Failed to setup MLflow client: {e}")
            raise ConnectionError(f"Cannot connect to MLflow server: {e}")
    
    def check_mlflow_health(self) -> bool:
        """
        Check if MLflow tracking server is healthy and accessible.
        
        Performs health check with configurable retries and timeout.
        Verifies both server availability and basic API functionality.
        
        Returns:
            bool: True if MLflow server is healthy
            
        Raises:
            HealthCheckError: If health check fails after all retries
            
        Example:
            >>> ml_ops = IrisMLOpsManager(config)
            >>> is_healthy = ml_ops.check_mlflow_health()
            >>> print(f"MLflow healthy: {is_healthy}")
        """
        logger.info("Performing MLflow health check")
        
        max_retries = self.config.infrastructure.max_health_retries
        retry_delay = self.config.infrastructure.health_retry_delay
        timeout = self.config.infrastructure.mlflow_health_timeout
        
        for attempt in range(max_retries):
            try:
                # Check basic health endpoint
                response = requests.get(
                    f"{self.config.infrastructure.mlflow_tracking_uri}/health",
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    logger.info("MLflow server health check passed")
                    
                    # Additional check: try to list experiments
                    try:
                        experiments = self.mlflow_client.list_experiments()
                        logger.info(f"MLflow API check passed - {len(experiments)} experiments found")
                        return True
                    except Exception as api_error:
                        logger.warning(f"MLflow API check failed: {api_error}")
                        # Server is up but API might have issues
                        if attempt == max_retries - 1:
                            raise HealthCheckError(f"MLflow API not responding: {api_error}")
                else:
                    logger.warning(f"MLflow health check failed with status {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"MLflow health check attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                logger.info(f"Retrying health check in {retry_delay} seconds...")
                time.sleep(retry_delay)
        
        raise HealthCheckError("MLflow server health check failed after all retries")
    
    def check_kubeflow_health(self) -> bool:
        """
        Check if Kubeflow Pipelines is healthy and accessible.
        
        Verifies Kubeflow Pipelines API availability and basic functionality.
        
        Returns:
            bool: True if Kubeflow Pipelines is healthy
            
        Raises:
            HealthCheckError: If health check fails after all retries
            
        Example:
            >>> ml_ops = IrisMLOpsManager(config)
            >>> is_healthy = ml_ops.check_kubeflow_health()
            >>> print(f"Kubeflow healthy: {is_healthy}")
        """
        logger.info("Performing Kubeflow Pipelines health check")
        
        max_retries = self.config.infrastructure.max_health_retries
        retry_delay = self.config.infrastructure.health_retry_delay
        timeout = self.config.infrastructure.kubeflow_health_timeout
        
        for attempt in range(max_retries):
            try:
                # Check Kubeflow Pipelines API health
                response = requests.get(
                    f"{self.config.infrastructure.kubeflow_endpoint}/apis/v1beta1/healthz",
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    logger.info("Kubeflow Pipelines health check passed")
                    return True
                else:
                    logger.warning(f"Kubeflow health check failed with status {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Kubeflow health check attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                logger.info(f"Retrying health check in {retry_delay} seconds...")
                time.sleep(retry_delay)
        
        raise HealthCheckError("Kubeflow Pipelines health check failed after all retries")
    
    def check_services_health(self) -> Dict[str, bool]:
        """
        Check health of all required services.
        
        Returns:
            Dict[str, bool]: Health status of each service
            
        Example:
            >>> health = ml_ops.check_services_health()
            >>> print(health)
            {'mlflow': True, 'kubeflow': True}
        """
        health_status = {}
        
        try:
            health_status['mlflow'] = self.check_mlflow_health()
        except HealthCheckError:
            health_status['mlflow'] = False
        
        try:
            health_status['kubeflow'] = self.check_kubeflow_health()
        except HealthCheckError:
            health_status['kubeflow'] = False
        
        logger.info(f"Service health status: {health_status}")
        return health_status
    
    def prepare_pipeline_config(self, execution_date: datetime) -> Tuple[str, Dict[str, Any]]:
        """
        Prepare configuration for Kubeflow pipeline execution.
        
        Creates a unique configuration for each pipeline run with timestamps
        and experiment naming conventions.
        
        Args:
            execution_date (datetime): Pipeline execution timestamp
            
        Returns:
            Tuple[str, Dict[str, Any]]: Config file path and configuration dict
            
        Example:
            >>> config_path, config_dict = ml_ops.prepare_pipeline_config(datetime.now())
            >>> print(config_dict['experiment_name'])
            iris_classification_production_20240115_143022
        """
        logger.info("Preparing pipeline configuration")
        
        execution_str = execution_date.strftime('%Y%m%d_%H%M%S')
        
        pipeline_config = {
            "mlflow_tracking_uri": self.config.infrastructure.mlflow_tracking_uri,
            "experiment_name": f"{self.config.pipeline.experiment_name_prefix}_{execution_str}",
            "model_name": f"{self.config.pipeline.model_name_prefix}_{execution_str}",
            "accuracy_threshold": self.config.quality_thresholds.accuracy_threshold,
            "precision_threshold": self.config.quality_thresholds.precision_threshold,
            "recall_threshold": self.config.quality_thresholds.recall_threshold,
            "f1_threshold": self.config.quality_thresholds.f1_threshold,
            "perform_hyperparameter_tuning": self.config.pipeline.perform_hyperparameter_tuning,
            "deploy_if_valid": self.config.pipeline.deploy_if_valid,
            "execution_timestamp": execution_date.isoformat()
        }
        
        # Save configuration file
        config_path = f"{self.config.pipeline.config_temp_dir}/pipeline_config_{execution_str}.json"
        with open(config_path, 'w') as f:
            json.dump(pipeline_config, f, indent=2)
        
        logger.info(f"Pipeline configuration saved to: {config_path}")
        logger.debug(f"Configuration: {pipeline_config}")
        
        return config_path, pipeline_config
    
    def submit_kubeflow_pipeline(self, config_path: str, execution_date: datetime) -> str:
        """
        Submit pipeline to Kubeflow Pipelines for execution.
        
        Args:
            config_path (str): Path to pipeline configuration file
            execution_date (datetime): Pipeline execution timestamp
            
        Returns:
            str: Kubeflow run ID
            
        Raises:
            PipelineExecutionError: If pipeline submission fails
            
        Example:
            >>> run_id = ml_ops.submit_kubeflow_pipeline("/tmp/config.json", datetime.now())
            >>> print(f"Pipeline submitted with run ID: {run_id}")
        """
        logger.info("Submitting Kubeflow pipeline")
        
        execution_str = execution_date.strftime('%Y%m%d_%H%M%S')
        run_name = f"iris_mlops_run_{execution_str}"
        
        # Command to submit the pipeline
        cmd = [
            "python", self.config.pipeline.pipeline_script_path,
            "--action", "submit",
            "--pipeline", "mlops",
            "--run-name", run_name,
            "--config-file", config_path,
            "--kubeflow-endpoint", self.config.infrastructure.kubeflow_endpoint
        ]
        
        try:
            logger.info(f"Executing command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"Pipeline submitted successfully: {result.stdout}")
                
                # Extract run ID from output
                run_id = self._extract_run_id_from_output(result.stdout)
                if run_id:
                    logger.info(f"Extracted run ID: {run_id}")
                    return run_id
                else:
                    logger.warning("Could not extract run ID from output")
                    return run_name  # Fallback to run name
                    
            else:
                logger.error(f"Pipeline submission failed: {result.stderr}")
                raise PipelineExecutionError(f"Pipeline submission failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise PipelineExecutionError("Pipeline submission timed out")
        except Exception as e:
            logger.error(f"Error submitting pipeline: {e}")
            raise PipelineExecutionError(f"Pipeline submission error: {e}")
    
    def _extract_run_id_from_output(self, output: str) -> Optional[str]:
        """
        Extract run ID from pipeline submission output.
        
        Args:
            output (str): Command output text
            
        Returns:
            Optional[str]: Extracted run ID or None
        """
        for line in output.split('\n'):
            if 'Run ID:' in line:
                return line.split('Run ID:')[1].strip()
        return None
    
    def monitor_pipeline_execution(self, run_id: str) -> str:
        """
        Monitor Kubeflow pipeline execution until completion.
        
        Polls pipeline status at regular intervals until completion or timeout.
        
        Args:
            run_id (str): Kubeflow run identifier
            
        Returns:
            str: Final pipeline status ('success' or 'failed')
            
        Raises:
            PipelineExecutionError: If pipeline fails or times out
            
        Example:
            >>> status = ml_ops.monitor_pipeline_execution("run_12345")
            >>> print(f"Pipeline completed with status: {status}")
        """
        logger.info(f"Monitoring pipeline execution: {run_id}")
        
        max_wait_time = self.config.pipeline.max_pipeline_wait_time
        check_interval = self.config.pipeline.pipeline_check_interval
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            try:
                status = self._get_pipeline_status(run_id)
                logger.info(f"Pipeline status: {status}")
                
                if status in ['Succeeded', 'Completed']:
                    logger.info("Pipeline completed successfully")
                    return 'success'
                elif status in ['Failed', 'Error']:
                    logger.error("Pipeline failed")
                    raise PipelineExecutionError(f"Pipeline failed with status: {status}")
                
                # Still running, wait and check again
                time.sleep(check_interval)
                elapsed_time += check_interval
                
            except Exception as e:
                logger.warning(f"Error checking pipeline status: {e}")
                time.sleep(check_interval)
                elapsed_time += check_interval
        
        # Timeout reached
        raise PipelineExecutionError("Pipeline monitoring timed out")
    
    def _get_pipeline_status(self, run_id: str) -> str:
        """
        Get current status of pipeline run.
        
        Args:
            run_id (str): Pipeline run identifier
            
        Returns:
            str: Current pipeline status
        """
        response = requests.get(
            f"{self.config.infrastructure.kubeflow_endpoint}/apis/v1beta1/runs/{run_id}",
            timeout=30
        )
        
        if response.status_code == 200:
            run_info = response.json()
            return run_info.get('status', 'Unknown')
        else:
            raise Exception(f"Failed to get run status: {response.status_code}")
    
    def analyze_pipeline_results(self, experiment_name: str) -> ModelAnalysisResults:
        """
        Analyze results from completed pipeline execution.
        
        Retrieves and analyzes model performance metrics from MLflow experiment.
        Determines if model meets quality thresholds for potential promotion.
        
        Args:
            experiment_name (str): Name of MLflow experiment to analyze
            
        Returns:
            ModelAnalysisResults: Comprehensive analysis of model performance
            
        Raises:
            ModelAnalysisError: If analysis fails or no results found
            
        Example:
            >>> results = ml_ops.analyze_pipeline_results("iris_experiment_20240115")
            >>> print(f"Model accuracy: {results.accuracy:.3f}")
            >>> print(f"Meets thresholds: {results.meets_quality_thresholds}")
        """
        logger.info(f"Analyzing pipeline results for experiment: {experiment_name}")
        
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                raise ModelAnalysisError(f"Experiment {experiment_name} not found")
            
            # Get runs from the experiment
            runs = self.mlflow_client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=10,
                order_by=["start_time DESC"]
            )
            
            if not runs:
                raise ModelAnalysisError("No runs found in the experiment")
            
            # Analyze the latest run
            latest_run = runs[0]
            return self._analyze_run_metrics(latest_run, experiment_name)
            
        except Exception as e:
            logger.error(f"Error analyzing pipeline results: {e}")
            raise ModelAnalysisError(f"Pipeline analysis failed: {e}")
    
    def _analyze_run_metrics(self, run: Run, experiment_name: str) -> ModelAnalysisResults:
        """
        Analyze metrics from a specific MLflow run.
        
        Args:
            run (Run): MLflow run object
            experiment_name (str): Name of the experiment
            
        Returns:
            ModelAnalysisResults: Analysis results
        """
        metrics = run.data.metrics
        params = run.data.params
        
        # Extract key metrics with defaults
        accuracy = metrics.get('test_accuracy', metrics.get('accuracy', 0.0))
        precision = metrics.get('evaluation_precision', metrics.get('precision', 0.0))
        recall = metrics.get('evaluation_recall', metrics.get('recall', 0.0))
        f1_score = metrics.get('evaluation_f1', metrics.get('f1_score', 0.0))
        
        # Check quality thresholds
        thresholds = self.config.quality_thresholds
        meets_thresholds = (
            accuracy >= thresholds.accuracy_threshold and
            precision >= thresholds.precision_threshold and
            recall >= thresholds.recall_threshold and
            f1_score >= thresholds.f1_threshold
        )
        
        # Collect additional metrics
        additional_metrics = {
            k: v for k, v in metrics.items() 
            if k not in ['test_accuracy', 'accuracy', 'evaluation_precision', 
                        'precision', 'evaluation_recall', 'recall', 
                        'evaluation_f1', 'f1_score']
        }
        
        return ModelAnalysisResults(
            run_id=run.info.run_id,
            status=run.info.status,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            model_type=params.get('model_type', 'Unknown'),
            experiment_name=experiment_name,
            meets_quality_thresholds=meets_thresholds,
            additional_metrics=additional_metrics
        )
    
    def decide_model_promotion(self, analysis_results: ModelAnalysisResults) -> PromotionDecision:
        """
        Make decision about promoting model to production.
        
        Evaluates model against production criteria and makes promotion decision.
        
        Args:
            analysis_results (ModelAnalysisResults): Model performance analysis
            
        Returns:
            PromotionDecision: Detailed promotion decision with reasoning
            
        Example:
            >>> decision = ml_ops.decide_model_promotion(analysis_results)
            >>> if decision.should_promote:
            >>>     print(f"Promote model: {decision.promotion_reason}")
        """
        logger.info("Making model promotion decision")
        
        # Define promotion criteria
        criteria = {
            'meets_quality_thresholds': analysis_results.meets_quality_thresholds,
            'high_accuracy': analysis_results.accuracy >= self.config.quality_thresholds.production_accuracy_threshold,
            'balanced_metrics': (
                analysis_results.precision >= 0.85 and 
                analysis_results.recall >= 0.80 and 
                analysis_results.f1_score >= 0.82
            ),
            'model_completed': analysis_results.status == 'FINISHED'
        }
        
        should_promote = all(criteria.values())
        
        # Generate promotion reason
        if should_promote:
            promotion_reason = (
                f"Model meets all production criteria: "
                f"accuracy={analysis_results.accuracy:.3f}, "
                f"precision={analysis_results.precision:.3f}, "
                f"recall={analysis_results.recall:.3f}, "
                f"f1={analysis_results.f1_score:.3f}"
            )
            risk_assessment = "low"
        else:
            failed_criteria = [k for k, v in criteria.items() if not v]
            promotion_reason = f"Model failed criteria: {', '.join(failed_criteria)}"
            risk_assessment = "high"
        
        decision = PromotionDecision(
            should_promote=should_promote,
            criteria=criteria,
            model_metrics=analysis_results,
            promotion_reason=promotion_reason,
            risk_assessment=risk_assessment
        )
        
        logger.info(f"Promotion decision: {decision.should_promote} - {promotion_reason}")
        return decision
    
    def promote_model_to_production(self, model_name: str, promotion_decision: PromotionDecision) -> Optional[str]:
        """
        Promote model to production stage in MLflow Model Registry.
        
        Args:
            model_name (str): Name of the registered model
            promotion_decision (PromotionDecision): Promotion decision details
            
        Returns:
            Optional[str]: Version number of promoted model, None if not promoted
            
        Raises:
            ModelAnalysisError: If promotion fails
            
        Example:
            >>> version = ml_ops.promote_model_to_production("iris_model", decision)
            >>> if version:
            >>>     print(f"Model version {version} promoted to production")
        """
        if not promotion_decision.should_promote:
            logger.info("Model does not meet promotion criteria, skipping promotion")
            return None
        
        logger.info(f"Promoting model {model_name} to production")
        
        try:
            # Get the latest version in staging
            latest_versions = self.mlflow_client.get_latest_versions(model_name, stages=["Staging"])
            
            if not latest_versions:
                raise ModelAnalysisError(f"No staging version found for model {model_name}")
            
            model_version = latest_versions[0]
            
            # Promote to production
            self.mlflow_client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Production",
                archive_existing_versions=True
            )
            
            # Add promotion metadata
            self._add_promotion_tags(model_name, model_version.version, promotion_decision)
            
            logger.info(f"Model {model_name} version {model_version.version} promoted to Production")
            return model_version.version
            
        except Exception as e:
            logger.error(f"Error promoting model: {e}")
            raise ModelAnalysisError(f"Model promotion failed: {e}")
    
    def _add_promotion_tags(self, model_name: str, version: str, decision: PromotionDecision) -> None:
        """Add tags to promoted model version."""
        tags = {
            "promoted_by": "airflow_pipeline",
            "promotion_date": decision.decision_timestamp,
            "promotion_reason": decision.promotion_reason,
            "risk_assessment": decision.risk_assessment,
            "accuracy": str(decision.model_metrics.accuracy),
            "precision": str(decision.model_metrics.precision),
            "recall": str(decision.model_metrics.recall),
            "f1_score": str(decision.model_metrics.f1_score)
        }
        
        for key, value in tags.items():
            self.mlflow_client.set_model_version_tag(
                name=model_name,
                version=version,
                key=key,
                value=value
            )
