"""
Utility functions for Iris MLOps Pipeline

This module provides utility functions for notifications, reporting,
and common operations used throughout the Iris MLOps pipeline.

Functions:
    Notification utilities
    Report generation
    Data formatting and validation
    File management utilities

Example:
    >>> from iris_utils import NotificationManager
    >>> from iris_config import get_config
    >>> 
    >>> config = get_config()
    >>> notifier = NotificationManager(config)
    >>> notifier.send_pipeline_completion_notification(results)
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
import logging
import os
import shutil
from pathlib import Path

from iris_config import IrisMLOpsConfig
from iris_ml_operations import ModelAnalysisResults, PromotionDecision

logger = logging.getLogger(__name__)


class NotificationManager:
    """
    Manager for sending notifications about pipeline status and results.
    
    Handles email notifications, webhook integrations, and report generation
    for the MLOps pipeline execution results.
    
    Attributes:
        config (IrisMLOpsConfig): Configuration object
        
    Example:
        >>> notifier = NotificationManager(config)
        >>> notifier.send_success_notification(analysis_results, promotion_decision)
    """
    
    def __init__(self, config: IrisMLOpsConfig):
        """
        Initialize notification manager.
        
        Args:
            config (IrisMLOpsConfig): Complete configuration object
        """
        self.config = config
    
    def prepare_pipeline_notification(
        self,
        execution_date: datetime,
        analysis_results: Optional[ModelAnalysisResults] = None,
        promotion_decision: Optional[PromotionDecision] = None,
        promoted_version: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Prepare notification content for pipeline completion.
        
        Creates comprehensive notification content including performance metrics,
        promotion decisions, and execution details.
        
        Args:
            execution_date (datetime): Pipeline execution timestamp
            analysis_results (Optional[ModelAnalysisResults]): Model analysis results
            promotion_decision (Optional[PromotionDecision]): Promotion decision details
            promoted_version (Optional[str]): Version number if model was promoted
            error_message (Optional[str]): Error message if pipeline failed
            
        Returns:
            Dict[str, str]: Dictionary with 'subject' and 'content' keys
            
        Example:
            >>> notification = notifier.prepare_pipeline_notification(
            ...     datetime.now(), analysis_results, promotion_decision
            ... )
            >>> print(notification['subject'])
        """
        execution_str = execution_date.strftime('%Y-%m-%d %H:%M:%S')
        
        if error_message:
            return self._prepare_failure_notification(execution_str, error_message)
        elif analysis_results:
            return self._prepare_success_notification(
                execution_str, analysis_results, promotion_decision, promoted_version
            )
        else:
            return self._prepare_generic_notification(execution_str)
    
    def _prepare_success_notification(
        self,
        execution_str: str,
        analysis_results: ModelAnalysisResults,
        promotion_decision: Optional[PromotionDecision],
        promoted_version: Optional[str]
    ) -> Dict[str, str]:
        """Prepare notification for successful pipeline execution."""
        
        # Determine overall status
        promotion_status = "Not Evaluated"
        if promotion_decision:
            promotion_status = "✅ Promoted" if promotion_decision.should_promote else "❌ Not Promoted"
        
        subject = f"✅ MLOps Pipeline Completed Successfully - {execution_str}"
        
        # Create detailed content
        content = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #4CAF50; color: white; padding: 15px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #4CAF50; background-color: #f9f9f9; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .metric {{ background: white; padding: 10px; border-radius: 5px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
        .metric-label {{ color: #666; margin-top: 5px; }}
        .promotion-yes {{ color: #4CAF50; font-weight: bold; }}
        .promotion-no {{ color: #FF9800; font-weight: bold; }}
        .footer {{ margin-top: 30px; padding: 15px; background-color: #f0f0f0; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎉 MLOps Pipeline Execution Report</h1>
        <p>Execution completed successfully on {execution_str}</p>
    </div>
    
    <div class="section">
        <h2>📊 Model Performance Metrics</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{analysis_results.accuracy:.1%}</div>
                <div class="metric-label">Accuracy</div>
            </div>
            <div class="metric">
                <div class="metric-value">{analysis_results.precision:.1%}</div>
                <div class="metric-label">Precision</div>
            </div>
            <div class="metric">
                <div class="metric-value">{analysis_results.recall:.1%}</div>
                <div class="metric-label">Recall</div>
            </div>
            <div class="metric">
                <div class="metric-value">{analysis_results.f1_score:.1%}</div>
                <div class="metric-label">F1-Score</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>🔍 Quality Assessment</h2>
        <p><strong>Meets Quality Thresholds:</strong> {'✅ Yes' if analysis_results.meets_quality_thresholds else '❌ No'}</p>
        <p><strong>Model Type:</strong> {analysis_results.model_type}</p>
        <p><strong>Run Status:</strong> {analysis_results.status}</p>
    </div>
    
    <div class="section">
        <h2>🚀 Model Promotion</h2>
        <p><strong>Promotion Decision:</strong> <span class="{'promotion-yes' if promotion_decision and promotion_decision.should_promote else 'promotion-no'}">{promotion_status}</span></p>
"""
        
        if promotion_decision:
            content += f"""
        <p><strong>Promotion Reason:</strong> {promotion_decision.promotion_reason}</p>
        <p><strong>Risk Assessment:</strong> {promotion_decision.risk_assessment.title()}</p>
"""
        
        if promoted_version:
            content += f"""
        <p><strong>Promoted Version:</strong> {promoted_version}</p>
"""
        
        content += f"""
    </div>
    
    <div class="section">
        <h2>🔗 Experiment Details</h2>
        <p><strong>Experiment:</strong> {analysis_results.experiment_name}</p>
        <p><strong>Run ID:</strong> <code>{analysis_results.run_id}</code></p>
        <p><strong>Analysis Timestamp:</strong> {analysis_results.analysis_timestamp}</p>
    </div>
    
    <div class="footer">
        <p><strong>🔗 Links:</strong></p>
        <ul>
            <li><a href="{self.config.infrastructure.mlflow_tracking_uri}">MLflow Tracking UI</a></li>
            <li><a href="{self.config.infrastructure.kubeflow_endpoint}">Kubeflow Pipelines</a></li>
        </ul>
        <p><em>This report was generated automatically by the Iris MLOps Pipeline.</em></p>
    </div>
</body>
</html>
"""
        
        return {"subject": subject, "content": content}
    
    def _prepare_failure_notification(self, execution_str: str, error_message: str) -> Dict[str, str]:
        """Prepare notification for failed pipeline execution."""
        
        subject = f"❌ MLOps Pipeline Failed - {execution_str}"
        
        content = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f44336; color: white; padding: 15px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #f44336; background-color: #ffebee; }}
        .error {{ background-color: #ffcdd2; padding: 10px; border-radius: 5px; font-family: monospace; }}
        .footer {{ margin-top: 30px; padding: 15px; background-color: #f0f0f0; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>❌ MLOps Pipeline Execution Failed</h1>
        <p>Execution failed on {execution_str}</p>
    </div>
    
    <div class="section">
        <h2>🚨 Error Details</h2>
        <div class="error">
            {error_message}
        </div>
    </div>
    
    <div class="section">
        <h2>🔧 Troubleshooting Steps</h2>
        <ol>
            <li>Check Airflow task logs for detailed error information</li>
            <li>Verify MLflow and Kubeflow services are running</li>
            <li>Validate pipeline configuration and data sources</li>
            <li>Review system resource usage and availability</li>
        </ol>
    </div>
    
    <div class="footer">
        <p><strong>🔗 Links:</strong></p>
        <ul>
            <li><a href="{self.config.infrastructure.mlflow_tracking_uri}">MLflow Tracking UI</a></li>
            <li><a href="{self.config.infrastructure.kubeflow_endpoint}">Kubeflow Pipelines</a></li>
        </ul>
        <p><em>Please investigate and resolve the issue before the next scheduled run.</em></p>
    </div>
</body>
</html>
"""
        
        return {"subject": subject, "content": content}
    
    def _prepare_generic_notification(self, execution_str: str) -> Dict[str, str]:
        """Prepare generic notification when results are not available."""
        
        subject = f"ℹ️ MLOps Pipeline Status - {execution_str}"
        
        content = f"""
MLOps Pipeline Execution Status
===============================

Execution Date: {execution_str}

The pipeline has completed execution, but detailed results are not available.
Please check the Airflow logs and MLflow tracking UI for more information.

Links:
- MLflow Tracking: {self.config.infrastructure.mlflow_tracking_uri}
- Kubeflow Pipelines: {self.config.infrastructure.kubeflow_endpoint}
"""
        
        return {"subject": subject, "content": content}


class FileManager:
    """
    Manager for file operations and cleanup tasks.
    
    Handles temporary file management, configuration cleanup,
    and data file operations for the pipeline.
    """
    
    @staticmethod
    def cleanup_temp_files(temp_dir: str, pattern: str = "pipeline_config_*.json") -> None:
        """
        Clean up temporary files matching the given pattern.
        
        Args:
            temp_dir (str): Directory to clean up
            pattern (str): File pattern to match for deletion
            
        Example:
            >>> FileManager.cleanup_temp_files("/tmp", "pipeline_config_*.json")
        """
        logger.info(f"Cleaning up temporary files in {temp_dir} matching {pattern}")
        
        try:
            temp_path = Path(temp_dir)
            if temp_path.exists():
                # Remove files matching pattern
                for file_path in temp_path.glob(pattern):
                    file_path.unlink()
                    logger.debug(f"Removed file: {file_path}")
                
                logger.info("Temporary file cleanup completed")
            else:
                logger.warning(f"Temporary directory {temp_dir} does not exist")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    @staticmethod
    def ensure_directory_exists(directory_path: str) -> None:
        """
        Ensure a directory exists, creating it if necessary.
        
        Args:
            directory_path (str): Path to directory
        """
        try:
            Path(directory_path).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory_path}")
        except Exception as e:
            logger.error(f"Error creating directory {directory_path}: {e}")
            raise
    
    @staticmethod
    def backup_config_file(config_path: str, backup_dir: str) -> str:
        """
        Create a backup of a configuration file.
        
        Args:
            config_path (str): Path to configuration file
            backup_dir (str): Directory for backup
            
        Returns:
            str: Path to backup file
        """
        try:
            FileManager.ensure_directory_exists(backup_dir)
            
            config_file = Path(config_path)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"{config_file.stem}_{timestamp}{config_file.suffix}"
            backup_path = Path(backup_dir) / backup_name
            
            shutil.copy2(config_path, backup_path)
            logger.info(f"Config backed up to: {backup_path}")
            
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Error backing up config: {e}")
            raise


class MetricsFormatter:
    """
    Utility class for formatting and validating metrics data.
    
    Provides methods for formatting model metrics, validation results,
    and performance data for display and reporting.
    """
    
    @staticmethod
    def format_percentage(value: float, decimal_places: int = 1) -> str:
        """
        Format a decimal value as a percentage.
        
        Args:
            value (float): Decimal value (0.0-1.0)
            decimal_places (int): Number of decimal places
            
        Returns:
            str: Formatted percentage string
            
        Example:
            >>> MetricsFormatter.format_percentage(0.8567, 2)
            '85.67%'
        """
        return f"{value * 100:.{decimal_places}f}%"
    
    @staticmethod
    def format_metrics_summary(analysis_results: ModelAnalysisResults) -> Dict[str, str]:
        """
        Format model metrics for display.
        
        Args:
            analysis_results (ModelAnalysisResults): Model analysis results
            
        Returns:
            Dict[str, str]: Formatted metrics dictionary
        """
        return {
            "accuracy": MetricsFormatter.format_percentage(analysis_results.accuracy),
            "precision": MetricsFormatter.format_percentage(analysis_results.precision),
            "recall": MetricsFormatter.format_percentage(analysis_results.recall),
            "f1_score": MetricsFormatter.format_percentage(analysis_results.f1_score),
            "model_type": analysis_results.model_type,
            "status": analysis_results.status,
            "meets_thresholds": "✅ Yes" if analysis_results.meets_quality_thresholds else "❌ No"
        }
    
    @staticmethod
    def create_metrics_table(analysis_results: ModelAnalysisResults) -> str:
        """
        Create a formatted table of model metrics.
        
        Args:
            analysis_results (ModelAnalysisResults): Model analysis results
            
        Returns:
            str: Formatted table string
        """
        metrics = MetricsFormatter.format_metrics_summary(analysis_results)
        
        table = "Model Performance Metrics\n"
        table += "=" * 30 + "\n"
        
        for key, value in metrics.items():
            table += f"{key.replace('_', ' ').title():<15} : {value}\n"
        
        return table


class ValidationUtils:
    """
    Utility functions for data validation and verification.
    
    Provides validation methods for configuration, metrics,
    and pipeline data to ensure data quality and consistency.
    """
    
    @staticmethod
    def validate_metric_range(value: float, min_val: float = 0.0, max_val: float = 1.0) -> bool:
        """
        Validate that a metric value is within the expected range.
        
        Args:
            value (float): Metric value to validate
            min_val (float): Minimum acceptable value
            max_val (float): Maximum acceptable value
            
        Returns:
            bool: True if value is within range
        """
        return min_val <= value <= max_val
    
    @staticmethod
    def validate_analysis_results(analysis_results: ModelAnalysisResults) -> List[str]:
        """
        Validate model analysis results for completeness and correctness.
        
        Args:
            analysis_results (ModelAnalysisResults): Results to validate
            
        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required fields
        if not analysis_results.run_id:
            errors.append("Missing run_id")
        
        if not analysis_results.experiment_name:
            errors.append("Missing experiment_name")
        
        # Validate metric ranges
        metrics = [
            ("accuracy", analysis_results.accuracy),
            ("precision", analysis_results.precision),
            ("recall", analysis_results.recall),
            ("f1_score", analysis_results.f1_score)
        ]
        
        for metric_name, value in metrics:
            if not ValidationUtils.validate_metric_range(value):
                errors.append(f"Invalid {metric_name}: {value} (should be 0.0-1.0)")
        
        return errors
    
    @staticmethod
    def validate_promotion_decision(decision: PromotionDecision) -> List[str]:
        """
        Validate promotion decision for completeness.
        
        Args:
            decision (PromotionDecision): Promotion decision to validate
            
        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        errors = []
        
        if not decision.promotion_reason:
            errors.append("Missing promotion_reason")
        
        if not decision.criteria:
            errors.append("Missing promotion criteria")
        
        if decision.risk_assessment not in ["low", "medium", "high"]:
            errors.append(f"Invalid risk_assessment: {decision.risk_assessment}")
        
        return errors
