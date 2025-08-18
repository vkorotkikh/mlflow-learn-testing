"""
MLOps Utilities for Airflow
This module provides utility functions and classes for MLOps operations in Airflow.
"""

import logging
import json
import requests
from datetime import datetime
from typing import Dict, Any, List, Optional
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

logger = logging.getLogger(__name__)

class MLflowUtils:
    """Utility class for MLflow operations in Airflow"""
    
    def __init__(self, tracking_uri: str):
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri)
    
    def get_experiment_metrics(self, experiment_name: str, max_runs: int = 10) -> Dict[str, Any]:
        """Get metrics summary for an experiment"""
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                return {"error": f"Experiment {experiment_name} not found"}
            
            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=max_runs,
                order_by=["start_time DESC"]
            )
            
            if not runs:
                return {"error": "No runs found"}
            
            # Calculate summary statistics
            metrics_data = []
            for run in runs:
                metrics_data.append(run.data.metrics)
            
            df = pd.DataFrame(metrics_data)
            
            summary = {
                "experiment_name": experiment_name,
                "total_runs": len(runs),
                "latest_run_id": runs[0].info.run_id,
                "metrics_summary": df.describe().to_dict() if not df.empty else {}
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting experiment metrics: {e}")
            return {"error": str(e)}
    
    def get_model_performance_trend(self, model_name: str, metric_name: str = "test_accuracy") -> List[Dict]:
        """Get performance trend for a registered model"""
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            trend_data = []
            for version in versions:
                run = self.client.get_run(version.run_id)
                metric_value = run.data.metrics.get(metric_name)
                
                if metric_value is not None:
                    trend_data.append({
                        "version": version.version,
                        "stage": version.current_stage,
                        "metric_value": metric_value,
                        "creation_time": version.creation_timestamp,
                        "run_id": version.run_id
                    })
            
            # Sort by creation time
            trend_data.sort(key=lambda x: x["creation_time"])
            
            return trend_data
            
        except Exception as e:
            logger.error(f"Error getting model performance trend: {e}")
            return []
    
    def check_model_degradation(self, model_name: str, threshold_drop: float = 0.05) -> Dict[str, Any]:
        """Check if there's model performance degradation"""
        trend_data = self.get_model_performance_trend(model_name)
        
        if len(trend_data) < 2:
            return {"degradation_detected": False, "reason": "Insufficient data"}
        
        # Compare latest with previous
        latest = trend_data[-1]
        previous = trend_data[-2]
        
        performance_drop = previous["metric_value"] - latest["metric_value"]
        degradation_detected = performance_drop > threshold_drop
        
        return {
            "degradation_detected": degradation_detected,
            "performance_drop": performance_drop,
            "threshold": threshold_drop,
            "latest_performance": latest["metric_value"],
            "previous_performance": previous["metric_value"],
            "latest_version": latest["version"],
            "previous_version": previous["version"]
        }

class KubeflowUtils:
    """Utility class for Kubeflow operations in Airflow"""
    
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
    
    def get_pipeline_runs(self, max_results: int = 10) -> List[Dict]:
        """Get recent pipeline runs"""
        try:
            response = requests.get(
                f"{self.endpoint}/apis/v1beta1/runs",
                params={"page_size": max_results},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("runs", [])
            else:
                logger.error(f"Failed to get pipeline runs: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting pipeline runs: {e}")
            return []
    
    def get_pipeline_status(self, run_id: str) -> Dict[str, Any]:
        """Get status of a specific pipeline run"""
        try:
            response = requests.get(
                f"{self.endpoint}/apis/v1beta1/runs/{run_id}",
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get pipeline status: {response.status_code}")
                return {"error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error getting pipeline status: {e}")
            return {"error": str(e)}
    
    def wait_for_pipeline_completion(self, run_id: str, timeout: int = 3600, check_interval: int = 60) -> str:
        """Wait for pipeline to complete and return final status"""
        import time
        
        elapsed_time = 0
        
        while elapsed_time < timeout:
            status_info = self.get_pipeline_status(run_id)
            
            if "error" in status_info:
                time.sleep(check_interval)
                elapsed_time += check_interval
                continue
            
            status = status_info.get("status", "Unknown")
            
            if status in ["Succeeded", "Completed"]:
                return "Succeeded"
            elif status in ["Failed", "Error"]:
                return "Failed"
            
            time.sleep(check_interval)
            elapsed_time += check_interval
        
        return "Timeout"

class ModelValidationUtils:
    """Utility class for model validation operations"""
    
    @staticmethod
    def validate_model_metrics(metrics: Dict[str, float], thresholds: Dict[str, float]) -> Dict[str, Any]:
        """Validate model metrics against thresholds"""
        validation_results = {
            "passed": True,
            "failed_checks": [],
            "metrics": metrics,
            "thresholds": thresholds
        }
        
        for metric_name, threshold in thresholds.items():
            if metric_name in metrics:
                if metrics[metric_name] < threshold:
                    validation_results["passed"] = False
                    validation_results["failed_checks"].append({
                        "metric": metric_name,
                        "value": metrics[metric_name],
                        "threshold": threshold,
                        "difference": threshold - metrics[metric_name]
                    })
        
        return validation_results
    
    @staticmethod
    def check_data_drift(current_stats: Dict, baseline_stats: Dict, 
                        drift_threshold: float = 0.1) -> Dict[str, Any]:
        """Simple data drift detection based on statistical differences"""
        drift_detected = False
        drift_details = []
        
        for feature, current_value in current_stats.items():
            if feature in baseline_stats:
                baseline_value = baseline_stats[feature]
                
                # Calculate relative difference
                if baseline_value != 0:
                    relative_diff = abs(current_value - baseline_value) / abs(baseline_value)
                    
                    if relative_diff > drift_threshold:
                        drift_detected = True
                        drift_details.append({
                            "feature": feature,
                            "current_value": current_value,
                            "baseline_value": baseline_value,
                            "relative_difference": relative_diff,
                            "threshold": drift_threshold
                        })
        
        return {
            "drift_detected": drift_detected,
            "drift_details": drift_details,
            "threshold": drift_threshold
        }

class NotificationUtils:
    """Utility class for sending notifications"""
    
    @staticmethod
    def format_pipeline_report(pipeline_results: Dict[str, Any], 
                             model_metrics: Dict[str, Any],
                             deployment_info: Dict[str, Any]) -> str:
        """Format a comprehensive pipeline report"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
MLOps Pipeline Execution Report
===============================
Generated: {timestamp}

Pipeline Results:
{'-' * 20}
Status: {pipeline_results.get('status', 'Unknown')}
Run ID: {pipeline_results.get('run_id', 'N/A')}
Duration: {pipeline_results.get('duration', 'N/A')}

Model Performance:
{'-' * 20}
"""
        
        for metric, value in model_metrics.items():
            if isinstance(value, float):
                report += f"{metric.title()}: {value:.4f}\n"
            else:
                report += f"{metric.title()}: {value}\n"
        
        report += f"""
Deployment Information:
{'-' * 20}
Model Promoted: {deployment_info.get('promoted', 'No')}
Production Version: {deployment_info.get('production_version', 'N/A')}
Deployment Stage: {deployment_info.get('stage', 'N/A')}

Next Steps:
{'-' * 20}
"""
        
        if deployment_info.get('promoted'):
            report += "✅ Model successfully promoted to production\n"
            report += "- Monitor production performance\n"
            report += "- Update deployment documentation\n"
        else:
            report += "❌ Model not promoted to production\n"
            report += "- Review model performance metrics\n"
            report += "- Consider additional training or tuning\n"
        
        return report
    
    @staticmethod
    def create_alert_message(alert_type: str, details: Dict[str, Any]) -> str:
        """Create alert message for various scenarios"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if alert_type == "model_degradation":
            return f"""
🚨 MODEL PERFORMANCE ALERT
{timestamp}

Model degradation detected:
- Model: {details.get('model_name', 'Unknown')}
- Performance drop: {details.get('performance_drop', 0):.4f}
- Current performance: {details.get('latest_performance', 0):.4f}
- Previous performance: {details.get('previous_performance', 0):.4f}

Action required: Review model and consider retraining.
"""
        
        elif alert_type == "pipeline_failure":
            return f"""
❌ PIPELINE FAILURE ALERT
{timestamp}

Pipeline execution failed:
- Pipeline: {details.get('pipeline_name', 'Unknown')}
- Error: {details.get('error_message', 'Unknown error')}
- Run ID: {details.get('run_id', 'N/A')}

Action required: Check logs and investigate failure cause.
"""
        
        elif alert_type == "data_drift":
            return f"""
⚠️ DATA DRIFT ALERT
{timestamp}

Data drift detected:
- Features affected: {len(details.get('drift_details', []))}
- Drift threshold: {details.get('threshold', 0)}

Action required: Review data sources and consider model retraining.
"""
        
        else:
            return f"""
ℹ️ SYSTEM ALERT
{timestamp}

Alert Type: {alert_type}
Details: {details}
"""

def create_airflow_variables_config():
    """Create a configuration file for Airflow variables"""
    
    variables_config = {
        "mlflow_tracking_uri": "http://localhost:5000",
        "kubeflow_endpoint": "http://localhost:8080", 
        "experiment_name": "iris_classification_production",
        "model_name": "iris_random_forest_production",
        "notification_email": "admin@company.com",
        "accuracy_threshold": "0.85",
        "precision_threshold": "0.85",
        "model_degradation_threshold": "0.05",
        "data_drift_threshold": "0.1",
        "pipeline_timeout": "3600",
        "monitoring_interval": "60"
    }
    
    return variables_config

def setup_airflow_connections():
    """Instructions for setting up Airflow connections"""
    
    connections_setup = """
# Airflow Connections Setup Commands

# MLflow Connection
airflow connections add 'mlflow_default' \\
    --conn-type 'http' \\
    --conn-host 'localhost' \\
    --conn-port 5000 \\
    --conn-schema 'http'

# Kubeflow Connection  
airflow connections add 'kubeflow_default' \\
    --conn-type 'http' \\
    --conn-host 'localhost' \\
    --conn-port 8080 \\
    --conn-schema 'http'

# Email Connection (for notifications)
airflow connections add 'email_default' \\
    --conn-type 'email' \\
    --conn-host 'smtp.gmail.com' \\
    --conn-port 587 \\
    --conn-login 'your-email@gmail.com' \\
    --conn-password 'your-app-password'
"""
    
    return connections_setup


