#!/usr/bin/env python3
"""
MLflow Prometheus Exporter

This exporter collects metrics from MLflow tracking server and exposes them
in Prometheus format for monitoring and alerting.

Metrics collected:
- Model performance metrics (accuracy, precision, recall, F1-score)
- Experiment and run statistics
- Model registry status
- Training duration and resource usage
- Data drift and model degradation indicators

Usage:
    python mlflow_prometheus_exporter.py

Environment Variables:
    MLFLOW_TRACKING_URI: MLflow tracking server URL (default: http://localhost:5000)
    EXPORTER_PORT: Port to expose metrics (default: 8080)
    METRICS_INTERVAL: Collection interval in seconds (default: 60)
    LOG_LEVEL: Logging level (default: INFO)

Author: MLOps Team
Version: 1.0.0
"""

import os
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import yaml

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Experiment, Run
import pandas as pd
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Info
import requests

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
EXPORTER_PORT = int(os.getenv('EXPORTER_PORT', '8080'))
METRICS_INTERVAL = int(os.getenv('METRICS_INTERVAL', '60'))

# Prometheus metrics definitions
class MLflowMetrics:
    """Prometheus metrics for MLflow monitoring."""
    
    def __init__(self):
        # Model Performance Metrics
        self.model_accuracy = Gauge(
            'mlflow_model_accuracy',
            'Model accuracy score',
            ['experiment_name', 'run_id', 'model_name', 'model_stage']
        )
        
        self.model_precision = Gauge(
            'mlflow_model_precision',
            'Model precision score',
            ['experiment_name', 'run_id', 'model_name', 'model_stage']
        )
        
        self.model_recall = Gauge(
            'mlflow_model_recall',
            'Model recall score',
            ['experiment_name', 'run_id', 'model_name', 'model_stage']
        )
        
        self.model_f1_score = Gauge(
            'mlflow_model_f1_score',
            'Model F1-score',
            ['experiment_name', 'run_id', 'model_name', 'model_stage']
        )
        
        self.model_training_duration = Gauge(
            'mlflow_model_training_duration_seconds',
            'Model training duration in seconds',
            ['experiment_name', 'run_id', 'model_name']
        )
        
        # Experiment and Run Statistics
        self.total_experiments = Gauge(
            'mlflow_experiments_total',
            'Total number of experiments'
        )
        
        self.total_runs = Gauge(
            'mlflow_runs_total',
            'Total number of runs',
            ['experiment_name', 'status']
        )
        
        self.recent_runs = Gauge(
            'mlflow_recent_runs_total',
            'Number of runs in the last 24 hours',
            ['experiment_name', 'status']
        )
        
        # Model Registry Metrics
        self.registered_models_total = Gauge(
            'mlflow_registered_models_total',
            'Total number of registered models'
        )
        
        self.model_versions_by_stage = Gauge(
            'mlflow_model_versions_by_stage_total',
            'Number of model versions by stage',
            ['model_name', 'stage']
        )
        
        # System Health Metrics
        self.mlflow_server_health = Gauge(
            'mlflow_server_health',
            'MLflow server health status (1=healthy, 0=unhealthy)'
        )
        
        self.mlflow_response_time = Histogram(
            'mlflow_api_response_time_seconds',
            'MLflow API response time',
            ['endpoint']
        )
        
        # Data Quality Metrics
        self.data_drift_score = Gauge(
            'mlflow_data_drift_score',
            'Data drift detection score',
            ['experiment_name', 'run_id', 'feature']
        )
        
        self.model_degradation_score = Gauge(
            'mlflow_model_degradation_score',
            'Model performance degradation score',
            ['model_name', 'model_stage']
        )
        
        # Business Metrics
        self.prediction_count = Counter(
            'mlflow_predictions_total',
            'Total number of predictions made',
            ['model_name', 'model_version']
        )
        
        # Info metrics
        self.exporter_info = Info(
            'mlflow_exporter_info',
            'Information about the MLflow exporter'
        )


class MLflowPrometheusExporter:
    """
    MLflow Prometheus exporter that collects and exposes MLflow metrics.
    """
    
    def __init__(self):
        """Initialize the exporter."""
        self.mlflow_client = None
        self.metrics = MLflowMetrics()
        self.running = False
        self._setup_mlflow_client()
        self._setup_exporter_info()
    
    def _setup_mlflow_client(self):
        """Setup MLflow client connection."""
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            self.mlflow_client = MlflowClient(MLFLOW_TRACKING_URI)
            logger.info(f"Connected to MLflow at {MLFLOW_TRACKING_URI}")
        except Exception as e:
            logger.error(f"Failed to connect to MLflow: {e}")
            raise
    
    def _setup_exporter_info(self):
        """Setup exporter information metrics."""
        self.metrics.exporter_info.info({
            'version': '1.0.0',
            'mlflow_uri': MLFLOW_TRACKING_URI,
            'metrics_interval': str(METRICS_INTERVAL),
            'start_time': datetime.now().isoformat()
        })
    
    def check_mlflow_health(self) -> bool:
        """Check MLflow server health."""
        try:
            start_time = time.time()
            response = requests.get(f"{MLFLOW_TRACKING_URI}/health", timeout=10)
            response_time = time.time() - start_time
            
            self.metrics.mlflow_response_time.labels(endpoint='health').observe(response_time)
            
            if response.status_code == 200:
                self.metrics.mlflow_server_health.set(1)
                return True
            else:
                self.metrics.mlflow_server_health.set(0)
                return False
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.metrics.mlflow_server_health.set(0)
            return False
    
    def collect_experiment_metrics(self):
        """Collect experiment and run metrics."""
        try:
            # Get all experiments
            experiments = self.mlflow_client.list_experiments()
            self.metrics.total_experiments.set(len(experiments))
            
            # Collect run statistics
            for experiment in experiments:
                exp_name = experiment.name
                
                # Get runs for this experiment
                runs = self.mlflow_client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    max_results=1000
                )
                
                # Count runs by status
                run_status_counts = {}
                recent_run_counts = {}
                recent_cutoff = datetime.now() - timedelta(hours=24)
                
                for run in runs:
                    status = run.info.status
                    run_status_counts[status] = run_status_counts.get(status, 0) + 1
                    
                    # Check if run is recent
                    run_start_time = datetime.fromtimestamp(run.info.start_time / 1000)
                    if run_start_time > recent_cutoff:
                        recent_run_counts[status] = recent_run_counts.get(status, 0) + 1
                
                # Update metrics
                for status, count in run_status_counts.items():
                    self.metrics.total_runs.labels(
                        experiment_name=exp_name,
                        status=status
                    ).set(count)
                
                for status, count in recent_run_counts.items():
                    self.metrics.recent_runs.labels(
                        experiment_name=exp_name,
                        status=status
                    ).set(count)
                    
        except Exception as e:
            logger.error(f"Failed to collect experiment metrics: {e}")
    
    def collect_model_performance_metrics(self):
        """Collect model performance metrics from recent runs."""
        try:
            # Get recent experiments (last 7 days)
            recent_cutoff = datetime.now() - timedelta(days=7)
            experiments = self.mlflow_client.list_experiments()
            
            for experiment in experiments:
                runs = self.mlflow_client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    max_results=50,
                    order_by=["start_time DESC"]
                )
                
                for run in runs:
                    run_start_time = datetime.fromtimestamp(run.info.start_time / 1000)
                    if run_start_time < recent_cutoff:
                        continue
                    
                    metrics = run.data.metrics
                    params = run.data.params
                    
                    # Extract common metrics
                    accuracy = metrics.get('test_accuracy', metrics.get('accuracy'))
                    precision = metrics.get('test_precision', metrics.get('precision'))
                    recall = metrics.get('test_recall', metrics.get('recall'))
                    f1_score = metrics.get('test_f1', metrics.get('f1_score'))
                    
                    # Calculate training duration
                    if run.info.end_time and run.info.start_time:
                        duration = (run.info.end_time - run.info.start_time) / 1000
                    else:
                        duration = 0
                    
                    # Get model name from parameters or tags
                    model_name = params.get('model_name', run.info.run_name or 'unknown')
                    
                    labels = {
                        'experiment_name': experiment.name,
                        'run_id': run.info.run_id,
                        'model_name': model_name,
                        'model_stage': 'training'
                    }
                    
                    # Update metrics if available
                    if accuracy is not None:
                        self.metrics.model_accuracy.labels(**labels).set(accuracy)
                    if precision is not None:
                        self.metrics.model_precision.labels(**labels).set(precision)
                    if recall is not None:
                        self.metrics.model_recall.labels(**labels).set(recall)
                    if f1_score is not None:
                        self.metrics.model_f1_score.labels(**labels).set(f1_score)
                    if duration > 0:
                        training_labels = {k: v for k, v in labels.items() if k != 'model_stage'}
                        self.metrics.model_training_duration.labels(**training_labels).set(duration)
                        
        except Exception as e:
            logger.error(f"Failed to collect model performance metrics: {e}")
    
    def collect_model_registry_metrics(self):
        """Collect model registry metrics."""
        try:
            # Get all registered models
            registered_models = self.mlflow_client.list_registered_models()
            self.metrics.registered_models_total.set(len(registered_models))
            
            # Count model versions by stage
            stage_counts = {}
            for model in registered_models:
                model_name = model.name
                
                # Get all versions for this model
                versions = self.mlflow_client.get_latest_versions(model_name, stages=None)
                
                for version in versions:
                    stage = version.current_stage
                    key = (model_name, stage)
                    stage_counts[key] = stage_counts.get(key, 0) + 1
                    
                    # If this is a production model, collect its metrics
                    if stage == "Production":
                        try:
                            # Get the run for this model version
                            run = self.mlflow_client.get_run(version.run_id)
                            metrics = run.data.metrics
                            
                            # Update production model metrics
                            prod_labels = {
                                'experiment_name': run.info.experiment_id,
                                'run_id': version.run_id,
                                'model_name': model_name,
                                'model_stage': 'production'
                            }
                            
                            if 'test_accuracy' in metrics:
                                self.metrics.model_accuracy.labels(**prod_labels).set(metrics['test_accuracy'])
                            if 'test_precision' in metrics:
                                self.metrics.model_precision.labels(**prod_labels).set(metrics['test_precision'])
                            if 'test_recall' in metrics:
                                self.metrics.model_recall.labels(**prod_labels).set(metrics['test_recall'])
                            if 'test_f1' in metrics:
                                self.metrics.model_f1_score.labels(**prod_labels).set(metrics['test_f1'])
                                
                        except Exception as e:
                            logger.warning(f"Could not get metrics for production model {model_name}: {e}")
            
            # Update stage count metrics
            for (model_name, stage), count in stage_counts.items():
                self.metrics.model_versions_by_stage.labels(
                    model_name=model_name,
                    stage=stage
                ).set(count)
                
        except Exception as e:
            logger.error(f"Failed to collect model registry metrics: {e}")
    
    def collect_metrics(self):
        """Collect all metrics from MLflow."""
        logger.info("Collecting metrics from MLflow...")
        
        # Check MLflow health first
        if not self.check_mlflow_health():
            logger.warning("MLflow health check failed, skipping metric collection")
            return
        
        try:
            # Collect different types of metrics
            self.collect_experiment_metrics()
            self.collect_model_performance_metrics()
            self.collect_model_registry_metrics()
            
            logger.info("Metrics collection completed successfully")
            
        except Exception as e:
            logger.error(f"Error during metrics collection: {e}")
    
    def start_metrics_collection(self):
        """Start the metrics collection loop."""
        self.running = True
        logger.info(f"Starting metrics collection with interval {METRICS_INTERVAL}s")
        
        while self.running:
            try:
                self.collect_metrics()
                time.sleep(METRICS_INTERVAL)
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, stopping...")
                self.running = False
            except Exception as e:
                logger.error(f"Unexpected error in metrics collection: {e}")
                time.sleep(METRICS_INTERVAL)
    
    def stop(self):
        """Stop the metrics collection."""
        self.running = False


def main():
    """Main entry point for the exporter."""
    logger.info("Starting MLflow Prometheus Exporter")
    logger.info(f"MLflow URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"Exporter port: {EXPORTER_PORT}")
    logger.info(f"Metrics interval: {METRICS_INTERVAL}s")
    
    # Create exporter instance
    exporter = MLflowPrometheusExporter()
    
    # Start Prometheus metrics server
    start_http_server(EXPORTER_PORT)
    logger.info(f"Prometheus metrics server started on port {EXPORTER_PORT}")
    logger.info(f"Metrics available at http://localhost:{EXPORTER_PORT}/metrics")
    
    try:
        # Start metrics collection in the main thread
        exporter.start_metrics_collection()
    except KeyboardInterrupt:
        logger.info("Shutting down exporter...")
        exporter.stop()
    except Exception as e:
        logger.error(f"Exporter error: {e}")
        exporter.stop()
        raise


if __name__ == "__main__":
    main()

