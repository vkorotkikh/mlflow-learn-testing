"""
Airflow DAG for Iris MLOps Pipeline
This DAG orchestrates the complete ML lifecycle including Kubeflow pipeline execution,
MLflow experiment tracking, and model deployment decisions.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.providers.http.sensors.http import HttpSensor
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.sensors.filesystem import FileSensor
from airflow.operators.email import EmailOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
import os
import json
import logging

# Configuration from Airflow Variables
MLFLOW_TRACKING_URI = Variable.get("mlflow_tracking_uri", default_var="http://localhost:5000")
KUBEFLOW_ENDPOINT = Variable.get("kubeflow_endpoint", default_var="http://localhost:8080")
EXPERIMENT_NAME = Variable.get("experiment_name", default_var="iris_classification_production")
MODEL_NAME = Variable.get("model_name", default_var="iris_random_forest_production")
NOTIFICATION_EMAIL = Variable.get("notification_email", default_var="admin@company.com")

# Default arguments for the DAG
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email': [NOTIFICATION_EMAIL]
}

# Create the DAG
dag = DAG(
    'iris_mlops_pipeline',
    default_args=default_args,
    description='Complete MLOps pipeline for Iris classification',
    schedule_interval='@daily',  # Run daily
    catchup=False,
    max_active_runs=1,
    tags=['mlops', 'iris', 'classification', 'kubeflow', 'mlflow']
)

def check_mlflow_health(**context):
    """Check if MLflow tracking server is healthy"""
    import requests
    import time
    
    max_retries = 5
    retry_delay = 30
    
    for i in range(max_retries):
        try:
            response = requests.get(f"{MLFLOW_TRACKING_URI}/health", timeout=10)
            if response.status_code == 200:
                logging.info("MLflow server is healthy")
                return True
            else:
                logging.warning(f"MLflow health check failed with status {response.status_code}")
        except Exception as e:
            logging.warning(f"MLflow health check attempt {i+1} failed: {e}")
            
        if i < max_retries - 1:
            time.sleep(retry_delay)
    
    raise Exception("MLflow server health check failed after all retries")

def check_kubeflow_health(**context):
    """Check if Kubeflow Pipelines is healthy"""
    import requests
    import time
    
    max_retries = 5
    retry_delay = 30
    
    for i in range(max_retries):
        try:
            # Check Kubeflow Pipelines API health
            response = requests.get(f"{KUBEFLOW_ENDPOINT}/apis/v1beta1/healthz", timeout=10)
            if response.status_code == 200:
                logging.info("Kubeflow Pipelines is healthy")
                return True
            else:
                logging.warning(f"Kubeflow health check failed with status {response.status_code}")
        except Exception as e:
            logging.warning(f"Kubeflow health check attempt {i+1} failed: {e}")
            
        if i < max_retries - 1:
            time.sleep(retry_delay)
    
    raise Exception("Kubeflow Pipelines health check failed after all retries")

def prepare_pipeline_config(**context):
    """Prepare configuration for the Kubeflow pipeline"""
    import json
    
    # Get execution date for unique naming
    execution_date = context['execution_date'].strftime('%Y%m%d_%H%M%S')
    
    # Prepare pipeline configuration
    pipeline_config = {
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "experiment_name": f"{EXPERIMENT_NAME}_{execution_date}",
        "model_name": f"{MODEL_NAME}_{execution_date}",
        "accuracy_threshold": 0.85,
        "precision_threshold": 0.85,
        "perform_hyperparameter_tuning": True,
        "deploy_if_valid": True
    }
    
    # Save configuration for downstream tasks
    config_path = f"/tmp/pipeline_config_{execution_date}.json"
    with open(config_path, 'w') as f:
        json.dump(pipeline_config, f, indent=2)
    
    # Push configuration to XCom for other tasks
    context['task_instance'].xcom_push(key='pipeline_config', value=pipeline_config)
    context['task_instance'].xcom_push(key='config_path', value=config_path)
    
    logging.info(f"Pipeline configuration prepared: {pipeline_config}")
    return config_path

def submit_kubeflow_pipeline(**context):
    """Submit the Kubeflow pipeline for execution"""
    import subprocess
    import time
    import json
    
    # Get configuration from previous task
    pipeline_config = context['task_instance'].xcom_pull(key='pipeline_config')
    config_path = context['task_instance'].xcom_pull(key='config_path')
    
    execution_date = context['execution_date'].strftime('%Y%m%d_%H%M%S')
    run_name = f"iris_mlops_run_{execution_date}"
    
    # Path to the pipeline script
    pipeline_script = "/opt/airflow/dags/kubeflow/pipelines/iris_pipeline.py"
    
    # Command to submit the pipeline
    cmd = [
        "python", pipeline_script,
        "--action", "submit",
        "--pipeline", "mlops",
        "--run-name", run_name,
        "--config-file", config_path,
        "--kubeflow-endpoint", KUBEFLOW_ENDPOINT
    ]
    
    try:
        # Submit the pipeline
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logging.info(f"Kubeflow pipeline submitted successfully: {result.stdout}")
            
            # Extract run ID from output if possible
            run_id = None
            for line in result.stdout.split('\n'):
                if 'Run ID:' in line:
                    run_id = line.split('Run ID:')[1].strip()
                    break
            
            # Push run information to XCom
            context['task_instance'].xcom_push(key='kubeflow_run_id', value=run_id)
            context['task_instance'].xcom_push(key='kubeflow_run_name', value=run_name)
            
            return run_id
        else:
            logging.error(f"Pipeline submission failed: {result.stderr}")
            raise Exception(f"Kubeflow pipeline submission failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        raise Exception("Pipeline submission timed out")
    except Exception as e:
        logging.error(f"Error submitting pipeline: {e}")
        raise

def monitor_pipeline_execution(**context):
    """Monitor the Kubeflow pipeline execution status"""
    import time
    import requests
    import json
    
    run_id = context['task_instance'].xcom_pull(key='kubeflow_run_id')
    
    if not run_id:
        raise Exception("No Kubeflow run ID found")
    
    max_wait_time = 3600  # 1 hour
    check_interval = 60   # 1 minute
    elapsed_time = 0
    
    logging.info(f"Monitoring Kubeflow pipeline run: {run_id}")
    
    while elapsed_time < max_wait_time:
        try:
            # Check run status via Kubeflow API
            response = requests.get(
                f"{KUBEFLOW_ENDPOINT}/apis/v1beta1/runs/{run_id}",
                timeout=30
            )
            
            if response.status_code == 200:
                run_info = response.json()
                status = run_info.get('status', 'Unknown')
                
                logging.info(f"Pipeline status: {status}")
                
                if status in ['Succeeded', 'Completed']:
                    logging.info("Pipeline completed successfully")
                    context['task_instance'].xcom_push(key='pipeline_status', value='success')
                    return 'success'
                elif status in ['Failed', 'Error']:
                    logging.error("Pipeline failed")
                    context['task_instance'].xcom_push(key='pipeline_status', value='failed')
                    raise Exception(f"Kubeflow pipeline failed with status: {status}")
                
                # Still running, wait and check again
                time.sleep(check_interval)
                elapsed_time += check_interval
                
            else:
                logging.warning(f"Failed to get run status: {response.status_code}")
                time.sleep(check_interval)
                elapsed_time += check_interval
                
        except Exception as e:
            logging.warning(f"Error checking pipeline status: {e}")
            time.sleep(check_interval)
            elapsed_time += check_interval
    
    # Timeout reached
    raise Exception("Pipeline monitoring timed out")

def analyze_pipeline_results(**context):
    """Analyze the results of the pipeline execution"""
    import mlflow
    from mlflow.tracking import MlflowClient
    import pandas as pd
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(MLFLOW_TRACKING_URI)
    
    # Get experiment information
    pipeline_config = context['task_instance'].xcom_pull(key='pipeline_config')
    experiment_name = pipeline_config['experiment_name']
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            raise Exception(f"Experiment {experiment_name} not found")
        
        # Get runs from the experiment
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=10,
            order_by=["start_time DESC"]
        )
        
        if not runs:
            raise Exception("No runs found in the experiment")
        
        # Analyze the latest run
        latest_run = runs[0]
        metrics = latest_run.data.metrics
        params = latest_run.data.params
        
        # Extract key metrics
        analysis_results = {
            'run_id': latest_run.info.run_id,
            'status': latest_run.info.status,
            'accuracy': metrics.get('test_accuracy', 0),
            'precision': metrics.get('evaluation_precision', 0),
            'recall': metrics.get('evaluation_recall', 0),
            'f1_score': metrics.get('evaluation_f1', 0),
            'model_type': params.get('model_type', 'Unknown'),
            'experiment_name': experiment_name
        }
        
        # Check if model meets quality thresholds
        accuracy_threshold = pipeline_config.get('accuracy_threshold', 0.85)
        precision_threshold = pipeline_config.get('precision_threshold', 0.85)
        
        meets_thresholds = (
            analysis_results['accuracy'] >= accuracy_threshold and
            analysis_results['precision'] >= precision_threshold
        )
        
        analysis_results['meets_quality_thresholds'] = meets_thresholds
        
        # Push results to XCom
        context['task_instance'].xcom_push(key='analysis_results', value=analysis_results)
        
        logging.info(f"Pipeline analysis results: {analysis_results}")
        return analysis_results
        
    except Exception as e:
        logging.error(f"Error analyzing pipeline results: {e}")
        raise

def decide_model_promotion(**context):
    """Decide whether to promote the model to production"""
    analysis_results = context['task_instance'].xcom_pull(key='analysis_results')
    
    if not analysis_results:
        raise Exception("No analysis results found")
    
    meets_thresholds = analysis_results.get('meets_quality_thresholds', False)
    accuracy = analysis_results.get('accuracy', 0)
    
    # Additional checks for production promotion
    promotion_criteria = {
        'meets_quality_thresholds': meets_thresholds,
        'high_accuracy': accuracy >= 0.9,  # Higher bar for production
        'no_recent_failures': True  # Could check failure history
    }
    
    should_promote = all(promotion_criteria.values())
    
    promotion_decision = {
        'should_promote': should_promote,
        'criteria': promotion_criteria,
        'model_metrics': analysis_results,
        'decision_timestamp': datetime.now().isoformat()
    }
    
    context['task_instance'].xcom_push(key='promotion_decision', value=promotion_decision)
    
    logging.info(f"Model promotion decision: {promotion_decision}")
    return should_promote

def promote_model_to_production(**context):
    """Promote the model to production stage in MLflow"""
    import mlflow
    from mlflow.tracking import MlflowClient
    
    analysis_results = context['task_instance'].xcom_pull(key='analysis_results')
    promotion_decision = context['task_instance'].xcom_pull(key='promotion_decision')
    
    if not promotion_decision.get('should_promote', False):
        logging.info("Model does not meet promotion criteria, skipping promotion")
        return False
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(MLFLOW_TRACKING_URI)
    
    try:
        # Get the model from the registry
        pipeline_config = context['task_instance'].xcom_pull(key='pipeline_config')
        model_name = pipeline_config['model_name']
        
        # Get the latest version of the model
        latest_versions = client.get_latest_versions(model_name, stages=["Staging"])
        
        if not latest_versions:
            raise Exception(f"No staging version found for model {model_name}")
        
        model_version = latest_versions[0]
        
        # Promote to production
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production",
            archive_existing_versions=True
        )
        
        # Add tags
        client.set_model_version_tag(
            name=model_name,
            version=model_version.version,
            key="promoted_by",
            value="airflow_pipeline"
        )
        
        client.set_model_version_tag(
            name=model_name,
            version=model_version.version,
            key="promotion_date",
            value=datetime.now().isoformat()
        )
        
        logging.info(f"Model {model_name} version {model_version.version} promoted to Production")
        
        context['task_instance'].xcom_push(key='promoted_model_version', value=model_version.version)
        
        # Deploy with BentoML after promotion
        deploy_with_bentoml(model_name, model_version.version)
        
        return True
        
    except Exception as e:
        logging.error(f"Error promoting model: {e}")
        raise

def deploy_with_bentoml(model_name: str, model_version: str):
    """Deploy the promoted model with BentoML"""
    import subprocess
    import sys
    import os
    
    try:
        # Add the serving directory to Python path
        serving_path = os.path.join(os.path.dirname(__file__), '../../serving')
        sys.path.insert(0, serving_path)
        
        from bento_builder import BentoModelBuilder
        
        # Initialize BentoML builder
        builder = BentoModelBuilder(mlflow_tracking_uri=MLFLOW_TRACKING_URI)
        
        # Package the model
        logging.info(f"Packaging model {model_name} version {model_version} with BentoML")
        service_name = f"{model_name}_v{model_version}"
        bento_path = builder.package_mlflow_model(
            model_name=model_name,
            model_version=model_version,
            service_name=service_name
        )
        
        # Containerize the service
        logging.info("Containerizing BentoML service")
        docker_image = builder.containerize_bento(
            service_name=service_name,
            docker_image_name=f"iris-classifier-v{model_version}"
        )
        
        # Deploy to Kubernetes if configured
        if Variable.get("deploy_to_k8s", default_var="false") == "true":
            logging.info("Deploying to Kubernetes")
            k8s_config = builder.deploy_to_kubernetes(
                service_name=service_name,
                namespace="mlops",
                replicas=3
            )
            
            # Apply Kubernetes configurations
            subprocess.run([
                "kubectl", "apply", "-f", k8s_config["deployment_file"]
            ], check=True)
            
            subprocess.run([
                "kubectl", "apply", "-f", k8s_config["service_file"]
            ], check=True)
            
            logging.info(f"Model deployed to Kubernetes as {service_name}")
        
        logging.info(f"BentoML deployment completed for {model_name} version {model_version}")
        
    except Exception as e:
        logging.error(f"Error deploying with BentoML: {e}")
        # Don't fail the entire pipeline if BentoML deployment fails
        logging.warning("BentoML deployment failed but continuing with pipeline")

def send_pipeline_notification(**context):
    """Send notification about pipeline completion"""
    import json
    
    analysis_results = context['task_instance'].xcom_pull(key='analysis_results')
    promotion_decision = context['task_instance'].xcom_pull(key='promotion_decision')
    promoted_version = context['task_instance'].xcom_pull(key='promoted_model_version')
    
    # Prepare notification content
    execution_date = context['execution_date'].strftime('%Y-%m-%d %H:%M:%S')
    
    if analysis_results:
        subject = f"MLOps Pipeline Completed - {execution_date}"
        
        content = f"""
MLOps Pipeline Execution Report
==============================

Execution Date: {execution_date}
Experiment: {analysis_results['experiment_name']}
Run ID: {analysis_results['run_id']}

Model Performance:
- Accuracy: {analysis_results['accuracy']:.4f}
- Precision: {analysis_results['precision']:.4f}
- Recall: {analysis_results['recall']:.4f}
- F1-Score: {analysis_results['f1_score']:.4f}

Quality Assessment:
- Meets Thresholds: {analysis_results['meets_quality_thresholds']}

Model Promotion:
- Promotion Decision: {'Yes' if promotion_decision.get('should_promote') else 'No'}
- Promoted Version: {promoted_version if promoted_version else 'N/A'}

MLflow Tracking: {MLFLOW_TRACKING_URI}
"""
    else:
        subject = f"MLOps Pipeline Failed - {execution_date}"
        content = f"""
MLOps Pipeline Execution Failed
===============================

Execution Date: {execution_date}

The pipeline failed to complete successfully. Please check the Airflow logs for details.

MLflow Tracking: {MLFLOW_TRACKING_URI}
"""
    
    # Store notification content for email task
    context['task_instance'].xcom_push(key='notification_subject', value=subject)
    context['task_instance'].xcom_push(key='notification_content', value=content)
    
    logging.info(f"Notification prepared: {subject}")
    return content

# Task definitions

# Health checks
with TaskGroup("health_checks", dag=dag) as health_checks:
    mlflow_health = PythonOperator(
        task_id='check_mlflow_health',
        python_callable=check_mlflow_health,
        dag=dag
    )
    
    kubeflow_health = PythonOperator(
        task_id='check_kubeflow_health',
        python_callable=check_kubeflow_health,
        dag=dag
    )

# Pipeline execution
with TaskGroup("pipeline_execution", dag=dag) as pipeline_execution:
    prepare_config = PythonOperator(
        task_id='prepare_pipeline_config',
        python_callable=prepare_pipeline_config,
        dag=dag
    )
    
    submit_pipeline = PythonOperator(
        task_id='submit_kubeflow_pipeline',
        python_callable=submit_kubeflow_pipeline,
        dag=dag
    )
    
    monitor_execution = PythonOperator(
        task_id='monitor_pipeline_execution',
        python_callable=monitor_pipeline_execution,
        dag=dag
    )
    
    prepare_config >> submit_pipeline >> monitor_execution

# Analysis and deployment
with TaskGroup("analysis_deployment", dag=dag) as analysis_deployment:
    analyze_results = PythonOperator(
        task_id='analyze_pipeline_results',
        python_callable=analyze_pipeline_results,
        dag=dag
    )
    
    decide_promotion = PythonOperator(
        task_id='decide_model_promotion',
        python_callable=decide_model_promotion,
        dag=dag
    )
    
    promote_model = PythonOperator(
        task_id='promote_model_to_production',
        python_callable=promote_model_to_production,
        dag=dag
    )
    
    analyze_results >> decide_promotion >> promote_model

# Notification
notification_task = PythonOperator(
    task_id='send_pipeline_notification',
    python_callable=send_pipeline_notification,
    dag=dag,
    trigger_rule='all_done'  # Run regardless of upstream success/failure
)

# Define task dependencies
health_checks >> pipeline_execution >> analysis_deployment >> notification_task

# Optional: Add email notification
email_notification = EmailOperator(
    task_id='email_notification',
    to=[NOTIFICATION_EMAIL],
    subject="{{ task_instance.xcom_pull(key='notification_subject') }}",
    html_content="{{ task_instance.xcom_pull(key='notification_content') }}",
    dag=dag,
    trigger_rule='all_done'
)

notification_task >> email_notification


