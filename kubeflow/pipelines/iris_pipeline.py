"""
Complete Kubeflow Pipeline for Iris Classification MLOps
This pipeline orchestrates the entire ML workflow from training to deployment.
"""

from kfp import dsl
from kfp.client import Client
import os
import sys

# Add parent directory to path to import components
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'components'))

from training_component import train_iris_model, hyperparameter_tuning
from evaluation_component import evaluate_model, model_validation, register_model_if_valid

@dsl.pipeline(
    name="iris-classification-mlops-pipeline",
    description="Complete MLOps pipeline for Iris classification with MLflow tracking"
)
def iris_mlops_pipeline(
    mlflow_tracking_uri: str = "http://mlflow-server:5000",
    experiment_name: str = "iris_classification_kubeflow",
    model_name: str = "iris_random_forest_kubeflow",
    accuracy_threshold: float = 0.85,
    precision_threshold: float = 0.85,
    perform_hyperparameter_tuning: bool = True,
    deploy_if_valid: bool = True
):
    """
    Complete MLOps pipeline for Iris classification.
    
    Args:
        mlflow_tracking_uri: MLflow tracking server URI
        experiment_name: Name of the MLflow experiment
        model_name: Name for model registration
        accuracy_threshold: Minimum accuracy for model validation
        precision_threshold: Minimum precision for model validation
        perform_hyperparameter_tuning: Whether to perform hyperparameter tuning
        deploy_if_valid: Whether to register model if validation passes
    """
    
    # Step 1: Basic model training
    training_task = train_iris_model(
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        n_estimators=100,
        max_depth=5,
        random_state=42,
        test_size=0.2
    )
    training_task.set_display_name("Train Base Model")
    training_task.set_cpu_limit("1")
    training_task.set_memory_limit("2Gi")
    
    # Step 2: Hyperparameter tuning (conditional)
    with dsl.Condition(perform_hyperparameter_tuning == True):
        tuning_task = hyperparameter_tuning(
            mlflow_tracking_uri=mlflow_tracking_uri,
            experiment_name=experiment_name,
            param_grid={
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10]
            },
            cv_folds=5,
            random_state=42
        )
        tuning_task.set_display_name("Hyperparameter Tuning")
        tuning_task.set_cpu_limit("2")
        tuning_task.set_memory_limit("4Gi")
        tuning_task.after(training_task)
        
        # Use tuned model for evaluation
        best_model = tuning_task.outputs['best_model_output']
        best_run_id = tuning_task.outputs['best_run_id']
    
    # Step 3: Model evaluation
    with dsl.Condition(perform_hyperparameter_tuning == True):
        # Evaluate tuned model
        evaluation_task_tuned = evaluate_model(
            model_input=tuning_task.outputs['best_model_output'],
            mlflow_tracking_uri=mlflow_tracking_uri,
            experiment_name=experiment_name,
            model_run_id=tuning_task.outputs['best_run_id']
        )
        evaluation_task_tuned.set_display_name("Evaluate Tuned Model")
        evaluation_task_tuned.set_cpu_limit("1")
        evaluation_task_tuned.set_memory_limit("2Gi")
        evaluation_task_tuned.after(tuning_task)
        
        eval_model = tuning_task.outputs['best_model_output']
        eval_run_id = tuning_task.outputs['best_run_id']
    
    with dsl.Condition(perform_hyperparameter_tuning == False):
        # Evaluate base model
        evaluation_task_base = evaluate_model(
            model_input=training_task.outputs['model_output'],
            mlflow_tracking_uri=mlflow_tracking_uri,
            experiment_name=experiment_name,
            model_run_id=training_task.outputs['run_id']
        )
        evaluation_task_base.set_display_name("Evaluate Base Model")
        evaluation_task_base.set_cpu_limit("1")
        evaluation_task_base.set_memory_limit("2Gi")
        evaluation_task_base.after(training_task)
        
        eval_model = training_task.outputs['model_output']
        eval_run_id = training_task.outputs['run_id']
    
    # Step 4: Model validation
    # Get the appropriate model and run_id based on tuning condition
    validation_task = model_validation(
        model_input=dsl.OneOf(
            tuning_task.outputs['best_model_output'] if perform_hyperparameter_tuning else None,
            training_task.outputs['model_output']
        ),
        accuracy_threshold=accuracy_threshold,
        precision_threshold=precision_threshold
    )
    validation_task.set_display_name("Validate Model")
    validation_task.set_cpu_limit("1")
    validation_task.set_memory_limit("1Gi")
    
    # Step 5: Model registration (conditional on validation and deploy flag)
    with dsl.Condition(deploy_if_valid == True):
        registration_task = register_model_if_valid(
            mlflow_tracking_uri=mlflow_tracking_uri,
            model_run_id=dsl.OneOf(
                tuning_task.outputs['best_run_id'] if perform_hyperparameter_tuning else None,
                training_task.outputs['run_id']
            ),
            model_name=model_name,
            is_model_valid=validation_task.outputs['is_valid'],
            validation_score=validation_task.outputs['validation_score'],
            model_stage="Staging"
        )
        registration_task.set_display_name("Register Model")
        registration_task.set_cpu_limit("0.5")
        registration_task.set_memory_limit("1Gi")
        registration_task.after(validation_task)

@dsl.pipeline(
    name="iris-simple-training-pipeline",
    description="Simplified Iris training pipeline for quick experiments"
)
def iris_simple_pipeline(
    mlflow_tracking_uri: str = "http://mlflow-server:5000",
    experiment_name: str = "iris_simple_experiments",
    n_estimators: int = 100,
    max_depth: int = 5
):
    """
    Simplified pipeline for quick model experiments.
    
    Args:
        mlflow_tracking_uri: MLflow tracking server URI
        experiment_name: Name of the MLflow experiment
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees
    """
    
    # Train model
    training_task = train_iris_model(
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        test_size=0.2
    )
    training_task.set_display_name("Quick Training")
    
    # Evaluate model
    evaluation_task = evaluate_model(
        model_input=training_task.outputs['model_output'],
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        model_run_id=training_task.outputs['run_id']
    )
    evaluation_task.set_display_name("Quick Evaluation")
    evaluation_task.after(training_task)

@dsl.pipeline(
    name="iris-hyperparameter-optimization-pipeline",
    description="Dedicated pipeline for hyperparameter optimization"
)
def iris_hyperparameter_pipeline(
    mlflow_tracking_uri: str = "http://mlflow-server:5000",
    experiment_name: str = "iris_hyperparameter_optimization",
    param_grid: dict = None
):
    """
    Dedicated pipeline for hyperparameter optimization experiments.
    
    Args:
        mlflow_tracking_uri: MLflow tracking server URI
        experiment_name: Name of the MLflow experiment
        param_grid: Hyperparameter grid for tuning
    """
    
    if param_grid is None:
        param_grid = {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [3, 5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    
    # Hyperparameter tuning
    tuning_task = hyperparameter_tuning(
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        param_grid=param_grid,
        cv_folds=5,
        random_state=42
    )
    tuning_task.set_display_name("Extensive Hyperparameter Tuning")
    tuning_task.set_cpu_limit("4")
    tuning_task.set_memory_limit("8Gi")
    
    # Evaluate best model
    evaluation_task = evaluate_model(
        model_input=tuning_task.outputs['best_model_output'],
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        model_run_id=tuning_task.outputs['best_run_id']
    )
    evaluation_task.set_display_name("Evaluate Best Model")
    evaluation_task.after(tuning_task)

def compile_pipelines():
    """
    Compile all pipeline definitions to YAML files.
    """
    from kfp.compiler import Compiler
    
    compiler = Compiler()
    
    # Compile main pipeline
    compiler.compile(
        pipeline_func=iris_mlops_pipeline,
        package_path="iris_mlops_pipeline.yaml"
    )
    print("Compiled iris_mlops_pipeline.yaml")
    
    # Compile simple pipeline
    compiler.compile(
        pipeline_func=iris_simple_pipeline,
        package_path="iris_simple_pipeline.yaml"
    )
    print("Compiled iris_simple_pipeline.yaml")
    
    # Compile hyperparameter pipeline
    compiler.compile(
        pipeline_func=iris_hyperparameter_pipeline,
        package_path="iris_hyperparameter_pipeline.yaml"
    )
    print("Compiled iris_hyperparameter_pipeline.yaml")

def submit_pipeline(pipeline_func, run_name, arguments=None, kubeflow_endpoint=None):
    """
    Submit a pipeline to Kubeflow.
    
    Args:
        pipeline_func: Pipeline function to run
        run_name: Name for the pipeline run
        arguments: Dictionary of pipeline arguments
        kubeflow_endpoint: Kubeflow Pipelines endpoint URL
    """
    if arguments is None:
        arguments = {}
    
    if kubeflow_endpoint is None:
        kubeflow_endpoint = "http://localhost:8080"  # Default local endpoint
    
    try:
        client = Client(host=kubeflow_endpoint)
        
        # Create experiment if it doesn't exist
        experiment_name = arguments.get('experiment_name', 'iris_mlops_experiments')
        try:
            experiment = client.create_experiment(experiment_name)
        except Exception:
            # Experiment might already exist
            experiments = client.list_experiments()
            experiment = next((exp for exp in experiments.experiments if exp.name == experiment_name), None)
            if experiment is None:
                raise Exception(f"Could not create or find experiment: {experiment_name}")
        
        # Submit the run
        run = client.run_pipeline(
            experiment_id=experiment.id,
            job_name=run_name,
            pipeline_func=pipeline_func,
            params=arguments
        )
        
        print(f"Pipeline submitted successfully!")
        print(f"Run Name: {run_name}")
        print(f"Run ID: {run.id}")
        print(f"Experiment: {experiment_name}")
        
        return run
        
    except Exception as e:
        print(f"Failed to submit pipeline: {e}")
        return None

def create_sample_run_configs():
    """
    Create sample configuration files for different pipeline runs.
    """
    configs = {
        "development_run": {
            "mlflow_tracking_uri": "http://localhost:5000",
            "experiment_name": "iris_dev_experiments",
            "model_name": "iris_rf_dev",
            "accuracy_threshold": 0.8,
            "precision_threshold": 0.8,
            "perform_hyperparameter_tuning": False,
            "deploy_if_valid": False
        },
        "staging_run": {
            "mlflow_tracking_uri": "http://mlflow-server:5000",
            "experiment_name": "iris_staging_experiments", 
            "model_name": "iris_rf_staging",
            "accuracy_threshold": 0.85,
            "precision_threshold": 0.85,
            "perform_hyperparameter_tuning": True,
            "deploy_if_valid": True
        },
        "production_run": {
            "mlflow_tracking_uri": "http://mlflow-server:5000",
            "experiment_name": "iris_production_experiments",
            "model_name": "iris_rf_production",
            "accuracy_threshold": 0.9,
            "precision_threshold": 0.9,
            "perform_hyperparameter_tuning": True,
            "deploy_if_valid": True
        }
    }
    
    import json
    
    for config_name, config in configs.items():
        with open(f"{config_name}.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"Created {config_name}.json")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Kubeflow Pipeline Management")
    parser.add_argument("--action", choices=["compile", "submit", "create-configs"], 
                       default="compile", help="Action to perform")
    parser.add_argument("--pipeline", choices=["mlops", "simple", "hyperparameter"], 
                       default="mlops", help="Pipeline to use")
    parser.add_argument("--run-name", default="iris-pipeline-run", help="Name for pipeline run")
    parser.add_argument("--config-file", help="JSON config file with pipeline parameters")
    parser.add_argument("--kubeflow-endpoint", help="Kubeflow Pipelines endpoint URL")
    
    args = parser.parse_args()
    
    if args.action == "compile":
        compile_pipelines()
    
    elif args.action == "create-configs":
        create_sample_run_configs()
    
    elif args.action == "submit":
        # Load config if provided
        arguments = {}
        if args.config_file:
            import json
            with open(args.config_file, 'r') as f:
                arguments = json.load(f)
        
        # Select pipeline
        pipeline_funcs = {
            "mlops": iris_mlops_pipeline,
            "simple": iris_simple_pipeline,
            "hyperparameter": iris_hyperparameter_pipeline
        }
        
        pipeline_func = pipeline_funcs[args.pipeline]
        
        submit_pipeline(
            pipeline_func=pipeline_func,
            run_name=args.run_name,
            arguments=arguments,
            kubeflow_endpoint=args.kubeflow_endpoint
        )


