# MLOps Pipeline with Airflow, Kubeflow, and MLflow

A comprehensive MLOps pipeline example that demonstrates the integration of Airflow, Kubeflow, and MLflow for end-to-end machine learning lifecycle management using a scikit-learn Iris classification model.

## 🚀 Overview

This project showcases a complete MLOps pipeline that includes:

- **Model Training**: Scikit-learn Random Forest classifier on the Iris dataset
- **Experiment Tracking**: MLflow for logging experiments, metrics, and model registry
- **Pipeline Orchestration**: Kubeflow Pipelines for ML workflow execution
- **Workflow Management**: Airflow for scheduling and monitoring the entire pipeline
- **Containerization**: Docker containers for reproducible environments
- **Model Deployment**: Automated model promotion and deployment decisions

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Airflow     │───▶│   Kubeflow      │───▶│     MLflow      │
│   (Scheduler)   │    │  (Pipelines)    │    │ (Tracking/Registry)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Prep     │    │ Model Training  │    │ Model Registry  │
│   Validation    │    │   Evaluation    │    │   Deployment    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
MLflow-learn-testing/
├── airflow/                    # Airflow DAGs and plugins
│   ├── dags/
│   │   └── iris_mlops_dag.py  # Main MLOps pipeline DAG
│   └── plugins/
│       └── mlops_utils.py     # Utility functions
├── kubeflow/                   # Kubeflow pipeline components
│   ├── components/
│   │   ├── training_component.py    # Training component
│   │   └── evaluation_component.py  # Evaluation component
│   └── pipelines/
│       └── iris_pipeline.py    # Complete pipeline definition
├── models/                     # Model training and evaluation
│   ├── training.py            # Training script
│   └── evaluation.py          # Evaluation script
├── mlflow/                     # MLflow utilities
│   ├── experiments/
│   │   └── experiment_manager.py   # Experiment management
│   └── models/
│       └── model_registry.py  # Model registry operations
├── docker/                     # Docker configurations
│   ├── Dockerfile.mlflow      # MLflow server
│   ├── Dockerfile.pipeline    # Pipeline execution
│   └── docker-compose.yml     # Multi-service setup
├── config/                     # Configuration files
│   └── config.yaml           # Main configuration
├── requirements.txt           # Python dependencies
├── Makefile                  # Development commands
└── README.md                 # This file
```

## 🔧 Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Make (optional, for convenient commands)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd MLflow-learn-testing

# Install dependencies
make install
# or
pip install -r requirements.txt
```

### 2. Start MLflow Tracking Server

```bash
# Using Make
make mlflow-ui

# Or directly
mlflow ui --host 0.0.0.0 --port 5000
```

MLflow UI will be available at http://localhost:5000

### 3. Run Training Pipeline

```bash
# Run the training pipeline locally
make run-pipeline
# or
python models/training.py
```

### 4. Evaluate Model

```bash
# Run model evaluation
make run-evaluation
# or
python models/evaluation.py
```

## 🐳 Docker Setup

### Basic Setup (MLflow Only)

```bash
# Start MLflow server
make docker-up
# or
docker-compose -f docker/docker-compose.yml up -d
```

### Full Setup (All Services)

```bash
# Start all services (MLflow, PostgreSQL, Jupyter, Airflow)
make docker-up-full
# or
docker-compose -f docker/docker-compose.yml --profile postgres --profile jupyter --profile airflow up -d
```

### Services URLs

- **MLflow UI**: http://localhost:5000
- **Airflow UI**: http://localhost:8080 (admin/admin)
- **Jupyter Lab**: http://localhost:8888

## 🔄 Pipeline Components

### 1. Model Training (models/training.py)

- Loads Iris dataset from sklearn
- Trains Random Forest classifier with hyperparameter tuning
- Logs experiments to MLflow
- Saves model artifacts

```python
# Example usage
from models.training import IrisModelTrainer

trainer = IrisModelTrainer()
trainer.load_and_prepare_data()
results = trainer.train_model(n_estimators=100, max_depth=5)
```

### 2. Model Evaluation (models/evaluation.py)

- Evaluates trained models on test data
- Generates comprehensive metrics and reports
- Logs evaluation results to MLflow

```python
# Example usage
from models.evaluation import ModelEvaluator

evaluator = ModelEvaluator()
evaluator.load_test_data()
evaluator.load_model_from_mlflow("iris_random_forest", "latest")
results = evaluator.evaluate_model()
```

### 3. Kubeflow Pipeline (kubeflow/pipelines/iris_pipeline.py)

Complete ML pipeline with components for:
- Model training with hyperparameter tuning
- Model evaluation and validation
- Model registration and promotion

```bash
# Compile pipeline
cd kubeflow/pipelines
python iris_pipeline.py --action compile

# Submit pipeline (requires Kubeflow setup)
python iris_pipeline.py --action submit --pipeline mlops
```

### 4. Airflow DAG (airflow/dags/iris_mlops_dag.py)

Orchestrates the complete MLOps workflow:
- Health checks for MLflow and Kubeflow
- Pipeline execution and monitoring
- Model performance analysis
- Automated promotion decisions

## ⚙️ Configuration

### Environment Variables

Copy and customize the environment file:

```bash
cp docker/env-example docker/.env
# Edit docker/.env with your settings
```

### Configuration File

Edit `config/config.yaml` to customize:

- MLflow tracking URI
- Model hyperparameters
- Performance thresholds
- Monitoring settings

### Airflow Variables

Set up Airflow variables for pipeline configuration:

```bash
# Example Airflow variable setup
airflow variables set mlflow_tracking_uri "http://localhost:5000"
airflow variables set experiment_name "iris_classification_production"
airflow variables set notification_email "admin@company.com"
```

## 📊 MLflow Integration

### Experiment Tracking

The pipeline automatically tracks:
- Model hyperparameters
- Training and validation metrics
- Model artifacts and signatures
- Feature importance
- Classification reports

### Model Registry

Models are automatically registered with:
- Version management
- Stage transitions (Staging → Production)
- Model metadata and descriptions
- Performance metrics

### Example MLflow Operations

```python
# Using MLflow utilities
from mlflow.experiments.experiment_manager import MLflowExperimentManager

manager = MLflowExperimentManager()

# List experiments
experiments = manager.list_experiments()

# Compare model metrics
comparison = manager.compare_model_metrics("iris_classification", ["test_accuracy", "f1_score"])

# Export experiment data
manager.export_experiment_data("iris_classification", "experiment_data.json")
```

## 🚀 Kubeflow Pipeline

### Components

1. **Training Component**: Trains the model with configurable parameters
2. **Hyperparameter Tuning**: Automated hyperparameter optimization
3. **Evaluation Component**: Comprehensive model evaluation
4. **Validation Component**: Validates model against thresholds
5. **Registration Component**: Registers valid models in MLflow

### Pipeline Execution

```bash
# Local compilation
python kubeflow/pipelines/iris_pipeline.py --action compile

# Create sample configurations
python kubeflow/pipelines/iris_pipeline.py --action create-configs

# Submit to Kubeflow (requires Kubeflow cluster)
python kubeflow/pipelines/iris_pipeline.py --action submit --config-file development_run.json
```

## 🔄 Airflow Orchestration

### DAG Features

- **Daily scheduling** with configurable intervals
- **Health checks** for all services
- **Pipeline monitoring** with automatic retries
- **Model promotion** based on performance thresholds
- **Email notifications** for success/failure
- **Comprehensive logging** and error handling

### DAG Tasks

1. Health checks (MLflow, Kubeflow)
2. Pipeline configuration and submission
3. Execution monitoring
4. Results analysis
5. Model promotion decisions
6. Notifications

## 🧪 Testing and Development

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
pytest --cov=models --cov=mlflow --cov=kubeflow

# Run specific test file
pytest tests/test_training.py -v
```

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Run all pre-commit hooks
make pre-commit-all
```

### Development Environment

```bash
# Setup development environment
make setup-dev

# Start Jupyter notebook
make notebook

# Install development tools
make install-dev-tools
```

## 📈 Monitoring and Alerting

### Model Performance Monitoring

The pipeline includes built-in monitoring for:
- Model accuracy degradation
- Data drift detection
- Pipeline failure alerts
- Performance threshold violations

### Alerting

Configure alerts via:
- Email notifications
- Slack webhooks (configure in config.yaml)
- Custom monitoring dashboards

### Example Monitoring

```python
from airflow.plugins.mlops_utils import MLflowUtils

mlflow_utils = MLflowUtils("http://localhost:5000")

# Check for model degradation
degradation = mlflow_utils.check_model_degradation("iris_random_forest", threshold_drop=0.05)

if degradation["degradation_detected"]:
    # Send alert
    print(f"Model performance dropped by {degradation['performance_drop']:.3f}")
```

## 🔧 Troubleshooting

### Common Issues

1. **MLflow server not starting**:
   ```bash
   # Check port availability
   lsof -i :5000
   
   # Restart MLflow
   make mlflow-server
   ```

2. **Docker containers failing**:
   ```bash
   # Check logs
   make docker-logs
   
   # Restart services
   make docker-down && make docker-up
   ```

3. **Airflow DAG not appearing**:
   ```bash
   # Check DAG syntax
   python airflow/dags/iris_mlops_dag.py
   
   # Refresh DAGs
   airflow dags list-import-errors
   ```

4. **Kubeflow pipeline submission failing**:
   - Ensure Kubeflow cluster is accessible
   - Check network connectivity
   - Verify pipeline configuration

### Logs and Debugging

```bash
# View all logs
make docker-logs

# Check specific service logs
docker logs mlflow-server
docker logs airflow-standalone

# Debug pipeline execution
python models/training.py --debug
```

## 🚀 Deployment

### Development Environment

```bash
# Quick start for development
make quick-start
```

### Staging Environment

```bash
# Deploy to staging
make deploy-staging
```

### Production Environment

For production deployment:

1. Update configuration for production URLs
2. Set up external databases (PostgreSQL for MLflow)
3. Configure proper authentication and security
4. Set up monitoring and alerting
5. Deploy to Kubernetes cluster

```bash
# Production deployment
make deploy-prod
```

## 📚 Additional Resources

### Documentation

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/)
- [Apache Airflow](https://airflow.apache.org/docs/)
- [Scikit-learn](https://scikit-learn.org/stable/)

### Example Notebooks

Create Jupyter notebooks for experimentation:

```bash
# Start Jupyter
make notebook

# Access at http://localhost:8888
```

### API Documentation

Generate API documentation:

```bash
# Generate docs
make docs
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support

For questions and support:

1. Check the troubleshooting section
2. Review the documentation
3. Open an issue on GitHub
4. Contact the MLOps team

---

**Happy MLOps! 🚀**


