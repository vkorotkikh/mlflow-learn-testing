# MLOps Pipeline Makefile

.PHONY: help install setup-dev setup-prod clean test lint format run-pipeline docker-build docker-up docker-down

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install Python dependencies"
	@echo "  setup-dev    - Setup development environment"
	@echo "  setup-prod   - Setup production environment"
	@echo "  clean        - Clean temporary files and cache"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code"
	@echo "  run-pipeline - Run the training pipeline locally"
	@echo "  docker-build - Build Docker images"
	@echo "  docker-up    - Start services with Docker Compose"
	@echo "  docker-down  - Stop services"
	@echo "  mlflow-ui    - Start MLflow UI"
	@echo "  airflow-ui   - Start Airflow UI"

# Installation and Setup
install:
	pip install -r requirements.txt

setup-dev:
	pip install -r requirements.txt
	pip install -e .
	pre-commit install
	@echo "Development environment setup complete"

setup-prod:
	pip install -r requirements.txt --no-dev
	@echo "Production environment setup complete"

# Code Quality
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/

test:
	pytest tests/ -v --cov=models --cov=mlflow --cov=kubeflow --cov=airflow

lint:
	flake8 models/ mlflow/ kubeflow/ airflow/
	mypy models/ mlflow/ kubeflow/

format:
	black models/ mlflow/ kubeflow/ airflow/
	isort models/ mlflow/ kubeflow/ airflow/

# Pipeline Execution
run-pipeline:
	python models/training.py

run-evaluation:
	python models/evaluation.py

run-experiment:
	python mlflow/experiments/experiment_manager.py

# MLflow Operations
mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

mlflow-server:
	mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# Docker Operations
docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-up-full:
	docker-compose -f docker/docker-compose.yml --profile postgres --profile jupyter --profile airflow up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down

docker-logs:
	docker-compose -f docker/docker-compose.yml logs -f

docker-clean:
	docker-compose -f docker/docker-compose.yml down -v
	docker system prune -f

# Airflow Operations
airflow-init:
	cd airflow && airflow db init
	cd airflow && airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin

airflow-webserver:
	cd airflow && airflow webserver --port 8080

airflow-scheduler:
	cd airflow && airflow scheduler

airflow-ui: airflow-webserver

# Kubeflow Operations
compile-pipelines:
	cd kubeflow/pipelines && python iris_pipeline.py --action compile

submit-pipeline:
	cd kubeflow/pipelines && python iris_pipeline.py --action submit --pipeline mlops

create-configs:
	cd kubeflow/pipelines && python iris_pipeline.py --action create-configs

# Monitoring and Maintenance
check-health:
	@echo "Checking MLflow health..."
	@curl -f http://localhost:5000/health || echo "MLflow not responding"
	@echo "Checking Airflow health..."
	@curl -f http://localhost:8080/health || echo "Airflow not responding"

backup-mlflow:
	@echo "Creating MLflow backup..."
	cp mlflow.db mlflow_backup_$(shell date +%Y%m%d_%H%M%S).db
	tar -czf mlruns_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz mlruns/

# Data and Model Management
download-data:
	@echo "Data is loaded from sklearn.datasets, no download needed"

validate-data:
	python -c "from sklearn.datasets import load_iris; iris = load_iris(); print(f'Dataset loaded: {iris.data.shape[0]} samples, {iris.data.shape[1]} features')"

# Environment Management
create-env:
	conda create -n mlops-pipeline python=3.9 -y
	@echo "Activate with: conda activate mlops-pipeline"

update-requirements:
	pip freeze > requirements.txt

# Documentation
docs:
	@echo "Generating documentation..."
	python -m pydoc -w models mlflow kubeflow airflow

# Development Helpers
notebook:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

install-dev-tools:
	pip install black isort flake8 mypy pytest pytest-cov pre-commit jupyter

pre-commit-all:
	pre-commit run --all-files

# Quick Start
quick-start: install mlflow-server
	@echo "MLOps pipeline is ready!"
	@echo "MLflow UI: http://localhost:5000"
	@echo "Run 'make run-pipeline' to execute the training pipeline"

# Production Deployment
deploy-staging:
	@echo "Deploying to staging environment..."
	docker-compose -f docker/docker-compose.yml --profile postgres up -d

deploy-prod:
	@echo "Deploying to production environment..."
	@echo "This should be run in your production environment"
	docker-compose -f docker/docker-compose.yml --profile postgres --profile airflow up -d


