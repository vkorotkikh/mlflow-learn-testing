from mlflow import MlflowClient
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

client = MlflowClient(tracking_uri="http://127.0.0.1:8080")

