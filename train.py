import mlflow
import optuna
import uuid

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score    

UNIQUE_PREFIX = str(uuid.uuid4())[:8]

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 1000, step=100) 
    max_depth = trial.suggest_int("max_depth", 3, 10)

    iris = load_iris()
    X, y = iris.data, iris.target

    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf_classifier, X, y, cv=kfold, scoring="accuracy")

    acc_score = scores.mean()
    with mlflow.start_run(run_name=f"{UNIQUE_PREFIX}_trial_{trial.number}"):
        mlflow.log_params(trial.params)
        mlflow.log_metric("accuracy", acc_score)
    return acc_score

def train_best_model(params):
    run_name = f"{UNIQUE_PREFIX}_best_model"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        
        iris = load_iris()
        rf_classifier = RandomForestClassifier(**params, random_state=42)
        # n_estimators - tree count
        # max_depth - tree depth
        X = iris["data"]
        y = iris["target"]
        rf_classifier.fit(X, y)
        return rf_classifier


if __name__ == "__main__":
    study_name = "hpo_tutorial"
    
    mlflow.set_tracking_uri("http://0.0.0.0:5001")
    mlflow.set_experiment(study_name)

    sampler = optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=study_name, load_if_exists=True)
    
    study.optimize(objective, n_trials=10)

    print(f"Best params: {study.best_params}")
    print(f"Best value: {study.best_value}")

    best_params = study.best_params
    best_model = train_best_model(best_params)

