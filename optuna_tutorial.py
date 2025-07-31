import optuna

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def objective(trial):
    trial.suggest_int("n_estimators", 100, 1000, step=100) 
    trial.suggest_int("max_depth", 3, 10)

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

    rf_classifier = RandomForestClassifier(n_estimators=trial.params["n_estimators"], max_depth=trial.params["max_depth"], random_state=42)
    # n_estimators - tree count
    # max_depth - tree depth

    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc_score:.4f}")
    return acc_score


if __name__ == "__main__":
    sampler = optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="hpo_tutorial")
    
    study.optimize(objective, n_trials=10)

    print(f"Best params: {study.best_params}")
    print(f"Best value: {study.best_value}")