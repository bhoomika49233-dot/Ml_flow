import itertools

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from config import EXPERIMENT_NAME, RANDOM_STATE, TRACKING_URI


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load and split the data
    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    param_grid = {
        "n_estimators": [10, 50, 100],
        "max_depth": [2, 4, None],
    }

    for n_estimators, max_depth in itertools.product(
        param_grid["n_estimators"], param_grid["max_depth"]
    ):
        run_name = f"rf_n{n_estimators}_d{max_depth}"
        with mlflow.start_run(run_name=run_name):
            # 1. Log hyperparameters
            mlflow.log_params({
                "model": "RandomForestClassifier",
                "n_estimators": n_estimators,
                "max_depth": "None" if max_depth is None else max_depth,
                "random_state": RANDOM_STATE,
            })

            # 2. Train
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=RANDOM_STATE,
            )
            model.fit(X_train, y_train)

            # 3. Evaluate and log metrics
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")
            mlflow.log_metrics({"accuracy": acc, "f1_weighted": f1})

            # 4. Log the model with its signature and an input example
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                model,
                name="model",
                serialization_format="cloudpickle",
                signature=signature,
                input_example=X_test.iloc[:3],
            )

            print(f"{run_name:<18} accuracy={acc:.4f}  f1={f1:.4f}")

    print(f"\nDone. Open the UI with: mlflow ui --backend-store-uri {TRACKING_URI}")


if __name__ == "__main__":
    main()