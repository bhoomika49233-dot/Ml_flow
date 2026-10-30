import mlflow
import mlflow.sklearn
import pandas as pd

from config import EXPERIMENT_NAME, TRACKING_URI


def get_best_run_id() -> str:
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        order_by=["metrics.accuracy DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError("No runs found. Run train.py first.")

    best = runs.iloc[0]
    print(f"Best run: {best['run_id']}")
    print(f"  name={best.get('tags.mlflow.runName')}  "
          f"accuracy={best['metrics.accuracy']:.4f}")
    return best["run_id"]


def main():
    mlflow.set_tracking_uri(TRACKING_URI)

    run_id = get_best_run_id()
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    # Four new flowers: sepal length, sepal width, petal length, petal width
    samples = pd.DataFrame(
        [
            [5.1, 3.5, 1.4, 0.2],   # expected: setosa (0)
            [6.9, 3.1, 4.9, 1.5],   # expected: versicolor (1)
            [6.3, 3.3, 6.0, 2.5],   # expected: virginica (2)
        ],
        columns=["sepal length (cm)", "sepal width (cm)",
                 "petal length (cm)", "petal width (cm)"],
    )

    class_names = ["setosa", "versicolor", "virginica"]
    predictions = model.predict(samples)

    print("\nPredictions:")
    for row, pred in zip(samples.itertuples(index=False), predictions):
        print(f"  {list(row)} -> {class_names[pred]}")


if __name__ == "__main__":
    main()