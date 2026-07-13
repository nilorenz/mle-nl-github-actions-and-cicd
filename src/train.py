import os
from pathlib import Path

import click
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


DATA_PATH = Path("data/green_tripdata_2025-01.parquet")
FEATURES = ["PULocationID", "DOLocationID", "trip_distance"]
TARGET = "trip_duration_minutes"
MODEL_NAME = "green-taxi-trip-duration-xgb"


def load_taxi_data():
    if not DATA_PATH.exists():
        raise click.ClickException(
            f"Missing {DATA_PATH}. Download the taxi Parquet file with the "
            "command in 02-intro-to-dvc.md before training."
        )
    return pd.read_parquet(DATA_PATH)


def calculate_trip_duration_in_minutes(df):
    df = df.copy()
    df[TARGET] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    # Keep a realistic training range and remove obvious trip duration outliers.
    return df[(df[TARGET] >= 1) & (df[TARGET] <= 60)]


def preprocess(df):
    df = calculate_trip_duration_in_minutes(df)
    categorical_features = ["PULocationID", "DOLocationID"]
    df[categorical_features] = df[categorical_features].astype(str)
    # Combining pickup and dropoff zones lets DictVectorizer treat each route as a category.
    df["trip_route"] = df["PULocationID"] + "_" + df["DOLocationID"]
    return df[["trip_route", "trip_distance", TARGET]]


def train_model(df):
    df_processed = preprocess(df)
    y = df_processed[TARGET]
    X = df_processed.drop(columns=[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2
    )

    X_train = X_train.to_dict(orient="records")
    X_test = X_test.to_dict(orient="records")

    pipeline = make_pipeline(
        DictVectorizer(),
        xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    )
    pipeline.fit(X_train, y_train)

    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    metrics = {
        "rmse_train": root_mean_squared_error(y_train, y_pred_train),
        "rmse_test": root_mean_squared_error(y_test, y_pred_test),
        "rows_after_filtering": len(df_processed),
    }
    return pipeline, metrics


def configure_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("green-taxi-trip-duration-xgb")


def log_model(pipeline, metrics):
    with mlflow.start_run():
        mlflow.set_tags(
            {
                "model": "xgboost pipeline",
                "developer": "<your-name>",
                "dataset": "green-taxi",
                "features": ",".join(FEATURES),
                "target": TARGET,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name=MODEL_NAME,
            serialization_format="skops",
            skops_trusted_types=[
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBRegressor",
            ],
        )


def write_cml_metrics(metrics):
    Path("metrics.txt").write_text(
        "\n".join(
            [
                "# Training Metrics",
                "",
                f"- RMSE on the train set: {metrics['rmse_train']:.4f}",
                f"- RMSE on the test set: {metrics['rmse_test']:.4f}",
                f"- Rows after filtering: {metrics['rows_after_filtering']}",
                "",
            ]
        ),
        encoding="utf-8",
    )


@click.command()
@click.option(
    "--cml-run/--no-cml-run",
    default=False,
    help="Write metrics.txt for a CML pull request report.",
)
def main(cml_run):
    """Train a Green Taxi trip duration model."""
    df = load_taxi_data()
    configure_mlflow()

    pipeline, metrics = train_model(df)
    log_model(pipeline, metrics)

    if cml_run:
        write_cml_metrics(metrics)


if __name__ == "__main__":
    main()
