import argparse
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import wandb  # Import WandB for logging
import joblib  # For saving the model
import json  # For saving metrics
from google.cloud import storage
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--job_name", type=str)

    # Add this for Vertex AI to specify output directory
    return parser.parse_args()


def save_to_gcs(bucket_name, destination_blob_name, source_file_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")


def load_data():
    # Your existing load_data function remains the same
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    columns = [
        "CRIM",
        "ZN",
        "INDUS",
        "CHAS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "RAD",
        "TAX",
        "PTRATIO",
        "B",
        "LSTAT",
    ]
    df = pd.DataFrame(data, columns=columns)
    df["MEDV"] = target
    return df


def main():
    args = parse_args()
    #get from en
    wandb.login(key="xxxx")

    # Initialize Weights & Biases logging
    wandb.init(
        project="xgboost-hyperparam-tuning",
        config={
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
        },
    )

    # Load and prepare data
    df = load_data()
    X = df.drop(columns=["MEDV"])
    y = df["MEDV"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and train model
    model = XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
    )
    model.fit(X_train, y_train)

    # Calculate metrics
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    # Log metrics to Weights & Biases
    wandb.log({"rmse": rmse, "mse": mse})

    # Save model and metrics
    job_name = args.job_name

    local_model_path = f"/tmp/{job_name}/model.joblib"
    local_metrics_path = f"/tmp/{job_name}/metrics.json"
    # Ensure the local directory exists
    os.makedirs(os.path.dirname(local_model_path), exist_ok=True)

    # Save model locally first
    joblib.dump(model, local_model_path)

    

    with open(local_metrics_path, "w") as f:
        json.dump({"rmse": rmse, "mse": mse}, f)
    
    

    # model_folder = f"gs://boston-house-price/{args.job_name}"  # Ensure `job_name` is passed as an argument
    # model_path = f"{model_folder}/model.joblib"
    # metrics_path = f"{model_folder}/metrics.json"

    # joblib.dump(model, model_path)

    # # Save metrics to JSON file

    # Print metrics for logging
    print(f"RMSE: {rmse}")
    print(f"MSE: {mse}")
    print(
        f"Hyperparameters: n_estimators={args.n_estimators}, "
        f"max_depth={args.max_depth}, learning_rate={args.learning_rate}, "
        f"subsample={args.subsample}"
    )

    save_to_gcs("boston-house-price", f"{job_name}/model.joblib", local_model_path)
    save_to_gcs("boston-house-price", f"{job_name}/metrics.json", local_metrics_path)

    # Finish the WandB run
    wandb.finish()


if __name__ == "__main__":
    main()
