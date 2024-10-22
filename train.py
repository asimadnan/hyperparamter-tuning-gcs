import argparse
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# Parse hyperparameters from command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    return parser.parse_args()


def load_data():
    # Load the raw data
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

    # Combine the data into a structured format
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    # Create a DataFrame and add feature names
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

    # Create the DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Add the target variable (house prices) to the DataFrame
    df["MEDV"] = target
    return df


def main():
    args = parse_args()

    # Load your dataset (Boston house prices)
    df = load_data()
    X = df.drop(columns=["MEDV"])
    y = df["MEDV"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize the model with hyperparameters
    model = XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
    )

    # Train the model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    # Print metrics so Vertex AI can capture the result
    print(f"Mean Squared Error: {mse}")
    print(
        f"Hyperparameters: n_estimators={args.n_estimators}, max_depth={args.max_depth}, learning_rate={args.learning_rate}, subsample={args.subsample}"
    )

    # Save the model if you want to upload it later (optional)
    model.save_model("model.bst")


if __name__ == "__main__":
    main()
