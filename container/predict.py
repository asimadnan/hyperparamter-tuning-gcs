import json
import joblib
import numpy as np
import sys

# Load the model once when the script starts
model = joblib.load("model.joblib")


def predict(input_data):
    # Convert input_data to the format expected by the model
    features = np.array(
        [
            [
                input_data["CRIM"],
                input_data["ZN"],
                input_data["INDUS"],
                input_data["CHAS"],
                input_data["NOX"],
                input_data["RM"],
                input_data["AGE"],
                input_data["DIS"],
                input_data["RAD"],
                input_data["TAX"],
                input_data["PTRATIO"],
                input_data["B"],
                input_data["LSTAT"],
            ]
        ]
    )

    # Generate predictions
    return model.predict(features).tolist()


if __name__ == "__main__":
    # Read input from stdin (which is how Vertex AI will send data)
    input_data = json.loads(sys.stdin.read())
    predictions = predict(input_data)

    # Print output in the expected JSON format
    print(json.dumps({"predictions": predictions}))
