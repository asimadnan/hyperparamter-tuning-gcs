import os
import argparse
from string import Template
import subprocess

# Set the folder for saving job YAML files
job_folder = "jobs"
os.makedirs(job_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Define your project ID
PROJECT_ID = "boston-house-price-439411"  # Replace with your actual project ID
IMAGE_URI = "australia-southeast1-docker.pkg.dev/boston-house-price-439411/my-xgboost-repo/xgboost-train"

# Hyperparameter ranges (for demonstration, you can adjust the values)
param_ranges = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0]
}

# Template for Kubernetes job YAML with placeholders for dynamic substitution
job_template = Template("""\
apiVersion: batch/v1
kind: Job
metadata:
  name: xgboost-train-$index
spec:
  template:
    spec:
      serviceAccountName: my-service-account
      containers:
      - name: xgboost
        image: $IMAGE_URI  
        command: ["python", "train.py"]
        args: [
          "--n_estimators", "$n_estimators",
          "--max_depth", "$max_depth",
          "--learning_rate", "$learning_rate",
          "--subsample", "$subsample",
          "--job_name", "xgboost-train-$index"
        ]
        env:
        - name: GOOGLE_APPLICATION_CREDENTIALS
          value: /secrets/gcp-key/key.json
        volumeMounts:
        - name: gcp-key-volume
          mountPath: /secrets/gcp-key
          readOnly: true
      volumes:
      - name: gcp-key-volume
        secret:
          secretName: gcp-key
      restartPolicy: Never
  backoffLimit: 4
""")

# Parse arguments to allow for testing mode
parser = argparse.ArgumentParser(description="Generate and submit Kubernetes jobs for XGBoost hyperparameter tuning.")
parser.add_argument("--test", action="store_true", help="Generate only 3 jobs for testing purposes.")
args = parser.parse_args()

# Generate YAML for each hyperparameter configuration and save to job folder
job_count = 0
max_jobs = 2 if args.test else None  # Limit jobs to 3 if in testing mode

for n_estimators in param_ranges["n_estimators"]:
    for max_depth in param_ranges["max_depth"]:
        for learning_rate in param_ranges["learning_rate"]:
            for subsample in param_ranges["subsample"]:
                if max_jobs and job_count >= max_jobs:
                    break  # Stop if test mode and job limit reached

                # Fill in the template with actual values
                job_yaml = job_template.substitute(
                    index=job_count,
                    IMAGE_URI=IMAGE_URI,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                )

                # Save the job YAML to a file
                job_filename = os.path.join(job_folder, f"xgboost-train-{job_count}.yaml")
                with open(job_filename, "w") as f:
                    f.write(job_yaml)

                # Submit the job to Kubernetes
                subprocess.run(["kubectl", "apply", "-f", job_filename])

                job_count += 1

print(f"Generated and submitted {job_count} Kubernetes Jobs (limited to 3 for testing)." if args.test else f"Generated and submitted {job_count} Kubernetes Jobs.")
