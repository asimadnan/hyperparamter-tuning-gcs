# XGBoost Training & Deployment Pipeline on Google Cloud Platform (GCP)

This project provides an end-to-end pipeline for training, hyperparameter tuning, and deploying an XGBoost model on Google Cloud Platform. It leverages Kubernetes for parallelized hyperparameter tuning and Vertex AI for deployment, with model artifacts and metrics stored in Google Cloud Storage (GCS).



## Project Structure
```plaintext
├── README.md
├── check_job_metrics.py         # Script to monitor Kubernetes jobs and select the best model based on MSE
├── container
│   ├── Dockerfile               # Docker configuration file for building the container image
│   ├── entrypoint.sh            # Script to handle train or predict mode for Docker container
│   ├── image-gen.sh             # Script to build and push the Docker image to GCS Artifact Registry
│   ├── predict.py               # Script for generating predictions from the trained model
│   ├── requirements.txt         # Python dependencies
│   └── train.py                 # Script to train the model, save artifacts to GCS, and log to W&B
├── generate_jobs.py             # Script to generate Kubernetes job YAMLs for hyperparameter tuning
├── jobs                         # Directory containing Kubernetes job YAML files
├── kubernetes.sh                # Script to set up a Kubernetes cluster on GCP
├── deploy.py                   # Script to deploy best model

```

# Setup and Prerequisites

### GCP Account and Project:
- Ensure you have a Google Cloud account and a new project. Due to quota limitations on new accounts, Vertex AI hyperparameter tuning may not be available.
- Google Cloud SDK: Install and authenticate using gcloud for command-line interactions with GCP resources.
- Docker: Install Docker for containerization.
- Weights & Biases (W&B): Set up W&B for logging metrics and monitoring training processes.

# Step-by-Step Guide
1. Training and Prediction Scripts
train.py: Loads data, receives hyperparameters as arguments, and trains an XGBoost model. The model and metrics are saved to the GCS bucket, with logs also sent to W&B.
predict.py: Loads the trained model from GCS and generates predictions.
2. Entrypoint Script (entrypoint.sh)
This script serves as the container entrypoint, accepting train or predict as arguments to invoke the corresponding scripts.

3. Install Dependencies (requirements.txt)
Include necessary Python packages in requirements.txt. This will be referenced in the Dockerfile for building the environment.

4. Building and Pushing the Docker Image
image-gen.sh builds and pushes the Docker image to Google Artifact Registry.


``` 
bash container/image-gen.sh 
```

5. Kubernetes Cluster Setup
kubernetes.sh sets up a Kubernetes cluster in GCP with autoscaling enabled. The cluster will be used for running hyperparameter tuning jobs.

```
bash kubernetes.sh
```

6. Generate and Run Hyperparameter Tuning Jobs
generate_jobs.py creates Kubernetes job YAML files for various hyperparameter combinations (permutations) and applies them to the Kubernetes cluster. Each job trains a model with a unique set of hyperparameters.

```
python generate_jobs.py
```

7. Checking Model Performance and Selecting the Best Model
check_job_metrics.py parses Kubernetes job logs to find the Mean Squared Error (MSE) for each job and identifies the job with the lowest error. It then locates the corresponding model artifact in GCS.

```
python check_job_metrics.py
```

8. Deploying the Model to Vertex AI
Once the best model is identified, a deployment script deplpy.py deploys the selected model artifact as an endpoint in Vertex AI for online predictions.
```
python deploy.py
```

### Some Notes:
- Due to quota restrictions on new Google Cloud accounts, I was not able to use Vertex AI hyperparameter tuning. 
- Ensure that the Kubernetes cluster has permissions to write to the GCS bucket.
