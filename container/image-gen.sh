# Set variables for your GCP project and Artifact Registry repository
PROJECT_ID=$(gcloud config get-value project)
REGION=australia-southeast1
REPO_NAME=my-xgboost-repo  

# Enable the Artifact Registry API if it’s not already enabled
gcloud services enable artifactregistry.googleapis.com

# Create a Docker repository in Artifact Registry
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker \
    --location=$REGION

# Build your Docker image
docker build --platform=linux/amd64 -t $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/xgboost-train:latest .

# Authenticate Docker to your Google Artifact Registry
gcloud auth configure-docker $REGION-docker.pkg.dev

# Push the Docker image
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/xgboost-train:latest
