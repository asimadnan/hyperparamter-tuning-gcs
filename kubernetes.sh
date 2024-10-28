# Set variables for your GCP project and Artifact Registry repository
PROJECT_ID=$(gcloud config get-value project)
REGION=australia-southeast1
REPO_NAME=my-xgboost-repo  

gsutil iam ch serviceAccount:565132369705-compute@developer.gserviceaccount.com:objectAdmin gs://boston-house-price


gcloud container clusters create xgboost-cluster \
    --region australia-southeast1 \
    --machine-type "e2-standard-2" \
    --num-nodes "1" \
    --enable-autoscaling \
    --min-nodes "1" \
    --max-nodes "5" \
    --disk-type "pd-standard" \
    --project $PROJECT_ID



gcloud container clusters get-credentials xgboost-cluster --region australia-southeast1
