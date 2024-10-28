import json
import logging
from google.cloud import aiplatform
from google.cloud import storage

# Configure logging
logging.basicConfig(level=logging.INFO)

def deploy_model_from_gcs_folder(folder_name):
    # Set your GCP project and bucket name
    project_id = 'boston-house-price-439411'
    bucket_name = 'boston-house-price'
    region = 'australia-southeast1'  # Set the desired region

    # Initialize the AI Platform client
    logging.info("Initializing AI Platform client.")
    aiplatform.init(project=project_id, location=region)

    # Set the GCS path
    gcs_model_path = f'gs://{bucket_name}/{folder_name}/'
    gcs_metrics_path = f'gs://{bucket_name}/{folder_name}/metrics.json'
    
    # Load metrics from metrics.json
    logging.info(f"Loading metrics from {gcs_metrics_path}.")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    metrics_blob = bucket.blob(f'{folder_name}/metrics.json')

    try:
        metrics_content = metrics_blob.download_as_text()
        metrics = json.loads(metrics_content)
        logging.info("Metrics loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load metrics: {e}")
        return

    # Prepare model metadata
    model_name = f'model-{folder_name}'
    endpoint_name = f'endpoint-{folder_name}'

    # Check if endpoint already exists
    logging.info(f"Checking if endpoint '{endpoint_name}' exists in region '{region}'.")
    existing_endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_name}"',
        order_by="create_time desc"
    )
    endpoint = None

    if existing_endpoints:
        endpoint = existing_endpoints[0]
        logging.info(f"Endpoint '{endpoint.display_name}' already exists. Updating with new model.")
    else:
        # Create a new endpoint if it does not exist
        logging.info(f"Creating endpoint '{endpoint_name}' in region '{region}'.")
        try:
            endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
            logging.info(f"Endpoint '{endpoint.display_name}' created.")
        except Exception as e:
            logging.error(f"Failed to create endpoint: {e}")
            return

    # Upload and deploy the model
    logging.info(f"Uploading model '{model_name}' to GCS path: {gcs_model_path}.")
    try:
        model = aiplatform.Model.upload(
            display_name=model_name,
            artifact_uri=gcs_model_path,
            serving_container_image_uri='australia-southeast1-docker.pkg.dev/boston-house-price-439411/my-xgboost-repo/xgboost-train:latest'
        )
        logging.info(f"Model '{model_name}' uploaded successfully.")
    except Exception as e:
        logging.error(f"Failed to upload model: {e}")
        return

    # Deploy the model to the endpoint, undeploying previous models if necessary
    logging.info(f"Deploying model '{model_name}' to endpoint '{endpoint.display_name}'.")
    try:
        # Undeploy any previous models to free up resources before deploying the new model
        for deployed_model in endpoint.list_models():
            logging.info(f"Undeploying model '{deployed_model.model}' from endpoint '{endpoint.display_name}'.")
            endpoint.undeploy(deployed_model_id=deployed_model.id)

        # Deploy the new model
        model.deploy(
            endpoint=endpoint,
            traffic_percentage=100,
        )
        logging.info(f'Model {model_name} deployed to endpoint {endpoint.display_name} successfully.')
    except Exception as e:
        logging.error(f"Failed to deploy model: {e}")
        return

    logging.info(f'Model {model_name} deployed to endpoint {endpoint.display_name} with metrics: {metrics}')

    # Example input for prediction with the specified features
    input_data = {
        "CRIM": 0.00632,
        "ZN": 18.00,
        "INDUS": 2.310,
        "CHAS": 0,
        "NOX": 0.5380,
        "RM": 6.5750,
        "AGE": 65.20,
        "DIS": 4.0900,
        "RAD": 1,
        "TAX": 296.0,
        "PTRATIO": 15.30,
        "B": 396.90,
        "LSTAT": 4.98
    }

    # Make a prediction using the deployed model
    logging.info(f"Making a prediction with the deployed model.")
    try:
        prediction = endpoint.predict(instances=[input_data])
        logging.info(f'Prediction: {prediction.predictions}')
    except Exception as e:
        logging.error(f"Failed to make a prediction: {e}")

# If this script is run directly, the function can be called like this:
if __name__ == "__main__":
    folder_name = 'xgboost-train-14'  # Specify the folder name here
    deploy_model_from_gcs_folder(folder_name)
