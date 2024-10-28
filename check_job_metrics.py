import os
import time
import subprocess
import yaml
import re
from deploy import deploy_model_from_gcs_folder  # Assuming the filename is deploy_model.py

# Define the folder containing job YAML files
job_folder = "jobs"

def wait_for_job_completion(job_name):
    """Wait for a Kubernetes job to complete and return its status."""
    while True:
        # Get the job status
        result = subprocess.run(
                ["kubectl", "get", "jobs", job_name, "-o", "jsonpath={.status.conditions[?(@.type=='Complete')] }"],
            capture_output=True, text=True
        )
        status = result.stdout.strip()
        # print(f"status : {status}, result : {result}")

        # Check if the job is complete or failed
        if "True" in status:
            print(f"Job {job_name} completed successfully.")
            return "Completed"
        elif "False" in status:
            print(f"Job {job_name} failed.")
            return "Failed"
        
        print(f"Job {job_name} is still running...")
        time.sleep(10)  # Wait for a while before checking again

def extract_metrics_from_logs(job_name):
    """Extract metrics from the logs of a completed job."""
    # Get the pod name for the job
    pod_result = subprocess.run(
        ["kubectl", "get", "pods", "--selector=job-name=" + job_name, "-o", "jsonpath={.items[0].metadata.name}"],
        capture_output=True, text=True
    )
    pod_name = pod_result.stdout.strip()

    # Check logs of the pod
    log_result = subprocess.run(
        ["kubectl", "logs", pod_name],
        capture_output=True, text=True
    )

    # Extract RMSE and MSE from logs using regex
    log_output = log_result.stdout
    rmse_match = re.search(r"RMSE:\s*([0-9.]+)", log_output)
    mse_match = re.search(r"MSE:\s*([0-9.]+)", log_output)

    if rmse_match and mse_match:
        rmse = float(rmse_match.group(1))
        mse = float(mse_match.group(1))
        return {"rmse": rmse, "mse": mse}

    return None

        


def main():
    best_metrics = None
    best_config = None

    # Iterate over all job files in the folder
    for job_file in os.listdir(job_folder):
        if job_file.endswith(".yaml"):
            job_name = job_file.split('.')[0]  # Get job name from filename

            # Wait for job completion
            status = wait_for_job_completion(job_name)

            if status == "Completed":
                # Extract metrics from logs
                metrics = extract_metrics_from_logs(job_name)
                if metrics:
                    print(f"Metrics for job {job_name}: RMSE={metrics['rmse']}, MSE={metrics['mse']}")

                    # Check if this is the best configuration
                    if best_metrics is None or metrics['rmse'] < best_metrics['rmse']:
                        best_metrics = metrics
                        best_config = job_name

            print(f"Finished processing job {job_name}.")

    # Print the best configuration at the end
    if best_config:
        print(f"Best job: {best_config} with RMSE={best_metrics['rmse']} and MSE={best_metrics['mse']}")
        print(f"Deploying the best model from GCS to Vertex AI Endpoint...")
        deploy_model_from_gcs_folder(best_config)
    else:
        print("No jobs completed successfully.")

if __name__ == "__main__":
    main()
