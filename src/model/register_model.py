import os
import json
import argparse
import mlflow
import mlflow.tracking
import dagshub
from src.logger import logging # Assuming a custom logger setup

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# ==========================
# CONFIGURATION
# ==========================
CONFIG = {
    "mlflow_tracking_uri": "https://dagshub.com/das.99.ankan/MLOps-MyPrj2.mlflow",
    "dagshub_repo_owner": "das.99.ankan",
    "dagshub_repo_name": "MLOps-MyPrj2",
    "registered_model_name": "my_model"  # The name for all model versions
}

# ==========================
# SETUP
# ==========================
def setup_mlflow():
    """Initializes MLflow and DagsHub for tracking."""
    mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
    dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"], repo_name=CONFIG["dagshub_repo_name"], mlflow=True)
    logging.info("MLflow and DagsHub have been set up for model registration.")

# ==========================
# HELPER FUNCTIONS
# ==========================
def load_experiment_info(file_path: str) -> dict:
    """Loads run and model info from the specified JSON file."""
    try:
        with open(file_path, 'r') as file:
            experiment_info = json.load(file)
        logging.info(f"Experiment info loaded successfully from {file_path}")
        return experiment_info
    except FileNotFoundError:
        logging.error(f"Experiment info file not found at: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading experiment info from {file_path}: {e}")
        raise

def register_and_promote_model(model_name: str, experiment_info: dict):
    """
    Registers the model from the specified run and promotes it to the 'Staging' stage.
    """
    model_uri = f"runs:/{experiment_info['run_id']}/{experiment_info['model_path']}"
    logging.info(f"Attempting to register model from URI: {model_uri}")

    try:
        # Register the model, creating a new version
        model_version_details = mlflow.register_model(model_uri, model_name)
        version = model_version_details.version
        print(f"Successfully registered model '{model_name}' with new version: {version}")

        # Use the MlflowClient to transition the new version to the "Staging" stage
        client = mlflow.tracking.MlflowClient()
        
        print(f"Transitioning version {version} to 'Staging'...")
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging",
            archive_existing_versions=True # Optional: Archive older versions in "Staging"
        )
        print(f"Successfully transitioned model '{model_name}' version {version} to 'Staging'.")
        logging.info(f"Model {model_name} version {version} is now in Staging.")
        
    except Exception as e:
        logging.error(f"Error during model registration or promotion: {e}")
        raise

# ==========================
# MAIN EXECUTION LOGIC
# ==========================
def main(info_file_path: str):
    """Main function to run the model registration process."""
    setup_mlflow()
    
    try:
        # 1. Load the pointer to the model we want to register
        experiment_info = load_experiment_info(info_file_path)
        
        # 2. Register the model under the predefined name and promote it
        register_and_promote_model(CONFIG["registered_model_name"], experiment_info)

        print("\nModel registration process completed successfully.")

    except Exception as e:
        logging.error(f"Failed to complete the model registration process: {e}")
        print(f"\nAn error occurred. Check logs for details. Error: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Register a model to the MLflow Model Registry and promote it to Staging.")
    
    parser.add_argument("--experiment_info", type=str, default="reports/experiment_info.json",
                        help="Path to the experiment_info.json file from the evaluation stage.")
    
    args = parser.parse_args()
    
    main(info_file_path=args.experiment_info)

