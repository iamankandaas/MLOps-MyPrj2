import os
import json
import argparse
import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from src.logger import logging # Assuming a custom logger setup

# ==========================
# CONFIGURATION
# ==========================
CONFIG = {
    "dagshub_repo_owner": "das.99.ankan",
    "dagshub_repo_name": "MLOps-MyPrj2",
    "experiment_name": "DVC_Pipeline_Evaluation"
}

# ==========================
# SETUP
# ==========================
def setup_mlflow():
    """Initializes MLflow for tracking using production-safe environment variables."""
    
    # --- Production-Ready MLflow Setup ---

    # Set up DagsHub credentials for MLflow tracking from environment variables
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
        raise EnvironmentError("The environment variable 'CAPSTONE_TEST' is not set. Please provide your DagsHub token.")

    # Set environment variables for MLflow to use for authentication
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    # Construct the MLflow tracking URI
    repo_owner = CONFIG["dagshub_repo_owner"]
    repo_name = CONFIG["dagshub_repo_name"]
    mlflow_tracking_uri = f'https://dagshub.com/{repo_owner}/{repo_name}.mlflow'
    
    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Set the experiment name
    mlflow.set_experiment(CONFIG["experiment_name"])
    
    logging.info(f"MLflow setup for production complete. Tracking to: {mlflow_tracking_uri}")

# ==========================
# HELPER FUNCTIONS
# ==========================
def load_local_model(file_path: str):
    """Loads a model from a local .pkl file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info(f"Model loaded successfully from local path: {file_path}")
        return model
    except FileNotFoundError:
        logging.error(f"Model file not found at: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading model from {file_path}: {e}")
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from a specified CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise

def evaluate_model(model, X_test, y_test) -> dict:
    """Calculates evaluation metrics for the given model and data."""
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        logging.info(f"Model evaluation metrics calculated: {metrics}")
        return metrics
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise

def save_json_output(data: dict, file_path: str):
    """Saves a dictionary to a local JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
        logging.info(f"JSON data successfully saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving JSON to {file_path}: {e}")
        raise

# ==========================
# MAIN EXECUTION LOGIC
# ==========================
def main(model_path: str, test_data_path: str, metrics_output_path: str, experiment_info_path: str):
    """Main function to run the model evaluation pipeline."""
    setup_mlflow()
    
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logging.info(f"Starting evaluation run {run_id} for local model: {model_path}")
        
        mlflow.log_param("evaluated_model_path", model_path)
        
        try:
            # 1. Define the artifact path for the model
            model_artifact_path = "model"

            # 2. Load model and data
            model = load_local_model(model_path)
            test_data = load_data(test_data_path)
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values
            
            # 3. Evaluate the model
            metrics = evaluate_model(model, X_test, y_test)
            
            # 4. Create the experiment info dictionary with both keys
            experiment_info = {
                'run_id': run_id,
                'model_path': model_artifact_path # This is the crucial addition
            }
            
            # 5. Save metrics and experiment info locally
            save_json_output(metrics, metrics_output_path)
            save_json_output(experiment_info, experiment_info_path)
            
            # 6. Log everything to the MLflow run
            mlflow.log_metrics(metrics)
            
            if hasattr(model, 'get_params'):
                mlflow.log_params(model.get_params())
                
            # Log the model using the defined artifact path
            mlflow.sklearn.log_model(model, model_artifact_path)
            
            # Log the output files as artifacts
            mlflow.log_artifact(metrics_output_path)
            mlflow.log_artifact(experiment_info_path)
            
            logging.info("Model evaluation process completed successfully.")

        except Exception as e:
            logging.error(f"Failed to complete the model evaluation process: {e}")
            print(f"An error occurred. Check logs for details. Error: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a local .pkl model and log results to MLflow.")
    
    parser.add_argument("--model_path", type=str, default="./models/model.pkl",
                        help="Path to the local .pkl model file.")
    
    parser.add_argument("--test_data", type=str, default="./data/processed/test_bow.csv", 
                        help="Path to the test data CSV file.")
                        
    parser.add_argument("--metrics_output", type=str, default="reports/metrics.json", 
                        help="Path to save the output metrics JSON file.")
    
    # Restored the argument to its original name for clarity
    parser.add_argument("--experiment_info_output", type=str, default="reports/experiment_info.json", 
                        help="Path to save the evaluation run_id and model_path for pipeline use.")
    
    args = parser.parse_args()
    
    main(
        model_path=args.model_path,
        test_data_path=args.test_data,
        metrics_output_path=args.metrics_output,
        experiment_info_path=args.experiment_info_output
    )