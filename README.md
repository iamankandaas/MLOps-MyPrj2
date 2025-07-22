End-to-End Sentiment Analysis MLOps Pipeline

This repository contains a complete, end-to-end MLOps project that automates the training, evaluation, deployment, and monitoring of a sentiment analysis model. The entire workflow is orchestrated through a CI/CD pipeline, deploying the final application as a containerized service on Amazon EKS.

Overview

The core of this project is a Logistic Regression model trained to classify text as either 'positive' or 'negative'. The project's main goal is not just to build a model, but to build a robust, automated system around it. This includes versioning data, tracking experiments, containerizing the application, and deploying it to a scalable Kubernetes cluster, complete with monitoring for performance and health.

Key Features

Automated CI/CD Pipeline: Uses GitHub Actions to automatically test, build, and deploy the application on every code push.

Data & Pipeline Versioning: Leverages DVC to version large data files and define the ML pipeline stages (dvc.yaml).

Experiment Tracking: Integrates with MLflow (hosted on DagsHub) to log model parameters, metrics, and artifacts for every run.

Containerization: The final Flask application is containerized using Docker and pushed to Amazon ECR (Elastic Container Registry).

Cloud-Native Deployment: Deploys the containerized application to a Kubernetes cluster managed by Amazon EKS (Elastic Kubernetes Service).

Infrastructure as Code: Uses eksctl to programmatically create and manage the EKS cluster and its resources.

Application Monitoring: A Prometheus server scrapes custom metrics from the Flask application's /metrics endpoint, and Grafana is set up for visualization.

Project Workflow & Architecture

The project follows a standard MLOps workflow, triggered by a git push to the main branch.

CI Trigger: The GitHub Actions workflow begins.

Setup & Install: A Python environment is set up, and all dependencies from requirements.txt are installed. The local src package is also installed.

ML Pipeline Execution: dvc repro is run, which executes the stages defined in dvc.yaml:

data_ingestion: Downloads and splits the raw data.

data_preprocessing & feature_engineering: Cleans the text data and converts it into numerical features (BoW).

model_building: Trains the Logistic Regression model.

model_evaluation: Evaluates the model on the test set and logs metrics to MLflow.

register_model: Registers the newly trained model in the MLflow Model Registry and promotes it to the "Staging" stage.

Testing & Promotion:

Automated tests are run against the model.

A script promotes the model from "Staging" to "Production" in the registry.

Containerization & Push:

A Docker image of the Flask application is built.

The image is tagged and pushed to a private Amazon ECR repository.

Kubernetes Deployment:

kubectl apply updates the Kubernetes cluster with the new deployment configuration, pointing to the latest Docker image in ECR.

EKS pulls the new image and performs a rolling update, ensuring zero downtime.

Monitoring:

The live application exposes metrics at its /metrics endpoint.

A separate Prometheus server, running on an EC2 instance, scrapes this endpoint to collect data.

A Grafana server, also on an EC2 instance, connects to Prometheus as a data source to visualize application performance.

Challenges & Key Learnings

Deploying a real-world MLOps pipeline presents numerous challenges. This project was no exception, and overcoming these hurdles provided critical insights.

1. Environment and PATH Conflicts

Challenge: Initial attempts to run eksctl and aws commands failed with 'is not recognized' errors, even after installation.

Diagnosis: Using the where <command> utility revealed that multiple versions of these tools were installed on the system (e.g., one from a manual install in System32, another by chocolatey). The system PATH was prioritizing the outdated version.

Learning: Always ensure your system's PATH is clean and points to a single, correct installation of a command-line tool. Restarting the terminal after installation or PATH modification is crucial for changes to take effect.

2. Cloud Service and Local Tool Version Mismatch

Challenge: The eksctl create cluster command failed, reporting that the requested Kubernetes version (e.g., 1.25) was unsupported by AWS EKS. However, attempting to use a newer version (e.g., 1.29) resulted in an error from the eksctl tool itself.

Diagnosis: The version of the eksctl executable was too old and did not know how to provision modern Kubernetes versions, even though the cloud service required it.

Learning: Infrastructure-as-code tools like eksctl must be kept up-to-date to remain compatible with the ever-evolving cloud provider APIs. A simple choco upgrade eksctl resolved the conflict.

3. EKS Pod Deployment Failure (ImagePullBackOff)

Challenge: After a seemingly successful deployment, kubectl get pods showed the application pods were stuck in an ImagePullBackOff or ErrImagePull state. The curl command to the load balancer returned an empty reply.

Diagnosis: This was a multi-step debugging process:

The first assumption was a missing IAM permission. The AmazonEC2ContainerRegistryReadOnly policy was correctly attached to the EKS node's IAM role.

When the error persisted, kubectl describe pod <pod-name> was used to get a more detailed error message.

The detailed event logs revealed a 403 Forbidden error and, crucially, showed the exact image URI the pod was trying to pull.

Comparing this URI with the output of aws ecr describe-repositories revealed the root cause: the AWS Account ID in the deployment.yaml was incorrect.

Learning: ImagePullBackOff is a generic error. kubectl describe pod is the most critical tool for diagnostics, as it provides the specific underlying reason (e.g., 403 Forbidden, manifest not found). Always verify that the image URI in your Kubernetes deployment manifest is a 100% exact match with the image URI in your container registry.

Project Visuals
![After selecting Logistic Regression as the primary algorithm, this MLflow experiment was run to fine-tune its hyperparameters. This parallel coordinates plot visualizes how different values for the regularization parameter (C) and the penalty (l1 vs. l2) impact the model's final accuracy. The color of each line corresponds to the accuracy score, clearly highlighting the combination of parameters that yielded the best-performing model for deployment.](screenshots/MLflow Hyperparameter Tuning for Logistic Regression.png)

![This screenshot shows the MLflow Parallel Coordinates Plot, which was essential for the initial model selection phase of the project. This powerful visualization compares multiple experiment runs at once, testing different classification algorithms (such as XGBoost, Logistic Regression, and Random Forest) against different text feature engineering techniques (BoW and TF-IDF). Each line represents a single training run. The lines are colored by the accuracy metric, making it easy to identify the top-performing model-feature combinations at a glance. This chart was crucial for quickly narrowing down the options and justifying the choice of the final model that was carried forward for deployment.](screenshots/MLflow Model & Feature Comparison.png)

![Data Versioning](screenshots/Data Versioning.png)

![This screenshot shows the MLflow UI, hosted on DagsHub, which serves as the central hub for experiment tracking. The DVC_Pipeline_Evaluation experiment is specifically for runs triggered by the automated dvc repro pipeline. Each row represents a complete, end-to-end training and evaluation cycle, ensuring that every model produced is logged, versioned (my_model v1, v2), and fully reproducible.](screenshots/MLflow Automated Experiment Tracking.png)

![This screenshot showcases the local development environment in Visual Studio Code, with the project's file structure on the left and a terminal on the right. The terminal displays the output of the dvc repro command, which orchestrates the entire machine learning pipeline. The logs confirm that all stages ran successfully, culminating in the registration of a new model (version 3) in the MLflow Model Registry and its automatic promotion to the "Staging" environment.](screenshots/Local DVC Pipeline Execution and Model Registration.png)

![This screenshot shows the final Flask application running locally inside a Docker container. Before pushing the image to Amazon ECR and deploying it to the cloud, this local test validates that the application, the MLflow model loading, and the prediction logic are all working correctly together. Here, the model has successfully processed a text input and classified it as 'Positive Sentiment,' confirming the application is ready for the next stage of deployment.](screenshots/Local Test of the Dockerized Application.png)

![To enable effective monitoring, the Flask application was instrumented using the Prometheus client library. This screenshot shows the raw output of the /metrics endpoint, which exposes custom telemetry in a format that the Prometheus server can understand and scrape. Key metrics being tracked include app_request_count (total requests per endpoint), app_request_latency_seconds (a histogram of request durations), and model_prediction_count (a counter for each class predicted), providing deep insight into the application's real-time behavior.](screenshots/Exposing Custom Application Metrics.png)

![This screenshot from the GitHub Actions tab shows the history of the CI/CD pipeline runs. It clearly visualizes the iterative process required to build and debug a complex automation workflow. The initial runs failed due to issues ranging from dependency conflicts and Python pathing errors to missing cloud credentials. Each failure provided a clue, leading to a fix in the workflow configuration or application code, culminating in the first fully successful end-to-end pipeline run (CICD run - 7).](screenshots/First CICD success.png)

![This screenshot shows the mature and complete GitHub Actions workflow. The pipeline has been extended beyond local testing and now includes steps to build a Docker image, push it to Amazon ECR, and deploy it automatically to the Amazon EKS cluster. The final successful run, "EKS Deployment - 1," represents the fully automated, end-to-end MLOps process, moving from a code change directly to a live, containerized application in the cloud.](screenshots/First EKS success.png)

![Connecting to the Prometheus Data Source](screenshots/Connecting to the Prometheus Data Source.png)

![This screenshot shows the final, deployed sentiment analysis application running live on an Amazon EKS cluster. The URL in the address bar is the public endpoint of an AWS Elastic Load Balancer, which routes traffic to the containerized Flask application. Here, the model has successfully processed an input and returned a 'Negative Sentiment' prediction, demonstrating that the entire MLOps pipeline—from CI/CD automation to cloud deployment—is fully functional.](screenshots/Live Application on AWS EKS.png)

![This screenshot showcases the final Grafana dashboard, which provides real-time visualization of the live application's performance. The dashboard queries the Prometheus data source to display key metrics, including a graph of the model_prediction_count and another for Requests to home route. This setup allows for immediate insight into application health and usage patterns, completing the full MLOps cycle from development to production monitoring.](screenshots/Grafana Dashboards.png)

![This screenshot shows the Prometheus server's web UI, which is running on a dedicated EC2 instance. A query has been executed for the custom model_prediction_count_total metric. The result confirms that Prometheus is successfully scraping the /metrics endpoint of the live application (identified by its AWS Load Balancer instance URL) and is collecting valuable data, such as the count of 'negative' predictions (prediction="0"). This is the core of the project's monitoring stack.](screenshots/Prometheus Monitoring Live Application Metrics.png)
