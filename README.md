# 🚀 End-to-End Sentiment Analysis MLOps Pipeline

This repository showcases a complete, end-to-end MLOps project that automates the training, evaluation, deployment, and monitoring of a sentiment analysis model. The workflow is fully orchestrated via a CI/CD pipeline and deployed as a containerized service on Amazon EKS.

---

## 📌 Overview

At the core of this project is a **Logistic Regression** model that classifies text as *positive* or *negative*. However, the emphasis lies not just in model accuracy, but in building a **robust MLOps infrastructure** around it, including:

- ✅ Data versioning  
- ✅ Experiment tracking  
- ✅ Docker-based deployment  
- ✅ Kubernetes orchestration  
- ✅ Real-time monitoring with Prometheus and Grafana  

---

## 🔧 Key Features

- **CI/CD Pipeline:** GitHub Actions automates testing, building, and deployment.
- **Data & Pipeline Versioning:** DVC tracks data files and orchestrates ML pipeline stages.
- **Experiment Tracking:** MLflow (hosted on DagsHub) logs all model runs and metrics.
- **Containerization:** Flask application is Dockerized and pushed to Amazon ECR.
- **Cloud Deployment:** Kubernetes manifests deploy the app on Amazon EKS with rolling updates.
- **Infrastructure as Code:** `eksctl` automates EKS provisioning.
- **Monitoring:** Prometheus scrapes metrics, Grafana visualizes system behavior.

---

## 🔁 Workflow & Architecture

The end-to-end pipeline is triggered by a `git push`:

1. **CI/CD Triggered** via GitHub Actions.
2. **Python Environment Setup** + `pip install -r requirements.txt`.
3. **ML Pipeline Execution** via `dvc repro`:
   - `data_ingestion`: Download/split raw data
   - `data_preprocessing` & `feature_engineering`: Clean + transform text
   - `model_building`: Train Logistic Regression
   - `model_evaluation`: Evaluate & log to MLflow
   - `register_model`: Register & promote model

4. **Testing & Promotion**: Tests + automatic promotion from *Staging* to *Production*.
5. **Containerization & Push**: Docker image is built and pushed to ECR.
6. **Kubernetes Deployment**: `kubectl apply` triggers a rolling update on EKS.
7. **Monitoring**:
   - Prometheus scrapes `/metrics` endpoint.
   - Grafana visualizes metrics from Prometheus.

---

## ⚠️ Challenges & Learnings

### 1. Environment & PATH Conflicts
- ❌ `eksctl`/`aws` not recognized due to multiple PATH entries
- ✅ Resolved by cleaning up PATH and restarting terminal

### 2. Cloud Tool Version Mismatch
- ❌ `eksctl create cluster` failed due to outdated binary
- ✅ Fixed using `choco upgrade eksctl`

### 3. ImagePullBackOff During Pod Deployment
- ❌ Kubernetes pods failed to pull image due to incorrect AWS Account ID in `deployment.yaml`
- ✅ `kubectl describe pod` revealed the actual 403 error
- ✅ Corrected image URI resolved the issue

---

## 📸 Project Visuals

### 🔍 MLflow Hyperparameter Tuning
After selecting Logistic Regression, an MLflow experiment was run to tune hyperparameters.

![MLflow Hyperparameter Tuning](screenshots/MLflow%20Hyperparameter%20Tuning%20for%20Logistic%20Regression.png)

---

### 📊 Model & Feature Comparison
This parallel coordinates plot visualizes accuracy across algorithms (LogReg, XGBoost, RF) and features (BoW, TF-IDF).

![Model & Feature Comparison](screenshots/MLflow%20Model%20%26%20Feature%20Comparison.PNG)

---

### 📁 Data Versioning with DVC
Data versioning and reproducibility tracked via DVC.

![Data Versioning](screenshots/Data%20Versioning.png)

---

### 🔁 MLflow Experiment Tracking
Each DVC-triggered run is tracked and versioned on MLflow (hosted on DagsHub).

![MLflow Automated Tracking](screenshots/MLflow%20Automated%20Experiment%20Tracking.png)

---

### 🛠️ Local DVC Pipeline Execution
Successful `dvc repro` shows all stages and model promotion to *Staging*.

![DVC Pipeline Execution](screenshots/Local%20DVC%20Pipeline%20Execution%20and%20Model%20Registration.png)

---

### 🧪 Local Dockerized App Test
The app correctly classifies a sample input before being pushed to ECR.

![Dockerized App Test](screenshots/Local%20Test%20of%20the%20Dockerized%20Application.png)

---

### 📈 Exposing Custom Metrics
The `/metrics` endpoint is exposed using Prometheus client lib.

![Custom Metrics](screenshots/Exposing%20Custom%20Application%20Metrics.png)

---

### 🔁 CI/CD Pipeline - First Success
The first successful GitHub Actions run after debugging and fixes.

![First CICD Success](screenshots/First%20CICD%20success.png)

---

### ☁️ EKS Deployment
ECR push and EKS deployment automated via CI.

![First EKS Success](screenshots/First%20EKS%20success.png)

---

### 📡 Prometheus Integration
Prometheus added as a data source in Grafana.

![Prometheus Connection](screenshots/Connecting%20to%20the%20Prometheus%20Data%20Source.png)

---

### 🌐 Live Application on AWS
The app live on an AWS Load Balancer—predicts sentiment in real-time!

![Live App](screenshots/Live%20Application%20on%20AWS%20EKS.png)

---

### 📊 Grafana Dashboard
Real-time monitoring of app requests and predictions.

![Grafana Dashboard](screenshots/Grafana%20Dashboards.png)

---

### 📡 Prometheus Scraping Live Metrics
Prometheus server querying metrics from `/metrics` endpoint.

![Prometheus Live Monitoring](screenshots/Prometheus%20Monitoring%20Live%20Application%20Metrics.png)

---

## 💡 Final Note

This project demonstrates a complete MLOps cycle from model development to live cloud deployment, with automation, reproducibility, monitoring, and real-time feedback.  
Had great fun — and a tough time too 😅 — completing this project! XD
