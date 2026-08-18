# Customer Segmentation MLOps Pipeline

An end-to-end machine learning pipeline that ingests customer data from MongoDB, trains a CatBoost classifier to predict customer clusters, pushes the trained model to cloud storage, and serves live predictions through a FastAPI web app.

## Overview

This project automates the full lifecycle of a customer segmentation model — from raw data in a database to a live prediction endpoint — instead of relying on a one-off notebook run. Hit a single API endpoint and the entire pipeline runs: data is pulled from MongoDB, validated, transformed, clustered, and used to train a model, which is then evaluated and pushed to cloud storage. A second endpoint serves real-time predictions through a custom web form.

## Architecture

```
MongoDB Atlas  →  FastAPI (/train)  →  Data Validation  →  Data Transformation
                                              ↓
                                    Unsupervised KMeans Clustering
                                              ↓
                                   CatBoost Classifier Training
                                    (RandomizedSearchCV tuning)
                                              ↓
                                       Model Evaluation
                                              ↓
                              Backblaze B2 (S3-compatible) Cloud Storage
                                              ↓
                          FastAPI (/) → Prediction Pipeline → Web Form UI
```

## Tech Stack

- **API Framework:** FastAPI
- **Database:** MongoDB Atlas
- **ML Model:** CatBoost (classifier), scikit-learn (KMeans clustering, preprocessing pipeline)
- **Hyperparameter Tuning:** RandomizedSearchCV via `neuro_mf`
- **Cloud Storage:** Backblaze B2 (S3-compatible object storage, accessed via `boto3`)
- **Frontend:** Custom HTML/CSS form (Jinja2 templates)
- **Data Validation / Drift Detection:** Evidently

## Features

- **Automated retraining** — a single `GET /train` call pulls the latest data from MongoDB and retrains the model end-to-end, no manual notebook re-runs needed
- **Data validation & drift detection** — schema checks and dataset drift reporting before training proceeds
- **Cloud-backed model storage** — trained models are pushed to Backblaze B2, decoupling model training from wherever the app happens to be running
- **Live prediction UI** — a styled web form takes 21 customer attributes and returns a predicted cluster in real time
- **Config-driven pipeline** — dataset schema, model hyperparameters, and prediction input schema are all defined in YAML config files, not hardcoded

## Project Structure

```
├── app.py                          # FastAPI application entry point
├── config/
│   ├── schema.yaml                 # Dataset schema (columns, drop list)
│   ├── model.yaml                  # Model + hyperparameter search config
│   └── prediction_schema.yaml      # Input schema for prediction requests
├── src/
│   ├── components/                 # Pipeline stages (ingestion, validation, transformation, training, evaluation, pusher)
│   ├── configuration/              # MongoDB & cloud storage connection setup
│   ├── constant/                   # Centralized constants (paths, bucket names, env var keys)
│   ├── data_access/                # MongoDB data access layer
│   ├── entity/                     # Config & artifact dataclasses
│   ├── ml/model/                   # Model wrapper classes, cloud model estimator
│   ├── pipeline/                   # Training & prediction pipeline orchestration
│   └── utils/                      # Shared utility functions
├── templates/
│   └── customer.html               # Prediction form UI
├── notebook/                       # Exploratory data analysis & feature engineering
└── requirements.txt
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/avipluto/customer-Categorizer.git
cd customer-Categorizer
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```
MONGO_DB_URL=your_mongodb_connection_string

AWS_ACCESS_KEY_ID=your_backblaze_keyID
AWS_SECRET_ACCESS_KEY=your_backblaze_applicationKey
AWS_S3_ENDPOINT_URL=https://s3.<region>.backblazeb2.com
AWS_DEFAULT_REGION=<region>
```

> This project uses [Backblaze B2](https://www.backblaze.com/cloud-storage) as an S3-compatible alternative to AWS S3 — `boto3` is pointed at Backblaze's endpoint via `AWS_S3_ENDPOINT_URL`.

### 4. Run the app

```bash
python app.py
```

The API will be available at `http://127.0.0.1:8080`.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/train` | Runs the full training pipeline: ingestion → validation → transformation → clustering → training → evaluation → cloud upload |
| `GET` | `/` | Renders the customer data input form |
| `POST` | `/` | Accepts form input, runs the prediction pipeline, and returns the predicted cluster |
| `GET` | `/test_env` | Sanity-checks that environment variables are loading correctly |

## Model Details

- **Clustering:** customers are grouped into segments via unsupervised KMeans (post-PCA dimensionality reduction) based on demographic and spending behavior features
- **Classification:** a CatBoost classifier is trained to predict cluster assignment for new customers, tuned via `RandomizedSearchCV`
- **Features used:** 21 attributes spanning demographics (age, education, marital status, income), spending behavior (wine, meat, fish, sweets, gold purchases), and engagement metrics (tenure, recency, web/store/catalog purchase counts)

## Roadmap

- [ ] Containerize the application with Docker
- [ ] Deploy to a cloud compute platform for public access
- [ ] Add automated tests for the pipeline components

## Author

Built by [Aviral Yadav](https://www.linkedin.com/in/aviral-yadav-b484b4306/) — B.Tech CSE student, ML/CV Founding Engineer at Skynox Sentinels, Outreach Lead at the NVIDIA AI & Supercomputing Club.
