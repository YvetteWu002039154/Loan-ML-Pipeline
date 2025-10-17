# Loan Modeling Project

This repository contains EDA, preprocessing, and model training code for predicting loan outcomes based on loan posting data.

This README documents how the training script `model_training.py` works, how to run it, and how to load the saved model artifacts.

## Files of interest
- `eda.py` - EDA and preprocessing helper (creates cleaned DataFrame, encoders, and can standardize numeric columns).
- `eda_visualization.ipynb` - Jupyter notebook with exploratory plots and interactive EDA.
- `model_training.py` - Training script that trains multiple classifiers, evaluates with cross-validation, and saves artifacts in `models/`.
- `model_validation.py` - Small helper to load a saved model and run a single-row prediction for sanity checks.
- `models/` - output folder where trained models and metadata are written (saved as `.joblib` + `_metadata.json`).
- `Deployment/` - a minimal serving example (contains `app.py`, `Dockerfile`, and a `requirements.txt`) you can adapt for containerized serving or Cloud Run / Vertex AI custom container deployment.

## model_training.py — overview

Purpose: load raw CSV, build a preprocessing pipeline (numeric imputation + scaling, categorical one-hot encoding), train several classifiers, evaluate them with cross-validation, and save the trained artifacts.

Key behaviors:
- Loads CSV via `load_df(csv_path)` (default `loan_data.csv`).
- Builds a `ColumnTransformer` preprocessor with:
  - numeric pipeline: `SimpleImputer(strategy='median')` + `StandardScaler()`
  - categorical pipeline: `SimpleImputer(strategy='constant', fill_value='MISSING')` + `OneHotEncoder(handle_unknown='ignore')`
- Classifiers trained: Logistic Regression, SVM, KNN, simple ANN (`MLPClassifier`), RandomForest.
- Evaluation: the script computes accuracy, precision, recall, F1, and ROC-AUC (where applicable). It also runs cross-validation (StratifiedKFold) and reports mean/std for the CV scoring metric (default `f1_macro`).
- Model selection: models are ranked by cross-validation mean (or F1 if CV is unavailable) and the top model is saved as `models/best_model.joblib` with a JSON metadata file.

## What the script saves
- `models/<ModelName>.joblib` — the joblib artifact containing the saved pipeline (preprocessing + estimator) and metadata keys (we save the pipeline object and the `features` list).
- `models/<ModelName>_metadata.json` — small JSON with python & scikit-learn versions, timestamp, and features.
- `models/best_model.joblib` and `models/best_model_metadata.json` — the selected best model and metadata.

Why joblib: joblib is recommended for scikit-learn artifacts because it efficiently serializes NumPy arrays and supports compression.

## How to run

1. Create / activate your Python environment and install dependencies (example):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install scikit-learn pandas numpy joblib
```

If you have uv you can also run:
```bash
    uv sync
```

2. Run training (from this project root):

```bash
python model_training.py
```

or (if you use `uv` helper):

```bash
uv run python model_training.py
```

The script will read `loan_data.csv` (or the CSV you supply) and write models into the `models/` folder.

## Load a saved model for inference

There is a helper script `model_validation.py` to sanity-check a saved model. It:

- Loads `models/best_model.joblib` (expects keys `pipeline` and `features`).
- Builds a single-row pandas DataFrame from a sample record, fills any missing features with NaN so the pipeline imputers can handle them, and reorders columns to match the trained features.
- Runs `pipeline.predict` and (if available) `pipeline.predict_proba`, then prints results.

Example usage:

```bash
python model_validation.py
```

Make sure the categorical values in your sample match training text (case/spelling). If any preprocessing was done outside the saved pipeline (not recommended), you must reproduce it prior to prediction.

## Details for deployment (Deployment/)

This repo includes a `Deployment/` folder with a minimal FastAPI app and Dockerfile to help you build a container to serve the saved pipeline. Two common options:

- Vertex AI: Ipload the trained model (model.joblib) to the Vertex AI Model Registry and deployed as an endpoint for online prediction
- Cloud Run: build and push the Docker image that uses `Deployment/app.py` (FastAPI) and `Deployment/requirements.txt`.

Quick serve locally (in the virtualenv):

```bash
pip install -r Deployment/requirements.txt
uvicorn Deployment.app:app --host 0.0.0.0 --port 8080
```

The service accepts JSON POST requests (see `Deployment/app.py`) and internally converts incoming arrays to DataFrames using the saved `features` list so the pipeline can run reliably.

## API Access

The trained Random Forest model has been deployed as a FastAPI application using Google Cloud Run.
It provides a public REST API endpoint for real-time loan approval prediction.

### How to Test

```bash
curl -X POST "https://loan-api-963580054894.us-central1.run.app/predict" \
-H "Content-Type: application/json" \
-d @sample_payload.json
```

### Endpint

```bash
https://loan-api-963580054894.us-central1.run.app/predict
```

### Sample payload

```json
{
  "instances": [
    {
      "person_age": 22.0,
      "person_gender": "male",
      "person_education": "Master",
      "person_income": 66135.0,
      "person_emp_exp": 1,
      "person_home_ownership": "RENT",
      "loan_amnt": 35000.0,
      "loan_intent": "MEDICAL",
      "loan_int_rate": 14.27,
      "loan_percent_income": 0.53,
      "cb_person_cred_hist_length": 4.0,
      "credit_score": 586,
      "previous_loan_defaults_on_file": "NO"
    },
    {
      "person_age": 22.0,
      "person_gender": "female",
      "person_education": "Master",
      "person_income": 71948.0,
      "person_emp_exp": 0,
      "person_home_ownership": "RENT",
      "loan_amnt": 35000.0,
      "loan_intent": "PERSONAL",
      "loan_int_rate": 16.02,
      "loan_percent_income": 0.49,
      "cb_person_cred_hist_length": 3.0,
      "credit_score": 561,
      "previous_loan_defaults_on_file": "NO"
    }
  ]
}
```

### Example Response

```json
{
  "predictions": [0, 1]
}
```

## 📊 Data Source & License

The dataset used in this project originates from [Kaggle](https://www.kaggle.com/), titled **"Loan Approval Classification Dataset"**, and is provided under the **Apache 2.0 License**.

- **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)
- **License:** [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **Usability Score:** 10.00
- **Expected Update Frequency:** Never

This repository does not claim ownership of the dataset. It is used solely for educational and research purposes.
