# Loan Modeling Project

This repository contains EDA, preprocessing, and model training code for predicting loan outcomes based on loan posting data.

This README documents how the training script `model_training.py` works, how to run it, and how to load the saved model artifacts.

## Files of interest
- `eda.py` - EDA and preprocessing helper (creates cleaned DataFrame, encoders, and can standardize numeric columns).
- `eda_visualization.ipynb` - Jupyter notebook with exploratory plots and interactive EDA.
- `model_training.py` - Training script that trains multiple classifiers and saves artifacts (models + metadata).
- `models/` - output folder where trained models and metadata are written.

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
- `models/<ModelName>.joblib` — the joblib artifact containing `{'pipeline': Pipeline, 'features': [...column names...]}`. The pipeline includes preprocessing and the classifier.
- `models/<ModelName>_metadata.json` — small JSON with python & scikit-learn versions, timestamp, and features.
- `models/best_model.joblib` and `models/best_model_metadata.json` — the selected model and metadata.

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

or

```bash
uv run python model_training.py
```

The script will read `loan_data.csv` by default and write models into the `models/` folder.

## Load a saved model for inference

There is a small helper script `model_validation.py` included to quickly sanity-check a saved model. It:

- Loads `models/best_model.joblib` (expects keys `pipeline` and `features`).
- Builds a single-row pandas DataFrame from a sample record, fills any missing features with NaN so the pipeline imputers can handle them, and reorders columns to match the trained features.
- Runs `pipeline.predict` and (if available) `pipeline.predict_proba`, then prints results.

Use this sample CSV / row values as a test input (single row):

```
person_age,person_gender,person_education,person_income,person_emp_exp,person_home_ownership,loan_amnt,loan_intent,loan_int_rate,loan_percent_income,cb_person_cred_hist_length,credit_score,previous_loan_defaults_on_file
34.0,female,Bachelor,97265.0,11,MORTGAGE,15000.0,PERSONAL,12.73,0.15,9.0,631,No
```

Run the validator:

```bash
python model_validation.py
```

Notes:
- Ensure `models/best_model.joblib` exists (run `model_training.py` first).
- Categorical values must match those used during training (case/spelling).
- If preprocessing was performed outside the saved pipeline, apply identical transforms before calling `predict`.

## Details for deployment
- The model will be deployed on Google Cloud using one of two methods:
  - Vertex AI with a prebuilt Scikit-learn container for managed deployment and easy endpoint creation
  - Cloud Run with a custom Docker container for greater flexibility and automatic scaling to zero during idle periods

## Next improvements

- Add a `predict_server.py` or FastAPI app for containerized serving.

## 📊 Data Source & License

The dataset used in this project originates from [Kaggle](https://www.kaggle.com/), titled **"Loan Approval Classification Dataset"**, and is provided under the **Apache 2.0 License**.

- **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)
- **License:** [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **Usability Score:** 10.00
- **Expected Update Frequency:** Never

This repository does not claim ownership of the dataset. It is used solely for educational and research purposes.
