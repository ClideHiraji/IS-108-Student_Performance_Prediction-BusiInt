# Student Performance BI System

A Streamlit-based BI and machine learning dashboard for student performance analysis. The app supports dataset loading, preprocessing, model training, evaluation, and prediction using KNN, SVM, and ANN classifiers.

## Features

- Upload a CSV or Excel dataset.
- Load the included synthetic sample dataset for testing.
- Inspect dataset schema, column types, missing values, and GradeClass distribution.
- Preprocess data with missing-value handling, categorical encoding, scaling, and optional class balancing.
- Train KNN, SVM, and ANN models.
- Review evaluation metrics, confusion matrices, and training charts.
- Run single-student and batch predictions.

## Project Files

- `app.py` - Main Streamlit application.
- `dataset_handling.py` - Dataset loading, schema, and summary helpers.
- `pre-processing.py` - Preprocessing pipeline.
- `prediction.py` - Prediction helpers and category mappings.
- `train_models.py` - Standalone model-training script.
- `synthetic_student_performance.csv` - Bundled sample dataset.
- `Logo.png` - App logo.
- `requirements.txt` - Python dependencies.

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run The App

```powershell
streamlit run app.py
```

Then open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Dataset Notes

The **Load Bundled Sample Dataset** button loads `synthetic_student_performance.csv`. This is real demo data processing, but it is not user-uploaded data.

You can also upload your own `.csv`, `.xlsx`, or `.xls` file. For the full workflow, the dataset should include a target column such as `GradeClass`.

## Workflow

1. Open the app.
2. Upload a dataset or load the bundled sample dataset.
3. Review the dataset preview and charts.
4. Run preprocessing.
5. Train models.
6. Evaluate results.
7. Make predictions.
