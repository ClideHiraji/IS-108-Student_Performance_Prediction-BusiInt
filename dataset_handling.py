from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO

import pandas as pd


FEATURE_COLS = [
    "Age",
    "Gender",
    "Ethnicity",
    "ParentalEducation",
    "StudyTimeWeekly",
    "Absences",
    "Tutoring",
    "ParentalSupport",
    "Extracurricular",
    "Sports",
    "Music",
    "Volunteering",
    "GPA",
]

ID_COLUMNS = ["StudentID"]
TARGET_CANDIDATES = ["GradeClass", "Class"]

GRADE_LABELS = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "F",
}

CATEGORY_LABELS = {
    "Gender": {0: "Male", 1: "Female"},
    "Ethnicity": {
        0: "Caucasian",
        1: "African American",
        2: "Asian",
        3: "Other",
    },
    "ParentalEducation": {
        0: "None",
        1: "High School",
        2: "Some College",
        3: "Bachelor's",
        4: "Higher",
    },
    "Tutoring": {0: "No", 1: "Yes"},
    "ParentalSupport": {
        0: "None",
        1: "Low",
        2: "Moderate",
        3: "High",
        4: "Very High",
    },
    "Extracurricular": {0: "No", 1: "Yes"},
    "Sports": {0: "No", 1: "Yes"},
    "Music": {0: "No", 1: "Yes"},
    "Volunteering": {0: "No", 1: "Yes"},
}

COLUMN_DESCRIPTIONS = [
    ("StudentID", "Unique identifier for each student, from 1001 to 6000."),
    ("Age", "Student age, from 15 to 18."),
    ("Gender", "0 = Male, 1 = Female."),
    ("Ethnicity", "0 = Caucasian, 1 = African American, 2 = Asian, 3 = Other."),
    ("ParentalEducation", "0 = None, 1 = High School, 2 = Some College, 3 = Bachelor's, 4 = Higher."),
    ("StudyTimeWeekly", "Weekly study time in hours, from 0 to 20."),
    ("Absences", "Number of school absences, from 0 to 30."),
    ("Tutoring", "0 = No tutoring, 1 = Receives tutoring."),
    ("ParentalSupport", "0 = None, 1 = Low, 2 = Moderate, 3 = High, 4 = Very High."),
    ("Extracurricular", "0 = No, 1 = Participates in extracurricular activities."),
    ("Sports", "0 = No, 1 = Participates in sports."),
    ("Music", "0 = No, 1 = Participates in music."),
    ("Volunteering", "0 = No, 1 = Participates in volunteering."),
    ("GPA", "Grade Point Average on a 2.0 to 4.0 scale."),
    ("GradeClass", "Target class: 0 = A, 1 = B, 2 = C, 3 = D, 4 = F."),
]


@dataclass(frozen=True)
class DatasetInfo:
    rows: int
    columns: int
    missing_values: int
    duplicate_rows: int
    numeric_columns: int
    non_numeric_columns: int
    target_column: str


def load_dataset(uploaded_file: BinaryIO) -> pd.DataFrame:
    """Load a CSV or Excel file uploaded through Streamlit."""
    file_name = getattr(uploaded_file, "name", "").lower()
    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")


def choose_target_column(df: pd.DataFrame) -> str:
    for target in TARGET_CANDIDATES:
        if target in df.columns:
            return target
    return df.columns[-1]


def get_dataset_info(df: pd.DataFrame) -> DatasetInfo:
    target_column = choose_target_column(df)
    return DatasetInfo(
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        missing_values=int(df.isna().sum().sum()),
        duplicate_rows=int(df.duplicated().sum()),
        numeric_columns=int(df.select_dtypes(include="number").shape[1]),
        non_numeric_columns=int(df.select_dtypes(exclude="number").shape[1]),
        target_column=target_column,
    )


def get_dtype_table(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    return pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [str(dtype) for dtype in df.dtypes],
            "Missing": [int(missing[col]) for col in df.columns],
            "Unique": [int(df[col].nunique(dropna=True)) for col in df.columns],
        }
    )


def get_dictionary_table() -> pd.DataFrame:
    return pd.DataFrame(COLUMN_DESCRIPTIONS, columns=["Column", "Description"])


def get_grade_distribution(df: pd.DataFrame) -> pd.DataFrame:
    target = choose_target_column(df)
    counts = df[target].value_counts(dropna=False).sort_index()
    total = max(int(counts.sum()), 1)
    rows = []
    for raw_value, count in counts.items():
        label = GRADE_LABELS.get(int(raw_value), str(raw_value)) if pd.notna(raw_value) else "Missing"
        rows.append(
            {
                "Class": raw_value,
                "Grade": label,
                "Count": int(count),
                "Percent": round((int(count) / total) * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def get_schema_status(df: pd.DataFrame) -> pd.DataFrame:
    expected = ID_COLUMNS + FEATURE_COLS + ["GradeClass"]
    rows = []
    for column in expected:
        rows.append(
            {
                "Column": column,
                "Status": "Available" if column in df.columns else "Missing",
                "Role": "Target" if column == "GradeClass" else "Identifier" if column in ID_COLUMNS else "Feature",
            }
        )
    return pd.DataFrame(rows)
