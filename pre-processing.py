from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


TARGET_CANDIDATES = ["GradeClass", "Class"]
DROP_FEATURE_COLUMNS = ["StudentID"]


@dataclass
class PreprocessResult:
    raw_shape: tuple[int, int]
    cleaned_shape: tuple[int, int]
    target_column: str
    feature_columns: list[str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    X_train_scaled: pd.DataFrame
    X_test_scaled: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    scaler: StandardScaler
    feature_encoders: dict[str, LabelEncoder]
    target_encoder: LabelEncoder | None
    medians: dict[str, float]
    class_counts_before: pd.Series
    class_counts_after: pd.Series
    missing_before: pd.Series
    missing_after: pd.Series
    logs: list[str]
    metadata: dict[str, Any]


def choose_target_column(df: pd.DataFrame) -> str:
    for target in TARGET_CANDIDATES:
        if target in df.columns:
            return target
    return df.columns[-1]


def _can_stratify(y: pd.Series) -> bool:
    counts = y.value_counts()
    return len(counts) > 1 and bool((counts >= 2).all())


def _balance_by_undersampling(df: pd.DataFrame, target_col: str, random_state: int) -> pd.DataFrame:
    counts = df[target_col].value_counts()
    min_count = int(counts.min())
    if min_count <= 0:
        return df
    sampled_groups = [
        group.sample(n=min_count, random_state=random_state)
        for _, group in df.groupby(target_col, sort=False)
    ]
    return pd.concat(sampled_groups, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)


def preprocess_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    balance_classes: bool = True,
    scale_features: bool = True,
    handle_missing: bool = True,
    encode_categorical: bool = True,
    on_log: Callable[[str], None] | None = None,
) -> PreprocessResult:
    logs: list[str] = []

    def log(msg: str) -> None:
        logs.append(msg)
        if on_log is not None:
            on_log(msg)

    working = df.copy()
    raw_shape = working.shape
    log(f"Loaded dataset with {raw_shape[0]:,} rows and {raw_shape[1]:,} columns.")

    target_col = choose_target_column(working)
    log(f"Selected target column: '{target_col}'.")

    missing_before = working.isna().sum()
    target_missing = int(working[target_col].isna().sum())
    if target_missing:
        working = working.dropna(subset=[target_col]).copy()
        log(f"Dropped {target_missing:,} rows with missing target values.")
    else:
        log("No missing target values found.")

    feature_cols = [col for col in working.columns if col != target_col and col not in DROP_FEATURE_COLUMNS]
    X = working[feature_cols].copy()
    y = working[target_col].copy()

    feature_encoders: dict[str, LabelEncoder] = {}
    medians: dict[str, float] = {}
    encoded_columns: list[str] = []

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            missing_count = int(X[col].isna().sum())
            median = float(X[col].median()) if not X[col].dropna().empty else 0.0
            medians[col] = median
            if missing_count:
                if handle_missing:
                    X[col] = X[col].fillna(median)
                    log(f"[handle_missing] Filled {missing_count:,} NaN in '{col}' with median {median:.3f}.")
                else:
                    X = X.dropna(subset=[col])
                    y = y.loc[X.index]
                    log(f"[handle_missing=off] Dropped {missing_count:,} rows with NaN in '{col}'.")
        else:
            if encode_categorical:
                encoder = LabelEncoder()
                values = X[col].astype("string").fillna("Missing")
                X[col] = encoder.fit_transform(values)
                feature_encoders[col] = encoder
                medians[col] = float(pd.Series(X[col]).median())
                encoded_columns.append(col)
            else:
                log(f"[encode_categorical=off] Dropping non-numeric column '{col}'.")
                X = X.drop(columns=[col])

    if encoded_columns:
        log("[encode_categorical] Label-encoded: " + ", ".join(encoded_columns) + ".")
    elif encode_categorical:
        log("[encode_categorical] No non-numeric feature columns required label encoding.")

    target_encoder: LabelEncoder | None = None
    if not pd.api.types.is_numeric_dtype(y):
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y.astype("string")), index=y.index, name=target_col)
        log("Label-encoded non-numeric target column.")
    else:
        y = pd.to_numeric(y)

    prepared = X.copy()
    prepared[target_col] = y
    class_counts_before = prepared[target_col].value_counts().sort_index()

    if balance_classes and len(class_counts_before) > 1:
        min_count = int(class_counts_before.min())
        if min_count >= 2:
            prepared = _balance_by_undersampling(prepared, target_col, random_state)
            log(f"[balance] Undersampled classes to {min_count:,} rows each.")
        else:
            log("[balance] Skipped — a class has fewer than 2 rows.")
    else:
        log("[balance] Skipped — disabled or single class.")

    X_balanced = prepared.drop(columns=[target_col])
    y_balanced = prepared[target_col]
    class_counts_after = y_balanced.value_counts().sort_index()

    stratify = y_balanced if _can_stratify(y_balanced) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    log(f"Split → {len(X_train):,} train rows / {len(X_test):,} test rows (test_size={test_size:.0%}).")

    scaler = StandardScaler()
    if scale_features:
        X_train_values = scaler.fit_transform(X_train)
        X_test_values = scaler.transform(X_test)
        log("[scale] Applied StandardScaler to train and test features.")
    else:
        X_train_values = X_train.to_numpy()
        X_test_values = X_test.to_numpy()
        log("[scale] Feature scaling skipped.")

    X_train_scaled = pd.DataFrame(X_train_values, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_values, columns=X_test.columns, index=X_test.index)
    missing_after = X_balanced.isna().sum()

    log("Preprocessing complete.")

    metadata = {
        "target_column": target_col,
        "feature_columns": list(X_balanced.columns),
        "class_labels": sorted(y_balanced.unique().tolist()),
        "feature_encoders": feature_encoders,
        "target_encoder": target_encoder,
        "medians": medians,
        "scaler": scaler,
        "scale_features": scale_features,
    }

    return PreprocessResult(
        raw_shape=raw_shape,
        cleaned_shape=prepared.shape,
        target_column=target_col,
        feature_columns=list(X_balanced.columns),
        X_train=X_train,
        X_test=X_test,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        feature_encoders=feature_encoders,
        target_encoder=target_encoder,
        medians=medians,
        class_counts_before=class_counts_before,
        class_counts_after=class_counts_after,
        missing_before=missing_before,
        missing_after=missing_after,
        logs=logs,
        metadata=metadata,
    )
