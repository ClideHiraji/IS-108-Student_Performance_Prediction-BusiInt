from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


GRADE_LABELS = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "F",
}

CATEGORY_OPTIONS = {
    "Gender": {"Male": 0, "Female": 1},
    "Ethnicity": {
        "Caucasian": 0,
        "African American": 1,
        "Asian": 2,
        "Other": 3,
    },
    "ParentalEducation": {
        "None": 0,
        "High School": 1,
        "Some College": 2,
        "Bachelor's": 3,
        "Higher": 4,
    },
    "Tutoring": {"No": 0, "Yes": 1},
    "ParentalSupport": {
        "None": 0,
        "Low": 1,
        "Moderate": 2,
        "High": 3,
        "Very High": 4,
    },
    "Extracurricular": {"No": 0, "Yes": 1},
    "Sports": {"No": 0, "Yes": 1},
    "Music": {"No": 0, "Yes": 1},
    "Volunteering": {"No": 0, "Yes": 1},
}


def grade_name(value: Any, target_encoder: Any | None = None) -> str:
    if target_encoder is not None:
        try:
            value = target_encoder.inverse_transform([int(value)])[0]
        except Exception:
            pass
    try:
        numeric = int(value)
        return f"{GRADE_LABELS.get(numeric, str(value))} ({numeric})"
    except Exception:
        return str(value)


def _safe_label_transform(series: pd.Series, encoder: Any) -> pd.Series:
    classes = list(encoder.classes_)
    fallback = classes[0]
    mapping = {value: index for index, value in enumerate(classes)}
    values = series.astype("string").fillna("Missing")
    return values.map(lambda value: mapping.get(value, mapping[fallback])).astype(int)


def prepare_features(input_df: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    feature_cols = metadata["feature_columns"]
    missing = [col for col in feature_cols if col not in input_df.columns]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))

    X = input_df[feature_cols].copy()
    encoders = metadata.get("feature_encoders", {})
    medians = metadata.get("medians", {})

    for col in X.columns:
        if col in encoders:
            X[col] = _safe_label_transform(X[col], encoders[col])
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].fillna(float(medians.get(col, 0.0)))

    scaler = metadata["scaler"]
    values = scaler.transform(X)
    return pd.DataFrame(values, columns=feature_cols, index=input_df.index)


def predict_model(model: Any, model_name: str, scaled_features: pd.DataFrame, metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray | None]:
    if model_name == "ANN":
        probabilities = model.predict(scaled_features, verbose=0)
        class_labels = metadata.get("ann_classes", metadata.get("class_labels", []))
        indices = np.argmax(probabilities, axis=1)
        predictions = np.array([class_labels[index] for index in indices])
        confidence = np.max(probabilities, axis=1)
        return predictions, confidence

    predictions = model.predict(scaled_features)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled_features)
        confidence = np.max(probabilities, axis=1)
    else:
        confidence = None
    return np.asarray(predictions), confidence


def predict_single(input_row: dict[str, Any], model: Any, model_name: str, metadata: dict[str, Any]) -> dict[str, Any]:
    input_df = pd.DataFrame([input_row])
    scaled = prepare_features(input_df, metadata)
    predictions, confidence = predict_model(model, model_name, scaled, metadata)
    predicted_value = predictions[0]
    return {
        "prediction": predicted_value,
        "label": grade_name(predicted_value, metadata.get("target_encoder")),
        "confidence": None if confidence is None else float(confidence[0]),
    }


def batch_predict(input_df: pd.DataFrame, model: Any, model_name: str, metadata: dict[str, Any]) -> pd.DataFrame:
    scaled = prepare_features(input_df, metadata)
    predictions, confidence = predict_model(model, model_name, scaled, metadata)
    output = input_df.copy()
    output["PredictedClass"] = predictions
    output["PredictedGrade"] = [
        grade_name(value, metadata.get("target_encoder")) for value in predictions
    ]
    if confidence is not None:
        output["Confidence"] = np.round(confidence, 4)
    return output
