from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

try:
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.models import Sequential

    TF_AVAILABLE = True
    TF_IMPORT_ERROR = ""
except Exception as exc:
    EarlyStopping = None
    Dense = None
    Dropout = None
    Input = None
    Sequential = None
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = str(exc)


def build_ann(input_dim: int, class_count: int) -> Any:
    if not TF_AVAILABLE:
        raise RuntimeError(f"TensorFlow unavailable: {TF_IMPORT_ERROR}")

    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(96, activation="relu"),
        Dropout(0.25),
        Dense(48, activation="relu"),
        Dropout(0.15),
        Dense(class_count, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def ann_predict_classes(model: Any, X: pd.DataFrame, class_labels: list) -> np.ndarray:
    probs = model.predict(X, verbose=0)
    indices = np.argmax(probs, axis=1)
    return np.array([class_labels[i] for i in indices])


def score_model(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def _make_knn_model(
    feature_cols: list[str],
    k: int,
    weights: str,
    metric: str,
) -> Pipeline:
    return Pipeline([
        ("features", ColumnTransformer(
            [("selected", "passthrough", feature_cols)],
            remainder="drop",
        )),
        ("knn", KNeighborsClassifier(
            n_neighbors=k,
            weights=weights,
            metric=metric,
        )),
    ])


def _knn_feature_sets(columns: list[str]) -> list[tuple[str, list[str]]]:
    available_cols = set(columns)
    feature_sets = [("All features", columns)]

    focused = [col for col in ["GPA", "StudyTimeWeekly", "Absences"] if col in available_cols]
    if len(focused) == 3:
        feature_sets.insert(0, ("GPA + study + absences", focused))

    return feature_sets


def _train_knn(
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    y_tr: pd.Series,
    y_te: pd.Series,
    k_max: int,
    log: Callable[[str], None],
) -> tuple[Pipeline, np.ndarray, dict[str, float], dict[str, list[float] | list[int]]]:
    k_list = list(range(1, k_max + 1, 2))
    if k_max not in k_list:
        k_list.append(k_max)

    log(f"KNN - tuning k in {k_list} with feature-set search ...")

    best_score = -1.0
    best_info: dict[str, Any] = {}
    best_train_curve: list[float] = []
    best_test_curve: list[float] = []
    knn_stratify = y_tr if bool((y_tr.value_counts() >= 2).all()) else None
    X_knn_fit, X_knn_val, y_knn_fit, y_knn_val = train_test_split(
        X_tr,
        y_tr,
        test_size=0.2,
        random_state=42,
        stratify=knn_stratify,
    )

    for feature_name, feature_cols in _knn_feature_sets(list(X_tr.columns)):
        log(f"  feature set: {feature_name} ({len(feature_cols)} column(s))")
        for weights in ("uniform", "distance"):
            for metric in ("euclidean", "manhattan"):
                val_curve = []
                for k in k_list:
                    candidate = _make_knn_model(feature_cols, k, weights, metric)
                    candidate.fit(X_knn_fit, y_knn_fit)
                    val_acc = accuracy_score(y_knn_val, candidate.predict(X_knn_val))
                    val_curve.append(val_acc)

                combo_best_index = int(np.argmax(val_curve))
                combo_best_score = val_curve[combo_best_index]
                if combo_best_score > best_score:
                    best_score = combo_best_score
                    best_info = {
                        "k": k_list[combo_best_index],
                        "weights": weights,
                        "metric": metric,
                        "feature_name": feature_name,
                        "feature_cols": feature_cols,
                        "validation": combo_best_score,
                    }

                log(
                    f"    {weights:8s}/{metric:9s}  "
                    f"best validation={max(val_curve):.4f}"
                )

    if not best_info:
        raise RuntimeError("KNN tuning failed to find a valid model.")

    for k in k_list:
        curve_model = _make_knn_model(
            best_info["feature_cols"],
            k,
            best_info["weights"],
            best_info["metric"],
        )
        curve_model.fit(X_tr, y_tr)
        best_train_curve.append(accuracy_score(y_tr, curve_model.predict(X_tr)))
        best_test_curve.append(accuracy_score(y_te, curve_model.predict(X_te)))

    knn = _make_knn_model(
        best_info["feature_cols"],
        best_info["k"],
        best_info["weights"],
        best_info["metric"],
    )
    knn.fit(X_tr, y_tr)
    predictions = knn.predict(X_te)
    result = score_model(y_te, predictions)
    curve = {"k_range": k_list, "train": best_train_curve, "test": best_test_curve}

    log(
        "KNN done - accuracy "
        f"{result['Accuracy']:.4f}  "
        f"(k={best_info['k']}, {best_info['weights']}, {best_info['metric']}, "
        f"{best_info['feature_name']}, validation={best_info['validation']:.4f})."
    )

    return knn, predictions, result, curve


def _train_svm(
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    y_tr: pd.Series,
    y_te: pd.Series,
    kernel: str,
    c_value: float,
    log: Callable[[str], None],
) -> tuple[SVC, np.ndarray, dict[str, float], dict[str, list[float] | list[int]]]:
    n = len(X_tr)
    fracs = [0.2, 0.4, 0.6, 0.8, 1.0]
    sizes = [max(int(n * f), 10) for f in fracs]
    Xtr_a, ytr_a = X_tr.values, y_tr.values
    Xte_a, yte_a = X_te.values, y_te.values

    log(f"SVM - learning curve  kernel={kernel}  C={c_value} ...")
    svm_tr, svm_te = [], []
    for size in sizes:
        partial = SVC(kernel=kernel, C=c_value, probability=False, random_state=42)
        partial.fit(Xtr_a[:size], ytr_a[:size])
        svm_tr.append(accuracy_score(ytr_a[:size], partial.predict(Xtr_a[:size])))
        svm_te.append(accuracy_score(yte_a, partial.predict(Xte_a)))
        log(f"  n={size:4d}  train={svm_tr[-1]:.4f}  test={svm_te[-1]:.4f}")

    svm = SVC(kernel=kernel, C=c_value, probability=True, random_state=42)
    svm.fit(X_tr, y_tr)
    predictions = svm.predict(X_te)
    result = score_model(y_te, predictions)
    curve = {"sizes": sizes, "train": svm_tr, "test": svm_te}
    log(f"SVM done - accuracy {result['Accuracy']:.4f}.")

    return svm, predictions, result, curve


def _train_ann(
    prep: Any,
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    y_tr: pd.Series,
    y_te: pd.Series,
    epochs: int,
    batch_size: int,
    log: Callable[[str], None],
) -> tuple[Any | None, np.ndarray | None, dict[str, float] | None, Any | None]:
    if not TF_AVAILABLE:
        log("ANN skipped - TensorFlow unavailable.")
        return None, None, None, None

    class_labels = sorted(y_tr.unique().tolist())
    class_to_index = {label: index for index, label in enumerate(class_labels)}
    y_tr_idx = y_tr.map(class_to_index)
    log(f"ANN - epochs={epochs}  batch={batch_size} ...")

    ann = build_ann(X_tr.shape[1], len(class_labels))
    callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    history = ann.fit(
        X_tr,
        y_tr_idx,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    prep.metadata["ann_classes"] = class_labels
    predictions = ann_predict_classes(ann, X_te, class_labels)
    result = score_model(y_te, predictions)
    log(f"ANN done - accuracy {result['Accuracy']:.4f} after {len(history.history['loss'])} epochs.")

    return ann, predictions, result, history


def train_models(prep: Any, params: dict, on_log: Callable[[str], None] | None = None):
    X_tr, X_te = prep.X_train_scaled, prep.X_test_scaled
    y_tr, y_te = prep.y_train, prep.y_test
    logs: list[str] = []

    def log(msg: str) -> None:
        logs.append(msg)
        if on_log:
            on_log(msg)

    models: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    results: dict[str, dict[str, float]] = {}
    ann_history = None

    log(f"Data ready - {len(X_tr):,} train / {len(X_te):,} test / {X_tr.shape[1]} features.")
    log(f"Classes: {', '.join(str(label) for label in sorted(y_tr.unique().tolist()))}.")

    knn, knn_pred, knn_result, knn_curve = _train_knn(
        X_tr,
        X_te,
        y_tr,
        y_te,
        params["knn_k"],
        log,
    )
    models["KNN"] = knn
    predictions["KNN"] = knn_pred
    results["KNN"] = knn_result

    svm, svm_pred, svm_result, svm_curve = _train_svm(
        X_tr,
        X_te,
        y_tr,
        y_te,
        params["svm_kernel"],
        params["svm_c"],
        log,
    )
    models["SVM"] = svm
    predictions["SVM"] = svm_pred
    results["SVM"] = svm_result

    ann, ann_pred, ann_result, ann_history = _train_ann(
        prep,
        X_tr,
        X_te,
        y_tr,
        y_te,
        params["ann_epochs"],
        params["ann_batch"],
        log,
    )
    if ann is not None and ann_pred is not None and ann_result is not None:
        models["ANN"] = ann
        predictions["ANN"] = ann_pred
        results["ANN"] = ann_result

    log("All models trained successfully.")
    return models, results, predictions, ann_history, logs, knn_curve, svm_curve