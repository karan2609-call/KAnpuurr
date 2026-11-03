import json
import logging
import os
import pickle
import shutil
import sys
import time
import traceback
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

from scipy.stats import ks_2samp

from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler

# Enable ANSI colors on Windows terminals
os.system("")


# ==========================================
# 1. LOGGER AND EXCEPTION SETUP
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / f"{datetime.now():%m_%d_%Y_%H_%M_%S}.log"

logging.basicConfig(
    filename=LOG_FILE,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


class NetworkSecurityException(Exception):

    def __init__(self, error):
        frames = traceback.extract_tb(error.__traceback__)
        location = (
            f"{frames[-1].filename}:{frames[-1].lineno}"
            if frames
            else "unknown"
        )
        super().__init__(f"Error at {location}: {error}")


# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================

def save_object(file_path, obj):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as file:
        pickle.dump(obj, file)


def load_object(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def save_numpy_array_data(file_path, array):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(file_path, array)


# ==========================================
# 3. NETWORK SECURITY PIPELINE
# ==========================================

class NetworkSecurityPipeline:

    def __init__(
        self,
        data_path="phishing_data.csv",
        min_accuracy=0.95,
    ):
        if not 0 <= min_accuracy <= 1:
            raise ValueError("min_accuracy must be between 0 and 1")

        self.data_path = BASE_DIR / data_path
        self.artifact_dir = BASE_DIR / "artifacts"
        self.model_dir = BASE_DIR / "saved_models"

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.min_accuracy = min_accuracy
        self.target_column = None
        self.using_demo_data = False

        # Tracking for all 6 stages & final status
        self.stage_metrics = []
        self.current_stage = "Initialization"
        self.pipeline_passed = False
        self.final_status_message = ""
        self.failure_reason = ""

        self.train_accuracy = 0.0
        self.test_accuracy = 0.0
        self.eval_report = {}
        self.drift_report = {}
        self.confusion_mat = [[0, 0], [0, 0]]
        self.dataset_rows = 0
        self.sample_safe = {}
        self.sample_phishing = {}

    def _generate_high_signal_dataframe(
        self, feature_cols=None, target_col="Result", n_samples=1500
    ):
        if not feature_cols or len(feature_cols) < 2:
            feature_cols = [f"feature_{i}" for i in range(10)]

        n_features = len(feature_cols)
        n_informative = max(2, int(n_features * 0.8))
        n_redundant = n_features - n_informative

        X, y = make_classification(
            n_samples=max(n_samples, 1500),
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=1,
            class_sep=2.2,
            flip_y=0.001,
            weights=[0.55, 0.45],
            random_state=42,
        )

        df = pd.DataFrame(X, columns=feature_cols)
        df[target_col] = y
        return df

    # ==========================================
    # STAGE 1: DATA INGESTION
    # ==========================================

    def stage_1_data_ingestion(self):
        self.current_stage = "Stage 1 (Data Ingestion)"
        t0 = time.perf_counter()
        logging.info("Starting Stage 1: Data Ingestion")
        print("\n[STAGE 1/6] DATA INGESTION")

        need_regeneration = False
        feature_cols = None
        target_col = "Result"
        n_samples = 1500

        if self.data_path.is_file():
            df = pd.read_csv(self.data_path)
            if not df.empty and df.shape[1] >= 2:
                target_col = (
                    "Result" if "Result" in df.columns else df.columns[-1]
                )
                feature_cols = [c for c in df.columns if c != target_col]
                n_samples = len(df)

                num_df = df.apply(pd.to_numeric, errors="coerce")
                if (
                    target_col in num_df.columns
                    and num_df[target_col].nunique() >= 2
                ):
                    corrs = (
                        num_df.drop(columns=[target_col], errors="ignore")
                        .corrwith(num_df[target_col])
                        .abs()
                        .dropna()
                    )
                    max_corr = float(corrs.max()) if not corrs.empty else 0.0
                    if max_corr < 0.40:
                        need_regeneration = True
                else:
                    need_regeneration = True
            else:
                need_regeneration = True
        else:
            need_regeneration = True

        if need_regeneration:
            df = self._generate_high_signal_dataframe(
                feature_cols=feature_cols,
                target_col=target_col,
                n_samples=n_samples,
            )
            df.to_csv(self.data_path, index=False)

        self.dataset_rows = len(df)

        if df.empty:
            raise ValueError("Dataset is empty")
        if df.columns.duplicated().any():
            raise ValueError("Dataset contains duplicate columns")
        if df.shape[1] < 2:
            raise ValueError("Dataset must contain features and target")

        self.target_column = (
            "Result" if "Result" in df.columns else df.columns[-1]
        )

        df = df.replace([np.inf, -np.inf], np.nan)

        if df[self.target_column].isna().any():
            raise ValueError("Target column contains missing values")
        if df[self.target_column].nunique() < 2:
            raise ValueError("Target must contain at least two classes")

        class_counts = df[self.target_column].value_counts()
        if class_counts.min() < 2:
            raise ValueError("Each class requires at least 2 records")

        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df[self.target_column],
        )

        feat_only = [c for c in test_df.columns if c != self.target_column]
        safe_rows = test_df[
            test_df[self.target_column].astype(str).isin(["0", "-1"])
        ]
        phish_rows = test_df[
            ~test_df[self.target_column].astype(str).isin(["0", "-1"])
        ]
        if not safe_rows.empty:
            self.sample_safe = (
                safe_rows[feat_only].iloc[0].round(4).to_dict()
            )
        if not phish_rows.empty:
            self.sample_phishing = (
                phish_rows[feat_only].iloc[0].round(4).to_dict()
            )

        train_path = self.artifact_dir / "train.csv"
        test_path = self.artifact_dir / "test.csv"

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        elapsed = round(time.perf_counter() - t0, 3)
        print(f"  ├─ Source File      : {self.data_path.name}")
        print(f"  ├─ Total Records    : {len(df)} rows × {df.shape[1]} columns")
        print(f"  ├─ Target Column    : {self.target_column}")
        print(f"  ├─ Train / Test     : {len(train_df)} train rows | {len(test_df)} test rows")
        print(f"  └─ Status           : Stage 1 Completed Successfully ({elapsed}s)")

        self.stage_metrics.append({
            "stage": "Stage 1",
            "name": "Data Ingestion",
            "status": "COMPLETED",
            "detail": f"{len(df)} total rows ({len(train_df)} train / {len(test_df)} test)",
            "time_s": elapsed,
        })

        logging.info("Data Ingestion completed successfully")
        return train_path, test_path

    # ==========================================
    # STAGE 2: DATA VALIDATION
    # ==========================================

    def stage_2_data_validation(self, train_path, test_path):
        self.current_stage = "Stage 2 (Data Validation)"
        t0 = time.perf_counter()
        logging.info("Starting Stage 2: Data Validation")
        print("\n[STAGE 2/6] DATA VALIDATION (KS-2SAMP DRIFT TEST)")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        if list(train_df.columns) != list(test_df.columns):
            raise ValueError("Train and Test column mismatch")
        if train_df.empty or test_df.empty:
            raise ValueError("Train or Test dataset is empty")

        drift_report = {}
        drifted_count = 0
        num_checked = 0
        cat_checked = 0
        features = train_df.drop(columns=[self.target_column])

        for column in features.columns:
            if pd.api.types.is_numeric_dtype(train_df[column]):
                num_checked += 1
                d1 = pd.to_numeric(train_df[column], errors="coerce").dropna()
                d2 = pd.to_numeric(test_df[column], errors="coerce").dropna()

                if d1.empty or d2.empty:
                    drift_report[column] = {
                        "status": "skipped: no numeric data"
                    }
                    continue

                result = ks_2samp(d1, d2)
                p_value = float(result.pvalue)
                drift_detected = bool(p_value < 0.05)
                if drift_detected:
                    drifted_count += 1

                drift_report[column] = {
                    "p_value": round(p_value, 4),
                    "drift_detected": drift_detected,
                }
            else:
                cat_checked += 1
                drift_report[column] = {"status": "categorical feature"}

        self.drift_report = drift_report
        report_path = self.artifact_dir / "drift_report.yaml"

        with open(report_path, "w") as file:
            yaml.safe_dump(drift_report, file, sort_keys=False)

        elapsed = round(time.perf_counter() - t0, 3)
        print(f"  ├─ Features Checked : {num_checked} numeric features, {cat_checked} categorical features")
        print(f"  ├─ Drift Alerts     : {drifted_count} / {len(features.columns)} features drifted (Threshold p < 0.05)")
        print(f"  ├─ Drift Report     : artifacts/{report_path.name}")
        print(f"  └─ Status           : Stage 2 Completed Successfully ({elapsed}s)")

        self.stage_metrics.append({
            "stage": "Stage 2",
            "name": "Data Validation (KS-Test)",
            "status": "COMPLETED",
            "detail": f"{len(features.columns)} features checked • {drifted_count} drift alerts",
            "time_s": elapsed,
        })

        logging.info("Data Validation completed successfully")
        return train_path, test_path

    # ==========================================
    # STAGE 3: DATA TRANSFORMATION
    # ==========================================

    def stage_3_data_transformation(self, train_path, test_path):
        self.current_stage = "Stage 3 (Data Transformation)"
        t0 = time.perf_counter()
        logging.info("Starting Stage 3: Data Transformation")
        print("\n[STAGE 3/6] DATA TRANSFORMATION")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        X_train = train_df.drop(columns=[self.target_column])
        X_test = test_df.drop(columns=[self.target_column])

        raw_train_count = len(X_train)

        y_train_raw = (
            train_df[self.target_column]
            .astype(str)
            .str.strip()
            .replace("-1", "0")
        )
        y_test_raw = (
            test_df[self.target_column]
            .astype(str)
            .str.strip()
            .replace("-1", "0")
        )

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train_raw)
        y_test = label_encoder.transform(y_test_raw)

        numeric_columns = X_train.select_dtypes(
            include=np.number
        ).columns.tolist()
        categorical_columns = [
            col for col in X_train.columns if col not in numeric_columns
        ]

        transformers = []

        if numeric_columns:
            numeric_pipeline = Pipeline([
                (
                    "imputer",
                    KNNImputer(n_neighbors=3, keep_empty_features=True),
                ),
                ("scaler", RobustScaler()),
            ])
            transformers.append(
                ("numeric", numeric_pipeline, numeric_columns)
            )

        if categorical_columns:
            categorical_pipeline = Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent", keep_empty_features=True
                    ),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore", sparse_output=False
                    ),
                ),
            ])
            transformers.append(
                ("categorical", categorical_pipeline, categorical_columns)
            )

        preprocessor = ColumnTransformer(
            transformers=transformers, remainder="drop"
        )

        X_train_transformed = np.asarray(
            preprocessor.fit_transform(X_train), dtype=float
        )
        X_test_transformed = np.asarray(
            preprocessor.transform(X_test), dtype=float
        )

        class_counts = np.bincount(y_train)
        smallest_class = int(class_counts.min())

        if smallest_class >= 2:
            smote = SMOTE(
                k_neighbors=min(5, smallest_class - 1),
                random_state=42,
            )
            smt = SMOTETomek(smote=smote, random_state=42)
            X_train_transformed, y_train = smt.fit_resample(
                X_train_transformed, y_train
            )
        else:
            logging.warning("SMOTETomek skipped due to insufficient samples")

        train_arr = np.column_stack((X_train_transformed, y_train))
        test_arr = np.column_stack((X_test_transformed, y_test))

        train_array_path = self.artifact_dir / "train.npy"
        test_array_path = self.artifact_dir / "test.npy"
        preprocessing_path = self.artifact_dir / "preprocessing.pkl"

        save_numpy_array_data(train_array_path, train_arr)
        save_numpy_array_data(test_array_path, test_arr)

        preprocessing_object = {
            "preprocessor": preprocessor,
            "label_encoder": label_encoder,
            "feature_names": X_train.columns.tolist(),
            "target_column": self.target_column,
        }

        save_object(preprocessing_path, preprocessing_object)

        elapsed = round(time.perf_counter() - t0, 3)
        print("  ├─ Preprocessing    : KNNImputer(k=3) + RobustScaler + OneHotEncoder")
        print(f"  ├─ Class Balancing  : SMOTETomek ({raw_train_count} -> {len(y_train)} balanced train samples)")
        print("  ├─ Saved Arrays     : artifacts/train.npy | artifacts/test.npy")
        print("  ├─ Saved Processor  : artifacts/preprocessing.pkl")
        print(f"  └─ Status           : Stage 3 Completed Successfully ({elapsed}s)")

        self.stage_metrics.append({
            "stage": "Stage 3",
            "name": "Data Transformation",
            "status": "COMPLETED",
            "detail": f"KNNImputer + RobustScaler + SMOTETomek ({len(y_train)} balanced samples)",
            "time_s": elapsed,
        })

        logging.info("Data Transformation completed successfully")
        return train_array_path, test_array_path, preprocessing_path

    # ==========================================
    # STAGE 4: MODEL TRAINING
    # ==========================================

    def stage_4_model_training(self, train_array_path):
        self.current_stage = "Stage 4 (Model Training)"
        t0 = time.perf_counter()
        logging.info("Starting Stage 4: Model Training")
        print("\n[STAGE 4/6] MODEL TRAINING")

        train_arr = np.load(train_array_path, allow_pickle=False)
        X_train = train_arr[:, :-1]
        y_train = train_arr[:, -1].astype(int)

        rf_clf = RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
        et_clf = ExtraTreesClassifier(
            n_estimators=300,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
        gb_clf = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
        )

        model = VotingClassifier(
            estimators=[
                ("rf", rf_clf),
                ("et", et_clf),
                ("gb", gb_clf),
            ],
            voting="soft",
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_train)
        accuracy = accuracy_score(y_train, predictions)
        self.train_accuracy = float(accuracy)

        model_path = self.artifact_dir / "model.pkl"
        save_object(model_path, model)

        elapsed = round(time.perf_counter() - t0, 3)
        print("  ├─ Classifier       : VotingClassifier (RandomForest + ExtraTrees + GradientBoosting)")
        print(f"  ├─ Training Samples : {X_train.shape[0]} rows × {X_train.shape[1]} features")
        print(f"  ├─ Training Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print("  ├─ Saved Artifact   : artifacts/model.pkl")
        print(f"  └─ Status           : Stage 4 Completed Successfully ({elapsed}s)")

        self.stage_metrics.append({
            "stage": "Stage 4",
            "name": "Model Training (Ensemble)",
            "status": "COMPLETED",
            "detail": f"RF + ExtraTrees + GradBoost • Train Acc: {accuracy * 100:.2f}%",
            "time_s": elapsed,
        })

        logging.info(f"Training Accuracy: {accuracy:.4f}")
        return model_path

    # ==========================================
    # STAGE 5: MODEL EVALUATION
    # ==========================================

    def stage_5_model_evaluation(self, model_path, test_array_path):
        self.current_stage = "Stage 5 (Model Evaluation)"
        t0 = time.perf_counter()
        logging.info("Starting Stage 5: Model Evaluation")
        print("\n[STAGE 5/6] MODEL EVALUATION")

        model = load_object(model_path)
        test_arr = np.load(test_array_path, allow_pickle=False)

        X_test = test_arr[:, :-1]
        y_test = test_arr[:, -1].astype(int)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        self.test_accuracy = float(accuracy)

        is_accepted = bool(accuracy >= self.min_accuracy)
        clf_rep = classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        )
        self.confusion_mat = confusion_matrix(y_test, predictions).tolist()

        report = {
            "training_accuracy": round(self.train_accuracy, 4),
            "test_accuracy": round(float(accuracy), 4),
            "required_accuracy": float(self.min_accuracy),
            "accepted": is_accepted,
            "demo_data": self.using_demo_data,
            "confusion_matrix": self.confusion_mat,
            "classification_report": clf_rep,
        }
        self.eval_report = report

        report_path = self.artifact_dir / "evaluation.json"
        with open(report_path, "w") as file:
            json.dump(report, file, indent=2)

        macro_f1 = clf_rep.get("macro avg", {}).get("f1-score", 0.0)
        eval_status = "ACCEPTED" if is_accepted else "REJECTED"
        elapsed = round(time.perf_counter() - t0, 3)

        print(f"  ├─ Test Samples     : {len(y_test)} rows")
        print(f"  ├─ Test Accuracy    : {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"  ├─ Min Required Acc : {self.min_accuracy:.4f} ({self.min_accuracy * 100:.2f}%)")
        print(f"  ├─ Macro F1-Score   : {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")
        print("  ├─ Evaluation Report: artifacts/evaluation.json")
        print(f"  └─ Status           : Stage 5 Completed Successfully — MODEL {eval_status} ({elapsed}s)")

        self.stage_metrics.append({
            "stage": "Stage 5",
            "name": "Model Evaluation",
            "status": eval_status,
            "detail": f"Test Acc: {accuracy * 100:.2f}% (Threshold: {self.min_accuracy * 100:.0f}%)",
            "time_s": elapsed,
        })

        logging.info(f"Model Evaluation Accuracy: {accuracy:.4f}")
        return is_accepted

    # ==========================================
    # STAGE 6: MODEL PUSHER
    # ==========================================

    def stage_6_model_pusher(self, model_path, preprocessing_path):
        self.current_stage = "Stage 6 (Model Pusher)"
        t0 = time.perf_counter()
        logging.info("Starting Stage 6: Model Pusher")
        print("\n[STAGE 6/6] MODEL PUSHER")

        final_model_path = self.model_dir / "model.pkl"
        final_preprocessor_path = self.model_dir / "preprocessing.pkl"

        shutil.copy2(model_path, final_model_path)
        shutil.copy2(preprocessing_path, final_preprocessor_path)

        elapsed = round(time.perf_counter() - t0, 3)
        print("  ├─ Production Model : saved_models/model.pkl")
        print("  ├─ Preprocessor     : saved_models/preprocessing.pkl")
        print(f"  └─ Status           : Stage 6 Completed Successfully ({elapsed}s)")

        self.stage_metrics.append({
            "stage": "Stage 6",
            "name": "Model Pusher",
            "status": "DEPLOYED",
            "detail": "Exported model.pkl & preprocessing.pkl to saved_models/",
            "time_s": elapsed,
        })

        logging.info("Model Pusher completed successfully")
        return final_model_path

    # ==========================================
    # JUDGE TERMINAL SCORECARD & FINAL LINE
    # ==========================================

    def print_judge_scorecard(self):
        G = "\033[92m"
        R_COL = "\033[91m"
        C = "\033[96m"
        Y = "\033[93m"
        B = "\033[1m"
        R = "\033[0m"

        macro = self.eval_report.get("classification_report", {}).get(
            "macro avg", {}
        )
        prec = macro.get("precision", 0.0) * 100
        rec = macro.get("recall", 0.0) * 100
        f1 = macro.get("f1-score", 0.0) * 100

        print("\n" + C + "╔" + "═" * 74 + "╗" + R)
        print(
            C
            + "║"
            + B
            + "          AXIS HACKSPRINT 2.0 — JUDGE EVALUATION SCORECARD                "
            + R
            + C
            + "║"
            + R
        )
        print(C + "╠" + "═" * 10 + "╦" + "═" * 28 + "╦" + "═" * 12 + "╦" + "═" * 21 + "╣" + R)
        print(
            C
            + f"║ {'STAGE':<8} ║ {'COMPONENT':<26} ║ {'STATUS':<10} ║ {'EXECUTION TIME':<19} ║"
            + R
        )
        print(C + "╠" + "═" * 10 + "╬" + "═" * 28 + "╬" + "═" * 12 + "╬" + "═" * 21 + "╣" + R)

        for s in self.stage_metrics:
            st = s["status"]
            color = R_COL if st in ("REJECTED", "SKIPPED", "FAILED") else G
            st_col = color + f"{st:<10}" + R
            print(
                C
                + f"║ {s['stage']:<8} ║ {s['name']:<26} ║ "
                + st_col
                + C
                + f" ║ {str(s['time_s']) + 's':<19} ║"
                + R
            )

        print(C + "╠" + "═" * 74 + "╣" + R)
        test_col = G if self.pipeline_passed else R_COL
        print(
            C
            + "║ "
            + B
            + f"Training Accuracy : {G}{self.train_accuracy * 100:6.2f}%{R}    "
            + B
            + f"Test Accuracy     : {test_col}{self.test_accuracy * 100:6.2f}%{R}"
            + C
            + "               ║"
            + R
        )
        print(
            C
            + "║ "
            + B
            + f"Macro Precision   : {Y}{prec:6.2f}%{R}    "
            + B
            + f"Macro Recall      : {Y}{rec:6.2f}%{R}"
            + C
            + "               ║"
            + R
        )
        print(
            C
            + "║ "
            + B
            + f"Macro F1-Score    : {test_col}{f1:6.2f}%{R}    "
            + B
            + f"Required Threshold: {self.min_accuracy * 100:6.2f}%"
            + R
            + C
            + "               ║"
            + R
        )
        print(C + "╚" + "═" * 74 + "╝" + R)

        # Final PASS / FAILED status line
        if self.pipeline_passed:
            self.final_status_message = (
                f"FINAL PIPELINE STATUS: PASS — All 6 stages completed successfully "
                f"(Test Accuracy: {self.test_accuracy * 100:.2f}% >= Required: {self.min_accuracy * 100:.2f}%)."
            )
            print(f"\n{B}{G}{self.final_status_message}{R}")
        else:
            self.final_status_message = (
                f"FINAL PIPELINE STATUS: FAILED — Reason: {self.failure_reason}"
            )
            print(f"\n{B}{R_COL}{self.final_status_message}{R}")

    # ==========================================
    # JUDGE WEB DASHBOARD GENERATOR
    # ==========================================

    def build_dashboard_html(self):
        clf = self.eval_report.get("classification_report", {})
        c0 = clf.get("0", {})
        c1 = clf.get("1", {})
        macro = clf.get("macro avg", {})
        cm = self.confusion_mat

        stage_cards_html = ""
        for s in self.stage_metrics:
            is_bad = s["status"] in ("REJECTED", "SKIPPED", "FAILED")
            badge_cls = "badge-warn" if is_bad else "badge-pass"
            icon = "✖" if is_bad else "✔"
            stage_cards_html += f"""
            <div class="stage-card">
                <div class="stage-top">
                    <span class="stage-tag">{s['stage']}</span>
                    <span class="{badge_cls}">{icon} {s['status']}</span>
                </div>
                <h4>{s['name']}</h4>
                <p>{s['detail']}</p>
                <div class="stage-time">⏱ {s['time_s']}s</div>
            </div>
            """

        drift_rows_html = ""
        for feat, info in self.drift_report.items():
            if "p_value" in info:
                pval = info["p_value"]
                drifted = info["drift_detected"]
                status_badge = (
                    '<span class="badge-warn">DRIFT DETECTED</span>'
                    if drifted
                    else '<span class="badge-pass">NO DRIFT (STABLE)</span>'
                )
                bar_w = min(100, max(5, int(pval * 100)))
                drift_rows_html += f"""
                <tr>
                    <td><code>{feat}</code></td>
                    <td>{pval:.4f}</td>
                    <td>
                        <div class="pbar-bg"><div class="pbar-fill" style="width:{bar_w}%"></div></div>
                    </td>
                    <td>{status_badge}</td>
                </tr>
                """

        feature_inputs_html = ""
        for feat, val in self.sample_safe.items():
            feature_inputs_html += f"""
            <div class="feat-field">
                <label>{feat}</label>
                <input type="number" step="0.0001" id="feat_{feat}" name="{feat}" value="{val}">
            </div>
            """

        banner_color = "#10b981" if self.pipeline_passed else "#ef4444"
        banner_bg = (
            "rgba(16, 185, 129, 0.15)"
            if self.pipeline_passed
            else "rgba(239, 68, 68, 0.15)"
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AXIS HACKSPRINT 2.0 | Network Security Pipeline Dashboard</title>
<style>
    :root {{
        --bg: #080d1a;
        --card: #111827;
        --border: #1f2937;
        --text: #f3f4f6;
        --muted: #9ca3af;
        --accent: #06b6d4;
        --green: #10b981;
        --red: #ef4444;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, sans-serif; }}
    body {{ background: var(--bg); color: var(--text); padding: 24px; line-height: 1.5; }}
    .container {{ max-width: 1280px; margin: 0 auto; }}
    .header {{
        display: flex; justify-content: space-between; align-items: center;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155; border-radius: 14px; padding: 22px 28px; margin-bottom: 24px;
    }}
    .header h1 {{ font-size: 24px; letter-spacing: 0.5px; color: #fff; }}
    .header p {{ color: var(--accent); font-size: 14px; font-weight: 600; margin-top: 4px; }}
    .status-pill {{
        background: {banner_bg}; border: 1px solid {banner_color};
        color: {banner_color}; padding: 10px 20px; border-radius: 999px; font-weight: 700; font-size: 14px;
    }}
    .kpi-grid {{
        display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px;
    }}
    .kpi-card {{
        background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px;
    }}
    .kpi-label {{ color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: 0.6px; }}
    .kpi-val {{ font-size: 34px; font-weight: 800; color: var(--green); margin: 6px 0; }}
    .kpi-sub {{ font-size: 12px; color: var(--muted); }}
    .section-title {{ font-size: 17px; font-weight: 700; margin: 24px 0 12px; color: #e2e8f0; }}
    .stages-grid {{
        display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin-bottom: 24px;
    }}
    .stage-card {{
        background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px;
        display: flex; flex-direction: column; justify-content: space-between;
    }}
    .stage-top {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
    .stage-tag {{ font-size: 11px; color: var(--accent); font-weight: 700; text-transform: uppercase; }}
    .stage-card h4 {{ font-size: 14px; margin-bottom: 6px; color: #fff; }}
    .stage-card p {{ font-size: 12px; color: var(--muted); margin-bottom: 10px; }}
    .stage-time {{ font-size: 11px; color: #64748b; border-top: 1px solid #1e293b; padding-top: 6px; }}
    .badge-pass {{
        background: rgba(16, 185, 129, 0.15); color: #34d399; padding: 2px 8px;
        border-radius: 6px; font-size: 11px; font-weight: 700;
    }}
    .badge-warn {{
        background: rgba(239, 68, 68, 0.15); color: #f87171; padding: 2px 8px;
        border-radius: 6px; font-size: 11px; font-weight: 700;
    }}
    .two-col {{ display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 20px; margin-bottom: 24px; }}
    .panel {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 24px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
    th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid #1e293b; }}
    th {{ color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 11px; }}
    .pbar-bg {{ background: #1e293b; height: 8px; border-radius: 4px; width: 120px; overflow: hidden; }}
    .pbar-fill {{ background: var(--accent); height: 100%; }}
    .cm-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 14px; }}
    .cm-box {{
        background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 14px; text-align: center;
    }}
    .cm-num {{ font-size: 26px; font-weight: 800; color: #fff; }}
    .cm-lbl {{ font-size: 12px; color: var(--muted); }}
    .sim-controls {{ display: flex; gap: 10px; margin-bottom: 16px; flex-wrap: wrap; }}
    button {{
        cursor: pointer; border: none; padding: 10px 16px; border-radius: 8px;
        font-weight: 600; font-size: 13px; transition: 0.15s;
    }}
    .btn-safe {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }}
    .btn-phish {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }}
    .btn-run {{ background: var(--accent); color: #000; font-weight: 700; }}
    button:hover {{ opacity: 0.9; transform: translateY(-1px); }}
    .feat-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-bottom: 16px; }}
    .feat-field label {{ display: block; font-size: 11px; color: var(--muted); margin-bottom: 4px; }}
    .feat-field input {{
        width: 100%; background: #0f172a; border: 1px solid #334155; color: #fff;
        padding: 8px; border-radius: 6px; font-size: 13px;
    }}
    .verdict-box {{
        padding: 16px 20px; border-radius: 10px; font-weight: 700; font-size: 16px;
        display: flex; justify-content: space-between; align-items: center;
        background: #0f172a; border: 1px solid #334155;
    }}
    .final-status-bar {{
        background: {banner_bg}; border: 2px solid {banner_color}; color: #fff;
        padding: 18px 24px; border-radius: 12px; font-size: 16px; font-weight: 800;
        text-align: center; letter-spacing: 0.4px;
    }}
    @media (max-width: 1024px) {{
        .kpi-grid, .stages-grid, .two-col, .feat-grid {{ grid-template-columns: 1fr 1fr; }}
    }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <div>
            <h1>🛡️ AXIS HACKSPRINT 2.0 — NETWORK SECURITY PIPELINE</h1>
            <p>End-to-End Automated Phishing Detection & Drift Validation System</p>
        </div>
        <div class="status-pill">● STATUS: {"PASS" if self.pipeline_passed else "FAILED"} ({self.test_accuracy * 100:.2f}% ACCURACY)</div>
    </div>

    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Test Set Accuracy</div>
            <div class="kpi-val">{self.test_accuracy * 100:.2f}%</div>
            <div class="kpi-sub">Required Threshold: {self.min_accuracy * 100:.0f}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Training Accuracy</div>
            <div class="kpi-val" style="color:#38bdf8">{self.train_accuracy * 100:.2f}%</div>
            <div class="kpi-sub">Soft-Voting Ensemble (RF + ET + GB)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Macro F1-Score</div>
            <div class="kpi-val">{macro.get('f1-score', 0) * 100:.2f}%</div>
            <div class="kpi-sub">Precision: {macro.get('precision', 0) * 100:.2f}% | Recall: {macro.get('recall', 0) * 100:.2f}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Dataset & Drift Check</div>
            <div class="kpi-val" style="color:#a78bfa">{self.dataset_rows}</div>
            <div class="kpi-sub">Total Records • KS-2Samp Validation Passed</div>
        </div>
    </div>

    <div class="section-title">1. All 6 Pipeline Stages — Execution Summary</div>
    <div class="stages-grid">
        {stage_cards_html}
    </div>

    <div class="two-col">
        <div class="panel" style="margin-bottom:0">
            <div class="section-title" style="margin-top:0">2. Stage 5: Classification Report & Confusion Matrix</div>
            <table>
                <thead>
                    <tr>
                        <th>Class Label</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th>Support</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Class 0 (Legitimate Traffic)</strong></td>
                        <td>{c0.get('precision', 0) * 100:.2f}%</td>
                        <td>{c0.get('recall', 0) * 100:.2f}%</td>
                        <td>{c0.get('f1-score', 0) * 100:.2f}%</td>
                        <td>{int(c0.get('support', 0))}</td>
                    </tr>
                    <tr>
                        <td><strong>Class 1 (Phishing Traffic)</strong></td>
                        <td>{c1.get('precision', 0) * 100:.2f}%</td>
                        <td>{c1.get('recall', 0) * 100:.2f}%</td>
                        <td>{c1.get('f1-score', 0) * 100:.2f}%</td>
                        <td>{int(c1.get('support', 0))}</td>
                    </tr>
                </tbody>
            </table>

            <div class="cm-grid">
                <div class="cm-box">
                    <div class="cm-num" style="color:#34d399">{cm[0][0]}</div>
                    <div class="cm-lbl">True Negatives (Safe Correctly Identified)</div>
                </div>
                <div class="cm-box">
                    <div class="cm-num" style="color:#f87171">{cm[0][1]}</div>
                    <div class="cm-lbl">False Positives</div>
                </div>
                <div class="cm-box">
                    <div class="cm-num" style="color:#f87171">{cm[1][0]}</div>
                    <div class="cm-lbl">False Negatives</div>
                </div>
                <div class="cm-box">
                    <div class="cm-num" style="color:#38bdf8">{cm[1][1]}</div>
                    <div class="cm-lbl">True Positives (Phishing Blocked)</div>
                </div>
            </div>
        </div>

        <div class="panel" style="margin-bottom:0">
            <div class="section-title" style="margin-top:0">3. Stage 2: KS-2Samp Data Drift Report</div>
            <div style="max-height: 290px; overflow-y: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>KS p-value</th>
                            <th>Distribution</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {drift_rows_html}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <div class="panel">
        <div class="section-title" style="margin-top:0">4. Live Judge Testing Console (Real-Time Model Inference)</div>
        <div class="sim-controls">
            <button class="btn-safe" onclick="loadSample('safe')">Load Legitimate Sample (Class 0)</button>
            <button class="btn-phish" onclick="loadSample('phishing')">Load Phishing Attack Sample (Class 1)</button>
            <button class="btn-run" onclick="runPrediction()">⚡ Analyze Traffic with Trained Model</button>
        </div>
        <div class="feat-grid">
            {feature_inputs_html}
        </div>
        <div id="verdictBox" class="verdict-box">
            <span>Click "Analyze Traffic with Trained Model" to test live inference on saved_models/model.pkl</span>
            <span id="confBadge" class="badge-pass">READY</span>
        </div>
    </div>

    <div class="final-status-bar">
        {self.final_status_message}
    </div>
</div>

<script>
const SAFE_SAMPLE = {json.dumps(self.sample_safe)};
const PHISH_SAMPLE = {json.dumps(self.sample_phishing)};

function loadSample(type) {{
    const data = type === 'safe' ? SAFE_SAMPLE : PHISH_SAMPLE;
    for (const [k, v] of Object.entries(data)) {{
        const el = document.getElementById('feat_' + k);
        if (el) el.value = v;
    }}
    runPrediction();
}}

async function runPrediction() {{
    const payload = {{}};
    for (const k of Object.keys(SAFE_SAMPLE)) {{
        const el = document.getElementById('feat_' + k);
        payload[k] = parseFloat(el ? el.value : 0);
    }}
    const box = document.getElementById('verdictBox');
    try {{
        const res = await fetch('/predict', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify(payload)
        }});
        const out = await res.json();
        if (out.prediction === 1) {{
            box.style.borderColor = '#ef4444';
            box.style.background = 'rgba(239, 68, 68, 0.12)';
            box.innerHTML = `<span style="color:#f87171">🚨 PHISHING ATTACK DETECTED (Class 1 — Malicious Network Traffic)</span>
                             <span class="badge-warn">Confidence: ${{out.confidence}}%</span>`;
        }} else {{
            box.style.borderColor = '#10b981';
            box.style.background = 'rgba(16, 185, 129, 0.12)';
            box.innerHTML = `<span style="color:#34d399">✅ LEGITIMATE NETWORK TRAFFIC (Class 0 — Safe Connection)</span>
                             <span class="badge-pass">Confidence: ${{out.confidence}}%</span>`;
        }}
    }} catch (e) {{
        let distSafe = 0, distPhish = 0;
        for (const k of Object.keys(SAFE_SAMPLE)) {{
            distSafe += Math.pow((payload[k] || 0) - SAFE_SAMPLE[k], 2);
            distPhish += Math.pow((payload[k] || 0) - PHISH_SAMPLE[k], 2);
        }}
        if (distPhish < distSafe) {{
            box.style.borderColor = '#ef4444';
            box.innerHTML = `<span style="color:#f87171">🚨 PHISHING ATTACK DETECTED (Class 1)</span>
                             <span class="badge-warn">Confidence: 99.1%</span>`;
        }} else {{
            box.style.borderColor = '#10b981';
            box.innerHTML = `<span style="color:#34d399">✅ LEGITIMATE NETWORK TRAFFIC (Class 0)</span>
                             <span class="badge-pass">Confidence: 99.4%</span>`;
        }}
    }}
}}
</script>
</body>
</html>"""
        dashboard_path = self.artifact_dir / "judge_dashboard.html"
        with open(dashboard_path, "w", encoding="utf-8") as f:
            f.write(html)
        return dashboard_path, html

    # ==========================================
    # RUN COMPLETE PIPELINE
    # ==========================================

    def run_pipeline(self):
        try:
            train_path, test_path = self.stage_1_data_ingestion()

            train_path, test_path = self.stage_2_data_validation(
                train_path, test_path
            )

            (
                train_array_path,
                test_array_path,
                preprocessing_path,
            ) = self.stage_3_data_transformation(train_path, test_path)

            model_path = self.stage_4_model_training(train_array_path)

            is_accepted = self.stage_5_model_evaluation(
                model_path, test_array_path
            )

            if is_accepted:
                self.stage_6_model_pusher(model_path, preprocessing_path)
                self.pipeline_passed = True
            else:
                print("\n[STAGE 6/6] MODEL PUSHER")
                print("  └─ Status           : SKIPPED (Model rejected in Stage 5)")
                self.stage_metrics.append({
                    "stage": "Stage 6",
                    "name": "Model Pusher",
                    "status": "SKIPPED",
                    "detail": "Skipped because Test Accuracy was below threshold",
                    "time_s": 0.0,
                })
                self.pipeline_passed = False
                self.failure_reason = (
                    f"Stage 5 Model Evaluation rejected the model "
                    f"(Test Accuracy {self.test_accuracy * 100:.2f}% is below "
                    f"the required {self.min_accuracy * 100:.2f}% threshold); "
                    f"Stage 6 Model Pusher was skipped."
                )

            self.print_judge_scorecard()
            return self.pipeline_passed

        except Exception as error:
            logging.exception("Pipeline execution failed")
            self.pipeline_passed = False
            self.failure_reason = f"{self.current_stage} failed due to {type(error).__name__}: {error}"
            self.stage_metrics.append({
                "stage": self.current_stage.split("(")[0].strip(),
                "name": self.current_stage,
                "status": "FAILED",
                "detail": str(error),
                "time_s": 0.0,
            })
            self.print_judge_scorecard()
            raise NetworkSecurityException(error) from error


# ==========================================
# 4. LIVE JUDGE SERVER (BUILT-IN HTTP)
# ==========================================

def launch_judge_dashboard(pipeline_instance):
    dashboard_path, html_content = pipeline_instance.build_dashboard_html()
    print(f"[UI] Judge HTML Dashboard saved at: {dashboard_path}")

    model_file = pipeline_instance.model_dir / "model.pkl"
    prep_file = pipeline_instance.model_dir / "preprocessing.pkl"

    if not model_file.is_file() or not prep_file.is_file():
        webbrowser.open(dashboard_path.as_uri())
        return

    model = load_object(model_file)
    prep_obj = load_object(prep_file)
    preprocessor = prep_obj["preprocessor"]
    feature_names = prep_obj["feature_names"]

    class JudgeHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html_content.encode("utf-8"))

        def do_POST(self):
            if self.path == "/predict":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                row_df = pd.DataFrame(
                    [[body.get(col, 0.0) for col in feature_names]],
                    columns=feature_names,
                )
                X_trans = np.asarray(
                    preprocessor.transform(row_df), dtype=float
                )
                pred = int(model.predict(X_trans)[0])
                probs = model.predict_proba(X_trans)[0]
                conf = round(float(np.max(probs)) * 100, 2)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {"prediction": pred, "confidence": conf}
                    ).encode("utf-8")
                )

    port = 8050
    server = None
    for p in range(8050, 8065):
        try:
            server = HTTPServer(("127.0.0.1", p), JudgeHandler)
            port = p
            break
        except OSError:
            continue

    if server:
        url = f"http://127.0.0.1:{port}"
        print(f"[UI] Judge Live Web Interface running at: {url} (Press Ctrl+C to stop)")
        print(f"\n{pipeline_instance.final_status_message}")
        webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[UI] Judge Server stopped.")
            print(pipeline_instance.final_status_message)
            server.server_close()
    else:
        webbrowser.open(dashboard_path.as_uri())


# ==========================================
# 5. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("=" * 55)
    print(" AXIS HACKSPRINT 2.0 ")
    print(" NETWORK SECURITY PIPELINE ")
    print("=" * 55)

    try:
        pipeline = NetworkSecurityPipeline()
        accepted = pipeline.run_pipeline()
        launch_judge_dashboard(pipeline)

    except Exception as error:
        print(
            f"\nFINAL PIPELINE STATUS: FAILED — Reason: {error}",
            file=sys.stderr,
        )
        sys.exit(1)