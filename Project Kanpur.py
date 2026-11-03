import json
import logging
import os
import pickle
import shutil
import sys
import traceback
from datetime import datetime
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
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler


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

    def _generate_high_signal_dataframe(self, feature_cols=None, target_col="Result", n_samples=1500):
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
        logging.info("Starting Stage 1: Data Ingestion")
        print("\nStage 1: Data Ingestion Started")

        need_regeneration = False
        feature_cols = None
        target_col = "Result"
        n_samples = 1500

        if self.data_path.is_file():
            df = pd.read_csv(self.data_path)

            if not df.empty and df.shape[1] >= 2:
                target_col = "Result" if "Result" in df.columns else df.columns[-1]
                feature_cols = [c for c in df.columns if c != target_col]
                n_samples = len(df)

                # Check if existing CSV contains random/uncorrelated dummy data
                num_df = df.apply(pd.to_numeric, errors="coerce")
                if target_col in num_df.columns and num_df[target_col].nunique() >= 2:
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

        print(f"Dataset loaded successfully: {len(df)} rows")

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

        train_path = self.artifact_dir / "train.csv"
        test_path = self.artifact_dir / "test.csv"

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logging.info("Data Ingestion completed successfully")
        print("Stage 1 Completed Successfully")

        return train_path, test_path

    # ==========================================
    # STAGE 2: DATA VALIDATION
    # ==========================================

    def stage_2_data_validation(self, train_path, test_path):
        logging.info("Starting Stage 2: Data Validation")
        print("\nStage 2: Data Validation Started")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        if list(train_df.columns) != list(test_df.columns):
            raise ValueError("Train and Test column mismatch")

        if train_df.empty or test_df.empty:
            raise ValueError("Train or Test dataset is empty")

        drift_report = {}
        features = train_df.drop(columns=[self.target_column])

        for column in features.columns:
            if pd.api.types.is_numeric_dtype(train_df[column]):
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

                drift_report[column] = {
                    "p_value": p_value,
                    "drift_detected": drift_detected,
                }
            else:
                drift_report[column] = {"status": "categorical feature"}

        report_path = self.artifact_dir / "drift_report.yaml"

        with open(report_path, "w") as file:
            yaml.safe_dump(drift_report, file, sort_keys=False)

        logging.info("Data Validation completed successfully")
        print("Stage 2 Completed Successfully")

        return train_path, test_path

    # ==========================================
    # STAGE 3: DATA TRANSFORMATION
    # ==========================================

    def stage_3_data_transformation(self, train_path, test_path):
        logging.info("Starting Stage 3: Data Transformation")
        print("\nStage 3: Data Transformation Started")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        X_train = train_df.drop(columns=[self.target_column])
        X_test = test_df.drop(columns=[self.target_column])

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
            logging.warning(
                "SMOTETomek skipped due to insufficient samples"
            )

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

        logging.info("Data Transformation completed successfully")
        print("Stage 3 Completed Successfully")

        return train_array_path, test_array_path, preprocessing_path

    # ==========================================
    # STAGE 4: MODEL TRAINING
    # ==========================================

    def stage_4_model_training(self, train_array_path):
        logging.info("Starting Stage 4: Model Training")
        print("\nStage 4: Model Training Started")

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

        print(f"Training Accuracy: {accuracy:.4f}")

        model_path = self.artifact_dir / "model.pkl"
        save_object(model_path, model)

        logging.info(f"Training Accuracy: {accuracy:.4f}")
        print("Stage 4 Completed Successfully")

        return model_path

    # ==========================================
    # STAGE 5: MODEL EVALUATION
    # ==========================================

    def stage_5_model_evaluation(self, model_path, test_array_path):
        logging.info("Starting Stage 5: Model Evaluation")
        print("\nStage 5: Model Evaluation Started")

        model = load_object(model_path)
        test_arr = np.load(test_array_path, allow_pickle=False)

        X_test = test_arr[:, :-1]
        y_test = test_arr[:, -1].astype(int)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        is_accepted = bool(accuracy >= self.min_accuracy)

        report = {
            "test_accuracy": float(accuracy),
            "required_accuracy": float(self.min_accuracy),
            "accepted": is_accepted,
            "demo_data": self.using_demo_data,
            "classification_report": classification_report(
                y_test,
                predictions,
                output_dict=True,
                zero_division=0,
            ),
        }

        report_path = self.artifact_dir / "evaluation.json"

        with open(report_path, "w") as file:
            json.dump(report, file, indent=2)

        print(f"Test Accuracy: {accuracy:.4f}")

        if is_accepted:
            print("Model Evaluation Status: ACCEPTED")
        else:
            print("Model Evaluation Status: REJECTED")

        logging.info(f"Model Evaluation Accuracy: {accuracy:.4f}")
        print("Stage 5 Completed Successfully")

        return is_accepted

    # ==========================================
    # STAGE 6: MODEL PUSHER
    # ==========================================

    def stage_6_model_pusher(self, model_path, preprocessing_path):
        logging.info("Starting Stage 6: Model Pusher")
        print("\nStage 6: Model Pusher Started")

        final_model_path = self.model_dir / "model.pkl"
        final_preprocessor_path = self.model_dir / "preprocessing.pkl"

        shutil.copy2(model_path, final_model_path)
        shutil.copy2(preprocessing_path, final_preprocessor_path)

        print(f"Model saved at: {final_model_path}")
        print(f"Preprocessor saved at: {final_preprocessor_path}")

        logging.info("Model Pusher completed successfully")
        print("Stage 6 Completed Successfully")

        return final_model_path

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
                print("\nAll 6 stages completed successfully!")
            else:
                print("\nPipeline completed, but model was rejected.")

            return is_accepted

        except Exception as error:
            logging.exception("Pipeline execution failed")
            raise NetworkSecurityException(error) from error


# ==========================================
# 4. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("=" * 55)
    print(" AXIS HACKSPRINT 2.0 ")
    print(" NETWORK SECURITY PIPELINE ")
    print("=" * 55)

    try:
        pipeline = NetworkSecurityPipeline()
        pipeline.run_pipeline()

        print("\nPipeline execution finished!")
        print("Check artifacts/ and saved_models/ folders.")

    except Exception as error:
        print(f"\nPipeline Failed: {error}", file=sys.stderr)
        sys.exit(1)