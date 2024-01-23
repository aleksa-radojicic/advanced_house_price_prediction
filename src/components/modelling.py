import os
from dataclasses import dataclass, field
from importlib import reload
from typing import Dict

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation.data_transformation import DataTransformation
from src.config import *

# reload(src.config);
from src.logger import LOG_ENDING, logging
from src.utils import json_object, pickle_object


@dataclass
class ModellingConfig:
    rf_classifier: RandomForestRegressor = field(
        default_factory=lambda: RandomForestRegressor(
            n_jobs=-1, random_state=RANDOM_SEED
        )
    )
    models: dict = field(init=False)

    best_trained_model_path = os.path.join("artifacts", "best_basic_model.pkl")
    best_trained_model_details_path = os.path.join(
        "artifacts", "best_basic_model_details.json"
    )

    def calculate_evaluation_metric(self, y_true, y_pred):
        root_mean_squared_error = np.sqrt(mean_squared_error(y_true, y_pred))
        return root_mean_squared_error

    def __post_init__(self):
        self.rf_classifier.estimator_ = DecisionTreeRegressor(random_state=RANDOM_SEED)

        self.models = {
            "ridge": Ridge(random_state=RANDOM_SEED),
            "svr": SVR(),
            "knn": KNeighborsRegressor(n_jobs=-1),
            "dt": DecisionTreeRegressor(random_state=RANDOM_SEED),
            "ada": AdaBoostRegressor(random_state=RANDOM_SEED),
            "rf": self.rf_classifier,
            "xgb": xgb.XGBRegressor(),
        }


class Modelling:
    def __init__(self):
        self.modelling_config = ModellingConfig()

    def start(self, df_train, df_test):
        logging.info(msg="Modelling method or component started")

        X_train, X_test = self.get_X_train_test_sets(df_train, df_test)
        y_train, y_test = self.get_y_train_test_sets(df_train, df_test)
        y_train_transformed, y_test_transformed = self.transform_y_train_test_sets(
            y_train, y_test
        )

        ct_obj = self.create_column_transformer()
        logging.info("Column transformer created.")

        eval_metrics: Dict[str, float] = {}
        models: Dict[str, RegressorMixin] = self.modelling_config.models

        for model_name, classif_obj in models.items():
            pipeline = self.create_pipeline(ct_obj, classif_obj)

            logging.info(f"Training {model_name} started.")
            self.train_model(pipeline, X_train, y_train_transformed)
            logging.info(f"Training {model_name} finished" + LOG_ENDING)

            logging.info(f"Evaluation of {model_name} started.")
            main_eval_metric = self.evaluate_model(pipeline, X_test, y_test_transformed)
            eval_metrics[model_name] = main_eval_metric
            logging.info(
                f"Evaluation of {model_name} finished.\nMain evaluation metric: {main_eval_metric}"
                + LOG_ENDING
            )
        logging.info("All models are trained and evaluated.")

        best_model_name, best_model, best_eval_metric = self.get_best_basic_model(
            models, eval_metrics
        )
        logging.info(f"Best basic model: {best_model_name}")
        logging.info(f"Best basic model evaluation metric: {best_eval_metric}")

        model_details = self.create_model_details(
            best_model_name, RANDOM_SEED, best_eval_metric
        )

        self.save_model(self.modelling_config.best_trained_model_path, best_model)
        logging.info(f"Saved best model {best_model_name}.")
        self.save_model_details(
            self.modelling_config.best_trained_model_details_path, model_details
        )
        logging.info(f"Saved best model details.")

        logging.info("Modelling method or component finished" + LOG_ENDING)

    def get_X_train_test_sets(self, ds_train, ds_test):
        X_train = ds_train.drop(LABEL, axis=1)
        X_test = ds_test.drop(LABEL, axis=1)

        return X_train, X_test

    def get_y_train_test_sets(self, ds_train, ds_test):
        y_train = ds_train[LABEL]
        y_test = ds_test[LABEL]

        return y_train, y_test

    def transform_y_train_test_sets(self, y_train, y_test):
        y_train_transformed = np.log1p(y_train)
        y_test_transformed = np.log1p(y_test)

        return y_train_transformed, y_test_transformed

    def create_column_transformer(self):
        ct = ColumnTransformer(
            [
                # ("numerical", MinMaxScaler(), make_column_selector("numerical__")),
                ("numerical", "passthrough", make_column_selector("numerical__")),
                # ("binary", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), features_info['binary']),
                ("binary", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), make_column_selector(pattern="binary__")),
                # ("ordinal", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), features_info["ordinal"]),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), make_column_selector(pattern="ordinal__")),
                ("nominal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int16), make_column_selector(pattern="nominal__"))
                # ("nominal", OneHotEncoder(handle_unknown='ignore', dtype=np.int8, sparse_output=False), features_info["nominal"])
            ],
            remainder="drop",
            verbose_feature_names_out=False,  # False because prefixes are added manually
        ).set_output(transform="pandas")
        return ct

    def train_model(self, pipeline: Pipeline, X, y):
        pipeline.fit(X, y)

    def evaluate_model(self, pipeline: Pipeline, X, y) -> float:
        y_pred = pipeline.predict(X).reshape(-1, 1)  # type: ignore
        # y_pred_linreg_exp = np.expm1(y_pred_linreg)

        main_metric = self.modelling_config.calculate_evaluation_metric(y, y_pred)
        return float(main_metric)

    def create_pipeline(self, column_transformer_obj, classifier_obj):
        pipeline = Pipeline(
            [("ct", column_transformer_obj), ("classifier", classifier_obj)]
        )
        return pipeline

    def get_best_basic_model(
        self, models: Dict[str, RegressorMixin], evaluation_metrics: Dict[str, float]
    ):
        best_model_str, best_model_evaluation_metric = sorted(
            evaluation_metrics.items(), key=lambda x: x[1]
        )[0]
        best_model = models[best_model_str]
        return best_model_str, best_model, best_model_evaluation_metric

    def save_model(self, file_path, model):
        pickle_object(file_path, model)

    def save_model_details(self, file_path, model_details):
        json_object(file_path, model_details)

    def create_model_details(
        self, best_model_name: str, RANDOM_SEED: int, best_eval_metric: float
    ) -> dict:
        model_details = {
            "Model basic": best_model_name,
            "random_state": RANDOM_SEED,
            "Evaluation metric (root mean squared error)": best_eval_metric,
        }
        return model_details


def delete_rows(df: pd.DataFrame):
    df.drop([598], inplace=True)


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    df, df_train, df_test, df_test_submission = data_ingestion.start()

    data_transformation = DataTransformation()
    df_train_preprocessed, df_test_preprocessed = data_transformation.start(
        df_train, df_test
    )

    # The deletion of rows should be handled differently
    delete_rows(df_train_preprocessed)

    modelling = Modelling()
    modelling.start(df_train_preprocessed, df_test_preprocessed)
