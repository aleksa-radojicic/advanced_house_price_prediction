import os
from copy import deepcopy
from dataclasses import dataclass, field
from importlib import reload
from typing import Dict

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation.data_transformation import \
    DataTransformation
from src.config import *
# reload(src.config);
from src.logger import LOG_ENDING, logging
from src.utils import (get_X_sets, get_y_sets, json_object, pickle_object,
                       transform_y_sets)


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
            "dummy_mean": DummyRegressor(strategy="mean"),
            "dummy_median": DummyRegressor(strategy="median"),
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

        X_train, X_test = get_X_sets([df_train, df_test])
        y_train, y_test = get_y_sets([df_train, df_test])
        y_train_transformed, y_test_transformed = transform_y_sets([y_train, y_test])

        eval_metrics: Dict[str, float] = {}
        models: Dict[str, RegressorMixin] = self.modelling_config.models

        for model_name, predictor_obj in models.items():
            pipeline = self.append_predictor_to_pipeline(Pipeline([]), predictor_obj)

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

        logging.info(f"Evaluation metrics for all models:")
        # Sorting by lowest RMSE
        for model_name, eval_metric in sorted(eval_metrics.items(), key=lambda x: x[1]):
            logging.info(f"{model_name}: {eval_metric}")

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

    def train_model(self, pipeline: Pipeline, X, y):
        pipeline.fit(X, y)

    def evaluate_model(self, pipeline: Pipeline, X, y) -> float:
        y_pred = pipeline.predict(X).reshape(-1, 1)  # type: ignore
        # y_pred_linreg_exp = np.expm1(y_pred_linreg)

        main_metric = self.modelling_config.calculate_evaluation_metric(y, y_pred)
        return float(main_metric)

    def append_predictor_to_pipeline(self, pipeline, predictor_obj):
        pipeline = deepcopy(pipeline)
        pipeline.steps.append(("predictor", predictor_obj))
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


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    df, df_train, df_test, df_test_submission = data_ingestion.start()

    data_transformation = DataTransformation()
    df_train_preprocessed, df_test_preprocessed = data_transformation.start(
        df_train, df_test
    )

    modelling = Modelling()
    modelling.start(df_train_preprocessed, df_test_preprocessed)
