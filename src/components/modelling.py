import os
from dataclasses import dataclass, field
from importlib import reload

import numpy as np
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation.data_transformation import \
    DataTransformation
from src.config import *

# reload(src.config);


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
        pass


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    df, df_train, df_test, df_test_submission = data_ingestion.start()

    data_transformation = DataTransformation()
    df_train_preprocessed, df_test_preprocessed = data_transformation.start(
        df_train, df_test
    )

    modelling = Modelling()
    modelling.start(df_train_preprocessed, df_test_preprocessed)
