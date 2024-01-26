from copy import deepcopy
from dataclasses import dataclass, field

import xgboost as xgb
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation.data_transformation import \
    DataTransformation
from src.config import *
from src.logger import LOG_ENDING, logging
from src.utils import get_X_sets, get_y_sets


@dataclass
class ModelTrainerConfig:
    rf_classifier: RandomForestRegressor = field(
        default_factory=lambda: RandomForestRegressor(
            n_jobs=-1, random_state=RANDOM_SEED
        )
    )
    all_models: dict = field(init=False)

    def __post_init__(self):
        self.rf_classifier.estimator_ = DecisionTreeRegressor(random_state=RANDOM_SEED)

        self.all_models = {
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


class ModelTrainer:
    def __init__(self, model_name: str):
        self.config = ModelTrainerConfig()
        self.model_name = model_name
        self.model = self.config.all_models[model_name]

    def add_predictor_to_pipeline(self, pipeline: Pipeline):
        train_pipeline = deepcopy(pipeline)
        train_pipeline.steps.append(("predictor", self.model))
        return train_pipeline

    def start(self, train_pipeline, X, y):
        logging.info(f"Training predictor {self.model_name} started.")
        train_pipeline.fit(X, y)
        logging.info(f"Training predictor {self.model_name} finished{LOG_ENDING}")


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    df, df_train, df_test, df_test_submission = data_ingestion.start()

    data_transformation = DataTransformation()
    df_train_preprocessed, df_test_preprocessed = data_transformation.start(
        df_train, df_test
    )

    X_train = get_X_sets(df_train_preprocessed)
    y_train = get_y_sets(df_train_preprocessed)

    model_trainer = ModelTrainer("ridge")
    train_pipeline = model_trainer.add_predictor_to_pipeline(Pipeline([]))
    model_trainer.start(train_pipeline, X_train, y_train)
