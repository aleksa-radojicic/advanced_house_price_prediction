from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation.data_transformation import \
    DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import LOG_ENDING, logging
from src.utils import get_X_sets, get_y_sets


@dataclass
class ModelEvaluatorConfig:
    def calculate_evaluation_metric(self, y_true, y_pred):
        root_mean_squared_error = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        return root_mean_squared_error


class ModelEvaluator:
    def __init__(self, model_name):
        self.config = ModelEvaluatorConfig()
        self.model_name = model_name

    def start(self, train_pipeline: Pipeline, X_train, y_train, X_test, y_test):
        y_train_pred = train_pipeline.predict(X_train)
        y_test_pred = train_pipeline.predict(X_test)

        evaluation_metric_train = self.config.calculate_evaluation_metric(
            y_train, y_train_pred
        )
        evaluation_metric_test = self.config.calculate_evaluation_metric(
            y_test, y_test_pred
        )
        logging.info(f"Evaluating predictor {self.model_name} finished{LOG_ENDING}")

        return {
            "train": evaluation_metric_train,
            "test": evaluation_metric_test,
        }


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    df_raw, df_train_raw, df_test_raw, df_test_submission_raw = data_ingestion.start()

    data_transformation = DataTransformation()
    df_train, df_test = data_transformation.start(df_train_raw, df_test_raw)

    X_train, X_test = get_X_sets([df_train, df_test])
    y_train, y_test = get_y_sets([df_train, df_test])

    model_name = "rf"

    model_trainer = ModelTrainer(model_name)
    train_pipeline = model_trainer.add_predictor_to_pipeline(Pipeline([]))
    model_trainer.start(train_pipeline, X_train, y_train)

    model_evaluator = ModelEvaluator(model_name)
    evaluation_metrics = model_evaluator.start(
        train_pipeline, X_train, y_train, X_test, y_test
    )

    print(f"train evaluation metric: {evaluation_metrics['train']}")
    print(f"test evaluation metric: {evaluation_metrics['test']}")
