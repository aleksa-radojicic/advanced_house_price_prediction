from dataclasses import dataclass

import numpy as np
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.pipeline import Pipeline

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation.data_transformation import DataTransformation
from src.components.data_transformation.utils import LabelTransformer
from src.components.model_trainer import ModelTrainer
from src.logger import LOG_ENDING, logging
from src.utils import get_X_sets, get_y_sets

def wrapper_compute_evaluation_metric(y_true, y_pred):
    return ModelEvaluatorConfig.compute_evaluation_metric(y_true, y_pred)


class ModelEvaluatorConfig:
    evaluation_metric_name = "root_mean_squared_error"
    greater_is_better = False

    @classmethod
    def compute_evaluation_metric(cls, y_true, y_pred):
        root_mean_squared_error = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        return root_mean_squared_error
    
    scorer = make_scorer(score_func=wrapper_compute_evaluation_metric, greater_is_better=greater_is_better)

    @classmethod
    def get_scorer(cls):
        return cls.scorer
    

class ModelEvaluator:
    def __init__(self, model_name):
        self.config = ModelEvaluatorConfig()
        self.model_name = model_name

    def start(self, train_pipeline: Pipeline, X_train, y_train, X_test, y_test):
        y_train_pred = train_pipeline.predict(X_train)
        y_test_pred = train_pipeline.predict(X_test)

        evaluation_metric_train = self.config.compute_evaluation_metric(
            y_train, y_train_pred
        )
        evaluation_metric_test = self.config.compute_evaluation_metric(
            y_test, y_test_pred
        )
        logging.info(f"Evaluating predictor {self.model_name} finished{LOG_ENDING}")

        return {
            "train": evaluation_metric_train,
            "test": evaluation_metric_test,
        }


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    df, df_train, df_test, df_test_submission = data_ingestion.start()

    data_transformation = DataTransformation()
    pipeline = Pipeline(
        [
            (
                "data_transformation",
                data_transformation.create_data_transformation_pipeline(),
            )
        ]
    )
    label_transformer = LabelTransformer()

    X_train, X_test = get_X_sets([df_train, df_test])
    y_train, y_test = get_y_sets([df_train, df_test])

    y_train_transformed = label_transformer.fit_transform(y_train)  # type: ignore
    y_test_transformed = label_transformer.fit_transform(y_test)  # type: ignore

    model_name = "rf"

    model_trainer = ModelTrainer(model_name)
    pipeline = model_trainer.add_predictor_to_pipeline(pipeline)
    model_trainer.start(pipeline, X_train, y_train_transformed)

    model_evaluator = ModelEvaluator(model_name)
    evaluation_metrics = model_evaluator.start(
        pipeline, X_train, y_train_transformed, X_test, y_test_transformed
    )

    print(f"train evaluation metric: {evaluation_metrics['train']}")
    print(f"test evaluation metric: {evaluation_metrics['test']}")
