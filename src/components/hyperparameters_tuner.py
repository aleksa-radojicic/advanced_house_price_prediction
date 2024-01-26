from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

import numpy as np

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation.data_transformation import \
    DataTransformation
from src.components.data_transformation.utils import LabelTransformer
from src.components.model_evaluator import ModelEvaluatorConfig
from src.components.model_trainer import ModelTrainer
from src.config import RANDOM_SEED
from src.logger import LOG_ENDING, log_message
from src.utils import get_X_sets, get_y_sets


class HyperparametersTunerConfig:
    types: dict = {
        "grid": lambda estimator, param_grid, cv: GridSearchCV(
            estimator,
            param_grid=param_grid,
            cv=cv,
            n_jobs=-1,
            return_train_score=True,
            verbose=True,
            refit=True,
            scoring=ModelEvaluatorConfig.get_scorer(),
            error_score='raise'
        )
    }


class HyperparametersTuner:
    def __init__(self, type: str, verbose: int = 0):
        self.config = HyperparametersTunerConfig()

        if type not in self.config.types:
            raise ValueError("Type of hyperparameters tuning not recognized!")

        self.type = type
        self.hyperparameters_tuner_partial = self.config.types[type]
        self.verbose = verbose

    def start(self, estimator, param_grid, X, y, cv):
        self.hyperparameters_tuner = self.hyperparameters_tuner_partial(
            estimator, param_grid, cv
        )

        log_message(f"Hyperparameter tuning {self.type} started.", self.verbose)
        self.hyperparameters_tuner.fit(X, y)
        log_message(
            f"Hyperparameter tuning {self.type} finished{LOG_ENDING}", self.verbose
        )
        return self.hyperparameters_tuner


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

    model_name = "ridge"

    model_trainer = ModelTrainer(model_name)
    pipeline = model_trainer.add_predictor_to_pipeline(pipeline)

    hyperparameters_tuner = HyperparametersTuner("grid", verbose=1)

    param_grid = {
        "predictor__alpha": np.linspace(0, 1, 100)
    }
    cv_no = 5
    cv = KFold(cv_no, shuffle=True, random_state=RANDOM_SEED)

    cv_obj = hyperparameters_tuner.start(pipeline, param_grid, X_train, y_train_transformed, cv)
    print(cv_obj.cv_results_)
    print(cv_obj.best_score_)