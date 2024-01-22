import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation.feature_engineering import \
    FeatureEngineeringTransformer
from src.components.data_transformation.multivariate_analysis import \
    MultivariateAnalysisTransformer
from src.components.data_transformation.post_feature_engineering_analysis import \
    PostFEAnalysisTransformer
from src.components.data_transformation.univariate_analysis import \
    UnivariateAnalysisTransformer
from src.components.data_transformation.utils import \
    ColumnDtPrefixerTransformer, DropColumnsScheduledForDeletionTransformer
from src.config import LABEL
from src.exception import CustomException
from src.logger import LOG_ENDING, logging
from src.utils import FeaturesInfo


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.idx_to_remove: List[int] = []

    def create_data_transformer_object(self):
        features_info: FeaturesInfo = {}

        ua_transformer = UnivariateAnalysisTransformer(features_info)
        ma_transformer = MultivariateAnalysisTransformer(ua_transformer)
        fe_transformer = FeatureEngineeringTransformer(ma_transformer)
        pfea_transformer = PostFEAnalysisTransformer(fe_transformer)
        drop_columns_scheduled_for_deletion_transformer = DropColumnsScheduledForDeletionTransformer(pfea_transformer)
        column_dt_prefixer = ColumnDtPrefixerTransformer(drop_columns_scheduled_for_deletion_transformer)

        transformer_pipeline = Pipeline(
            steps=[
                ("univariate_analysis", ua_transformer),
                ("multivariate_analysis", ma_transformer),
                ("feature_engineering", fe_transformer),
                ("post_feature_engineering_analysis", pfea_transformer),
                ("drop_columns_scheduled_for_deletion", drop_columns_scheduled_for_deletion_transformer),
                ("column_dt_prefixer", column_dt_prefixer),
            ]
        ).set_output(transform="pandas")
        return transformer_pipeline

    def start(self, df_train, df_test) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logging.info(msg="Data transformation method or component started")

        try:
            preprocessing_obj = self.create_data_transformer_object()

            X_train = df_train.drop(LABEL, axis=1)
            X_test = df_test.drop(LABEL, axis=1)

            y_train = df_train[LABEL]
            y_test = df_test[LABEL]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            X_train_preprocessed = preprocessing_obj.fit_transform(X_train)  # type: ignore
            X_test_preprocessed = preprocessing_obj.transform(X_test)  # type: ignore

            df_train_preprocessed = pd.concat([X_train_preprocessed, y_train], axis=1)
            df_test_preprocessed = pd.concat([X_test_preprocessed, y_test], axis=1)

            # logging.info(f"Saved preprocessing objects.")

            # save_object(
            #     file_path=self.data_transformation_config.preprocessor_obj_file_path,
            #     obj=preprocessing_obj,
            # )
        except Exception as e:
            raise CustomException(e, sys)  # type: ignore

        logging.info(
            msg="Data transformation method or component finished" + LOG_ENDING
        )

        return df_train_preprocessed, df_test_preprocessed


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    df, df_train, df_test, df_test_submission = data_ingestion.start()

    data_transformation = DataTransformation()
    df_train_preprocessed, df_test_preprocessed = data_transformation.start(
        df_train, df_test
    )
