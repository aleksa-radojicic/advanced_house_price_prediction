import os
import sys
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

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
        column_transformer = create_column_transformer()

        transformer_pipeline = Pipeline(
            steps=[
                ("univariate_analysis", ua_transformer),
                ("multivariate_analysis", ma_transformer),
                ("feature_engineering", fe_transformer),
                ("post_feature_engineering_analysis", pfea_transformer),
                ("drop_columns_scheduled_for_deletion", drop_columns_scheduled_for_deletion_transformer),
                ("column_dt_prefixer", column_dt_prefixer),
                ("column_transformer", column_transformer)
            ]
        ).set_output(transform="pandas")
        return transformer_pipeline

    def start(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logging.info(msg="Data transformation method or component started")

        try:
            preprocessing_obj = self.create_data_transformer_object()

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            df_train_preprocessed = preprocessing_obj.fit_transform(df_train)  # type: ignore
            df_test_preprocessed = preprocessing_obj.transform(df_test)  # type: ignore

            # The deletion of rows should be handled differently
            delete_rows(df_train_preprocessed)
            logging.info("Appropriate rows deleted.")

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

def create_column_transformer():
    ct = ColumnTransformer(
        [
            # ("numerical", MinMaxScaler(), make_column_selector("numerical__")),
            ("numerical", "passthrough", make_column_selector("numerical__")),
            # ("binary", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), features_info['binary']),
            ("binary", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), make_column_selector(pattern="binary__")),
            # ("ordinal", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), features_info["ordinal"]),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), make_column_selector(pattern="ordinal__")),
            ("nominal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int16), make_column_selector(pattern="nominal__")),
            ("label", "passthrough", [LABEL])
            # ("nominal", OneHotEncoder(handle_unknown='ignore', dtype=np.int8, sparse_output=False), features_info["nominal"])
        ],
        remainder="drop",
        verbose_feature_names_out=False,  # False because prefixes are added manually
    ).set_output(transform="pandas")
    return ct

def delete_rows(df: pd.DataFrame):
    df.drop([598], inplace=True)

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    df, df_train, df_test, df_test_submission = data_ingestion.start()

    data_transformation = DataTransformation()
    df_train_preprocessed, df_test_preprocessed = data_transformation.start(
        df_train, df_test
    )
