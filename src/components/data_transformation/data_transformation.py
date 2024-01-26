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
from src.components.data_transformation.utils import (
    ColumnDtPrefixerTransformer, DropColumnsScheduledForDeletionTransformer,
    LabelTransformer)
from src.config import LABEL
from src.utils import FeaturesInfo, get_X_sets, get_y_sets


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self, verbose: int = 0):
        self.verbose = verbose
        self.data_transformation_config = DataTransformationConfig()
        self.idx_to_remove: List[int] = []

    def create_data_transformation_pipeline(self):
        features_info: FeaturesInfo = {}

        ua_transformer = UnivariateAnalysisTransformer(features_info, self.verbose)
        
        ma_transformer = MultivariateAnalysisTransformer(self.verbose)
        ma_transformer.previous_transformer_obj = ua_transformer # type: ignore
        
        fe_transformer = FeatureEngineeringTransformer(self.verbose)
        fe_transformer.previous_transformer_obj = ma_transformer # type: ignore

        pfea_transformer = PostFEAnalysisTransformer(self.verbose)
        pfea_transformer.previous_transformer_obj = fe_transformer # type: ignore

        drop_columns_scheduled_for_deletion_transformer = (
            DropColumnsScheduledForDeletionTransformer(self.verbose)
        )
        drop_columns_scheduled_for_deletion_transformer.previous_transformer_obj = pfea_transformer # type: ignore

        
        column_dt_prefixer = ColumnDtPrefixerTransformer(self.verbose
        )
        column_dt_prefixer.previous_transformer_obj = drop_columns_scheduled_for_deletion_transformer # type: ignore

        column_transformer = create_column_transformer()

        data_transformation_pipeline = Pipeline([
                ("univariate_analysis", ua_transformer),
                ("multivariate_analysis", ma_transformer),
                ("feature_engineering", fe_transformer),
                ("post_feature_engineering_analysis", pfea_transformer),
                ("drop_columns_scheduled_for_deletion", drop_columns_scheduled_for_deletion_transformer),
                ("column_dt_prefixer", column_dt_prefixer),
                ("column_transformer", column_transformer)
            ]
        ).set_output(transform="pandas")
        return data_transformation_pipeline

def create_column_transformer():
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

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    df, df_train, df_test, df_test_submission = data_ingestion.start()

    data_transformation = DataTransformation()
    pipeline = Pipeline([("data_transformation", data_transformation.create_data_transformation_pipeline())])
    
    label_transformer = LabelTransformer()

    X_train, X_test = get_X_sets([df_train, df_test])
    y_train, y_test = get_y_sets([df_train, df_test])

    X_train_transformed = pipeline.fit_transform(X_train) # type: ignore
    print(X_train_transformed.info()) # type: ignore
    X_test_transformed = pipeline.transform(X_test) # type: ignore
    print(X_test_transformed.info()) # type: ignore

    y_train_transformed = label_transformer.fit_transform(y_train) # type: ignore
    y_test_transformed = label_transformer.fit_transform(y_test) # type: ignore
