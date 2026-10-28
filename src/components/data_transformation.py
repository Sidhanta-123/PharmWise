import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    scaler_obj_file_path: str = os.path.join('artifacts', 'scaler.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, clean_data_path: str):
        logging.info("Entered the data transformation component")
        try:
            df = pd.read_csv(clean_data_path)
            logging.info(f"Loaded clean dataset from {clean_data_path} with shape: {df.shape}")

            # Feature Engineering: Aggregate manufacturer metrics
            mfg_stats = df.groupby('manufacturer_name').agg(
                drug_count=('name', 'count'),
                avg_price=('price', 'mean')
            ).reset_index()
            logging.info(f"Aggregated stats for {len(mfg_stats)} unique manufacturers")

            # Log Transformation to fix extreme right-skewness in drug volume
            mfg_stats['log_drug_count'] = np.log1p(mfg_stats['drug_count'])

            # Feature Scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(mfg_stats[['log_drug_count', 'avg_price']])
            logging.info("Applied log transformation and StandardScaler to manufacturer features")

            # Save the scaler artifact
            save_object(
                file_path=self.transformation_config.scaler_obj_file_path,
                obj=scaler
            )
            logging.info(f"Saved scaler object to {self.transformation_config.scaler_obj_file_path}")

            return (
                df,
                mfg_stats,
                X_scaled,
                self.transformation_config.scaler_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)