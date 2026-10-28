import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        logging.info("Starting the PharmWise Training Pipeline")
        try:
            # Data Ingestion & ETL
            clean_data_path = self.data_ingestion.initiate_data_ingestion()

            # Feature Engineering, Log Transform & Scaling
            df, mfg_stats, X_scaled, _ = self.data_transformation.initiate_data_transformation(
                clean_data_path=clean_data_path
            )

            # Model Training & Tier Assignment
            metrics = self.model_trainer.initiate_model_trainer(
                df=df,
                mfg_stats=mfg_stats,
                X_scaled=X_scaled
            )

            logging.info(f"Training Pipeline completed successfully. Metrics: {metrics}")
            return metrics

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    results = pipeline.run_pipeline()
    print("Training Pipeline Executed Successfully!")
    print(f"Silhouette Score: {results['silhouette_score']}")
    print(f"Davies-Bouldin Index: {results['davies_bouldin_score']}")
    print(f"Calinski-Harabasz Score: {results['calinski_harabasz_score']}")