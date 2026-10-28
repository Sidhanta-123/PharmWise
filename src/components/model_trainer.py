import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'kmeans_model.pkl')
    tiered_data_file_path: str = os.path.join('artifacts', 'tiered_medicines.csv')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, df: pd.DataFrame, mfg_stats: pd.DataFrame, X_scaled):
        logging.info("Entered the Model Trainer component")
        try:
            # 1. Fit the optimal Log-Transformed K-Means model (K=3)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            mfg_stats['cluster'] = kmeans.fit_predict(X_scaled)
            logging.info("Fitted K-Means (K=3) clustering model on scaled manufacturer features")

            # 2. Calculate Unsupervised Validation Metrics
            sil_score = silhouette_score(X_scaled, mfg_stats['cluster'])
            db_score = davies_bouldin_score(X_scaled, mfg_stats['cluster'])
            ch_score = calinski_harabasz_score(X_scaled, mfg_stats['cluster'])

            logging.info(
                f"Clustering Metrics -> Silhouette: {sil_score:.4f}, "
                f"Davies-Bouldin: {db_score:.4f}, Calinski-Harabasz: {ch_score:.2f}"
            )

            # 3. Map clusters to logical Trust Tiers based on production volume
            cluster_order = (
                mfg_stats.groupby('cluster')['drug_count']
                .mean()
                .sort_values(ascending=False)
                .index
            )
            tier_map = {
                cluster_order[0]: 'Tier 1 (Premium/High Presence)',
                cluster_order[1]: 'Tier 2 (Standard)',
                cluster_order[2]: 'Tier 3 (Budget/Low Presence)'
            }
            mfg_stats['trust_tier'] = mfg_stats['cluster'].map(tier_map)

            # 4. Merge Trust Tiers back into the main medicine dataset
            if 'trust_tier' in df.columns:
                df = df.drop(columns=['trust_tier'])
                
            df_tiered = df.merge(
                mfg_stats[['manufacturer_name', 'trust_tier']], 
                on='manufacturer_name', 
                how='left'
            )

            # 5. Save the tiered dataset and trained K-Means model artifact
            os.makedirs(os.path.dirname(self.model_trainer_config.tiered_data_file_path), exist_ok=True)
            df_tiered.to_csv(self.model_trainer_config.tiered_data_file_path, index=False)
            logging.info(f"Saved tiered dataset to {self.model_trainer_config.tiered_data_file_path}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=kmeans
            )
            logging.info(f"Saved trained K-Means model to {self.model_trainer_config.trained_model_file_path}")

            return {
                "silhouette_score": round(sil_score, 4),
                "davies_bouldin_score": round(db_score, 4),
                "calinski_harabasz_score": round(ch_score, 2),
                "tiered_data_path": self.model_trainer_config.tiered_data_file_path
            }

        except Exception as e:
            raise CustomException(e, sys)