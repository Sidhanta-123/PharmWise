import os
import sys
import pandas as pd
from collections import Counter
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

@dataclass
class PredictPipelineConfig:
    tiered_data_path: str = os.path.join('artifacts', 'tiered_medicines.csv')

class PredictPipeline:
    def __init__(self):
        self.config = PredictPipelineConfig()

    def load_tiered_data(self) -> pd.DataFrame:
        try:
            if not os.path.exists(self.config.tiered_data_path):
                raise FileNotFoundError(
                    f"Tiered dataset not found at {self.config.tiered_data_path}. "
                    "Please run train_pipeline.py first."
                )
            return pd.read_csv(self.config.tiered_data_path)
        except Exception as e:
            raise CustomException(e, sys)

    def detect_redundancy(self, basket: list, data: pd.DataFrame = None):
        logging.info(f"Running redundancy check for basket: {basket}")
        try:
            if data is None:
                data = self.load_tiered_data()

            basket_df = data[data['name'].isin(basket)].drop_duplicates(subset=['name']).copy()
            
            all_salts = []
            for sig in basket_df['chemical_signature']:
                salts = [s.strip() for s in str(sig).split('+')]
                all_salts.extend(salts)

            salt_counts = Counter(all_salts)
            redundant_salts = {salt: count for salt, count in salt_counts.items() if count > 1}

            logging.info(f"Redundancy check complete. Overlaps found: {redundant_salts}")
            return basket_df, redundant_salts

        except Exception as e:
            raise CustomException(e, sys)

    def optimize_basket_cost(self, basket_df: pd.DataFrame, full_data: pd.DataFrame = None):
        logging.info("Running basket cost optimization for Tier 1 and Tier 2 generics")
        try:
            if full_data is None:
                full_data = self.load_tiered_data()

            recommendations = []
            total_original_cost = 0.0
            total_optimized_cost = 0.0

            for _, row in basket_df.iterrows():
                original_med = row['name']
                chem_sig = row['chemical_signature']
                original_price = float(row['price'])
                total_original_cost += original_price

                # Match exact chemical composition
                alternatives = full_data[full_data['chemical_signature'] == chem_sig]

                # Filter for reliable manufacturers (Tier 1 or Tier 2)
                reliable_alts = alternatives[
                    alternatives['trust_tier'].str.contains('Tier 1|Tier 2', na=False)
                ]

                if not reliable_alts.empty:
                    # Select the lowest-priced option among reliable tiers
                    best_alt = reliable_alts.loc[reliable_alts['price'].idxmin()]
                    optimized_med = best_alt['name']
                    optimized_price = float(best_alt['price'])
                    optimized_tier = best_alt['trust_tier']
                else:
                    optimized_med = original_med
                    optimized_price = original_price
                    optimized_tier = row['trust_tier']

                total_optimized_cost += optimized_price

                recommendations.append({
                    'Original Medicine': original_med,
                    'Original Price (₹)': round(original_price, 2),
                    'Recommended Generic': optimized_med,
                    'Optimized Price (₹)': round(optimized_price, 2),
                    'Savings (₹)': round(original_price - optimized_price, 2),
                    'Generic Tier': optimized_tier
                })

            summary_df = pd.DataFrame(recommendations)
            total_savings = round(total_original_cost - total_optimized_cost, 2)

            logging.info(
                f"Optimization complete. Original: ₹{total_original_cost:.2f}, "
                f"Optimized: ₹{total_optimized_cost:.2f}, Savings: ₹{total_savings:.2f}"
            )

            return (
                summary_df,
                round(total_original_cost, 2),
                round(total_optimized_cost, 2),
                total_savings
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    sample_basket = [
        "Augmentin 625 Duo Tablet",
        "Aldigesic P 100mg/325mg Tablet",
        "ACTION OR 1000mg Tablet"
    ]
    
    predictor = PredictPipeline()
    df_all = predictor.load_tiered_data()
    
    basket_details, overlaps = predictor.detect_redundancy(sample_basket, df_all)
    opt_df, orig_cost, new_cost, savings = predictor.optimize_basket_cost(basket_details, df_all)
    
    print("Predict Pipeline Executed Successfully!")
    print(f"Overdose Risk Salts: {overlaps}")
    print(f"Original Cost: ₹{orig_cost:.2f} | Optimized Cost: ₹{new_cost:.2f} | Total Savings: ₹{savings:.2f}\n")
    print(opt_df.to_string(index=False))