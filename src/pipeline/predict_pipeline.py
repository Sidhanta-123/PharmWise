import os
import re
import sys
import pandas as pd
from collections import Counter
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any

from src.exception import CustomException
from src.logger import logging

@dataclass
class PredictPipelineConfig:
    tiered_data_path: str = os.path.join('artifacts', 'tiered_medicines.csv')


class PredictPipeline:
    THERAPEUTIC_CLASSES = {
        "NSAID (Pain / Anti-inflammatory)": {
            "keywords": [
                "IBUPROFEN", "DICLOFENAC", "ACECLOFENAC", "NAPROXEN", "NIMESULIDE",
                "ETORICOXIB", "KETOROLAC", "PIROXICAM", "MEFENAMIC", "INDOMETHACIN", "CELECOXIB"
            ],
            "severity": "High",
            "note": "Concurrent use of multiple NSAIDs markedly increases gastrointestinal bleeding and renal toxicity risk without added analgesia."
        },
        "Proton Pump Inhibitor (Acid Suppressant)": {
            "keywords": [
                "PANTOPRAZOLE", "OMEPRAZOLE", "RABEPRAZOLE", "ESOMEPRAZOLE",
                "LANSOPRAZOLE", "DEXLANSOPRAZOLE"
            ],
            "severity": "Moderate",
            "note": "Duplicate PPI therapy; combining two proton pump inhibitors rarely provides additional acid suppression."
        },
        "H1 Antihistamine (Anti-allergic)": {
            "keywords": [
                "CETIRIZINE", "LEVOCETIRIZINE", "FEXOFENADINE", "LORATADINE",
                "DESLORATADINE", "CHLORPHENIRAMINE", "HYDROXYZINE"
            ],
            "severity": "Moderate",
            "note": "Combining multiple antihistamines compounds drowsiness, CNS depression, and anticholinergic effects."
        },
        "Statin (Lipid-Lowering)": {
            "keywords": [
                "ATORVASTATIN", "ROSUVASTATIN", "SIMVASTATIN", "PRAVASTATIN", "PITAVASTATIN"
            ],
            "severity": "High",
            "note": "Concurrent statin therapy significantly elevates the risk of myopathy and rhabdomyolysis."
        },
        "Sulfonylurea (Antidiabetic)": {
            "keywords": ["GLIMEPIRIDE", "GLICLAZIDE", "GLIPIZIDE", "GLIBENCLAMIDE"],
            "severity": "High",
            "note": "Combining two sulfonylureas creates a high risk of severe hypoglycemia."
        }
    }

    CROSS_CLASS_DDIS = [
        {
            "group_a": ["PARACETAMOL"],
            "group_b": ["NIMESULIDE"],
            "category": "Hepatotoxic Analgesic Combination",
            "severity": "High",
            "note": "Concurrent Paracetamol and Nimesulide increases the risk of acute drug-induced liver injury."
        },
        {
            "group_a": ["ASPIRIN", "WARFARIN", "CLOPIDOGREL", "RIVAROXABAN", "APIXABAN", "DABIGATRAN"],
            "group_b": ["IBUPROFEN", "DICLOFENAC", "ACECLOFENAC", "NAPROXEN", "KETOROLAC", "ETORICOXIB", "PIROXICAM"],
            "category": "Antithrombotic + NSAID Interaction",
            "severity": "High",
            "note": "Combining blood thinners/antiplatelets with NSAIDs substantially elevates gastrointestinal and systemic bleeding risk."
        },
        {
            "group_a": ["AZITHROMYCIN", "CLARITHROMYCIN", "ERYTHROMYCIN"],
            "group_b": ["ONDANSETRON", "DOMPERIDONE", "LEVOFLOXACIN", "MOXIFLOXACIN", "OFLOXACIN"],
            "category": "QT-Prolongation Interaction",
            "severity": "Moderate-High",
            "note": "Co-administration may cause additive QT-interval prolongation and cardiac arrhythmia risk."
        },
        {
            "group_a": ["SILDENAFIL", "TADALAFIL"],
            "group_b": ["NITROGLYCERIN", "ISOSORBIDE"],
            "category": "PDE5 Inhibitor + Nitrate Contraindication",
            "severity": "Critical",
            "note": "Strictly contraindicated; synergistic vasodilation can trigger life-threatening hypotension."
        }
    ]

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

    def _extract_dosage_signature(self, row) -> str:
        """Extracts normalized active salt + exact strength (e.g., PARACETAMOL(1000MG))."""
        parts = []
        for col in ['short_composition1', 'short_composition2']:
            if col in row.index and pd.notna(row[col]) and str(row[col]).strip() != "":
                cleaned = re.sub(r'\s+', '', str(row[col]).upper())
                parts.append(cleaned)
        if parts:
            med_name = str(row.get('name', '')).upper()
            name_doses = re.findall(r'\d+(?:\.\d+)?\s*(?:MG|MCG|GM|IU)', med_name)
            if len(name_doses) > len(parts):
                extra = "+".join(re.sub(r'\s+', '', d) for d in name_doses)
                return " + ".join(sorted(parts)) + f"|{extra}"
            return " + ".join(sorted(parts))

        med_name = str(row.get('name', '')).upper()
        strengths = re.findall(r'\d+(?:\.\d+)?\s*(?:MG|MCG|GM|G|ML|IU|%)', med_name)
        strength_norm = "+".join(sorted([re.sub(r'\s+', '', s) for s in strengths]))
        return f"{row.get('chemical_signature', '')}|{strength_norm}"

    def _extract_form(self, name: str) -> str:
        """Categorizes delivery form so tablets aren't swapped with syrups or injections."""
        name_upper = str(name).upper()
        if re.search(r'\b(TABLET|TABLETS|TAB|TABS)\b', name_upper):
            return 'TABLET'
        if re.search(r'\b(CAPSULE|CAPSULES|CAP|CAPS)\b', name_upper):
            return 'CAPSULE'
        if re.search(r'\b(SYRUP|SUSPENSION|LIQUID|ELIXIR)\b', name_upper):
            return 'ORAL_LIQUID'
        if re.search(r'\b(INJECTION|INJ|VIAL|AMPOULE)\b', name_upper):
            return 'INJECTION'
        if re.search(r'\b(DROP|DROPS)\b', name_upper):
            return 'DROPS'
        if re.search(r'\b(CREAM|GEL|OINTMENT|LOTION)\b', name_upper):
            return 'TOPICAL'
        if re.search(r'\b(INHALER|ROTACAP|RESPULES)\b', name_upper):
            return 'INHALER'
        return 'OTHER'

    def _parse_pack_size(self, pack_label) -> Tuple[float, str, str]:
        """Parses pack_size_label for unit price calculations."""
        if pd.isna(pack_label) or str(pack_label).strip() == "":
            return 1.0, "PACK", "1 Pack"

        raw = str(pack_label).strip()
        lower = raw.lower()

        multi_match = re.search(r'(\d+)\s*[x×]\s*(\d+(?:\.\d+)?)', lower)
        if multi_match:
            qty = float(multi_match.group(1)) * float(multi_match.group(2))
            return (qty if qty > 0 else 1.0), "UNIT", raw

        ml_match = re.search(r'(\d+(?:\.\d+)?)\s*ml\b', lower)
        if ml_match:
            qty = float(ml_match.group(1))
            return (qty if qty > 0 else 1.0), "ML", raw

        gm_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:gm|g)\b', lower)
        if gm_match:
            qty = float(gm_match.group(1))
            return (qty if qty > 0 else 1.0), "GM", raw

        dose_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:mdi|metered\s*doses?|doses?|puffs?)\b', lower)
        if dose_match:
            qty = float(dose_match.group(1))
            return (qty if qty > 0 else 1.0), "DOSE", raw

        count_match = re.search(
            r'(?:of\s+)?(\d+(?:\.\d+)?)\s*(?:tablet|cap|sachet|respule|rotacap|vial|ampoule|patch|suppositor|lozenge|strip|unit)',
            lower
        )
        if count_match:
            qty = float(count_match.group(1))
            return (qty if qty > 0 else 1.0), "UNIT", raw

        generic_of = re.search(r'\bof\s+(\d+(?:\.\d+)?)\b', lower)
        if generic_of:
            qty = float(generic_of.group(1))
            return (qty if qty > 0 else 1.0), "UNIT", raw

        return 1.0, "PACK", raw

    def _salt_matches_keywords(self, salt: str, keywords: List[str]) -> bool:
        for kw in keywords:
            if re.search(rf'\b{re.escape(kw)}\b', salt.upper()):
                return True
        return False

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

    def detect_clinical_interactions(self, basket_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detects cross-salt therapeutic duplications and high-risk Drug-Drug Interactions."""
        try:
            salt_to_meds: Dict[str, List[str]] = {}
            for _, row in basket_df.iterrows():
                med = row['name']
                salts = [s.strip() for s in str(row['chemical_signature']).split('+')]
                for salt in salts:
                    salt_to_meds.setdefault(salt, [])
                    if med not in salt_to_meds[salt]:
                        salt_to_meds[salt].append(med)

            unique_salts = list(salt_to_meds.keys())
            alerts: List[Dict[str, Any]] = []

            for class_name, rule in self.THERAPEUTIC_CLASSES.items():
                matched_salts = [s for s in unique_salts if self._salt_matches_keywords(s, rule["keywords"])]
                if len(matched_salts) >= 2:
                    involved_meds = sorted({m for s in matched_salts for m in salt_to_meds[s]})
                    if len(involved_meds) >= 2:
                        alerts.append({
                            "type": "Therapeutic Duplication",
                            "category": class_name,
                            "severity": rule["severity"],
                            "salts_involved": sorted(matched_salts),
                            "medicines_involved": involved_meds,
                            "clinical_note": rule["note"]
                        })

            for ddi in self.CROSS_CLASS_DDIS:
                salts_a = [s for s in unique_salts if self._salt_matches_keywords(s, ddi["group_a"])]
                salts_b = [s for s in unique_salts if self._salt_matches_keywords(s, ddi["group_b"])]
                if salts_a and salts_b:
                    combined_salts = sorted(set(salts_a + salts_b))
                    involved_meds = sorted({m for s in combined_salts for m in salt_to_meds[s]})
                    if len(involved_meds) >= 2 or ddi["category"] == "Hepatotoxic Analgesic Combination":
                        alerts.append({
                            "type": "Drug-Drug Interaction",
                            "category": ddi["category"],
                            "severity": ddi["severity"],
                            "salts_involved": combined_salts,
                            "medicines_involved": involved_meds,
                            "clinical_note": ddi["note"]
                        })

            logging.info(f"Clinical interaction scan complete. Alerts found: {len(alerts)}")
            return alerts

        except Exception as e:
            raise CustomException(e, sys)

    def optimize_basket_cost(self, basket_df: pd.DataFrame, full_data: pd.DataFrame = None):
        logging.info("Running unit-normalized, exact-dosage basket cost optimization")
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
                orig_dosage_sig = self._extract_dosage_signature(row)
                orig_form = self._extract_form(original_med)
                orig_qty, orig_measure, orig_pack_label = self._parse_pack_size(
                    row.get('pack_size_label', None)
                )
                orig_unit_price = original_price / orig_qty

                total_original_cost += original_price

                alternatives = full_data[full_data['chemical_signature'] == chem_sig]
                reliable_alts = alternatives[
                    alternatives['trust_tier'].str.contains('Tier 1|Tier 2', na=False)
                ].copy()

                optimized_med = original_med
                gen_pack_label = orig_pack_label
                gen_pack_price = original_price
                gen_unit_price = orig_unit_price
                eqv_dose_cost = original_price
                optimized_tier = row['trust_tier']
                match_status = "Original Retained (No Exact-Dose Generic)"

                if not reliable_alts.empty:
                    reliable_alts['_dosage_sig'] = reliable_alts.apply(self._extract_dosage_signature, axis=1)
                    reliable_alts['_form'] = reliable_alts['name'].apply(self._extract_form)

                    exact_alts = reliable_alts[
                        (reliable_alts['_dosage_sig'] == orig_dosage_sig) &
                        (reliable_alts['_form'] == orig_form)
                    ].copy()

                    if not exact_alts.empty:
                        if 'pack_size_label' in exact_alts.columns:
                            parsed_packs = exact_alts['pack_size_label'].apply(self._parse_pack_size)
                        else:
                            parsed_packs = pd.Series(
                                [(1.0, "PACK", "1 Pack")] * len(exact_alts),
                                index=exact_alts.index
                            )

                        exact_alts['_qty'] = [p[0] for p in parsed_packs]
                        exact_alts['_measure'] = [p[1] for p in parsed_packs]
                        exact_alts['_pack_label'] = [p[2] for p in parsed_packs]
                        exact_alts['_unit_price'] = exact_alts['price'] / exact_alts['_qty']

                        same_measure_alts = (
                            exact_alts[exact_alts['_measure'] == orig_measure]
                            if orig_measure != "PACK" else pd.DataFrame()
                        )

                        if not same_measure_alts.empty:
                            best_alt = same_measure_alts.loc[same_measure_alts['_unit_price'].idxmin()]
                            if float(best_alt['_unit_price']) < orig_unit_price - 1e-6:
                                optimized_med = best_alt['name']
                                gen_pack_label = best_alt['_pack_label']
                                gen_pack_price = float(best_alt['price'])
                                gen_unit_price = float(best_alt['_unit_price'])
                                eqv_dose_cost = gen_unit_price * orig_qty
                                optimized_tier = best_alt['trust_tier']
                                match_status = "Unit-Normalized Exact Match"
                            else:
                                match_status = "Original Already Lowest Unit Cost"
                        else:
                            best_alt = exact_alts.loc[exact_alts['price'].idxmin()]
                            if float(best_alt['price']) < original_price - 1e-6:
                                optimized_med = best_alt['name']
                                gen_pack_label = best_alt['_pack_label']
                                gen_pack_price = float(best_alt['price'])
                                gen_unit_price = float(best_alt['_unit_price'])
                                eqv_dose_cost = gen_pack_price
                                optimized_tier = best_alt['trust_tier']
                                match_status = "Pack-Level Exact Match"
                            else:
                                match_status = "Original Already Lowest Price"

                total_optimized_cost += eqv_dose_cost

                recommendations.append({
                    'Original Medicine': original_med,
                    'Original Pack': orig_pack_label,
                    'Original Price (₹)': round(original_price, 2),
                    'Orig Unit Cost (₹)': round(orig_unit_price, 2),
                    'Recommended Generic': optimized_med,
                    'Generic Pack': gen_pack_label,
                    'Generic Pack Price (₹)': round(gen_pack_price, 2),
                    'Gen Unit Cost (₹)': round(gen_unit_price, 2),
                    'Eqv. Dose Cost (₹)': round(eqv_dose_cost, 2),
                    'Savings (₹)': round(original_price - eqv_dose_cost, 2),
                    'Generic Tier': optimized_tier,
                    'Match Status': match_status
                })

            summary_df = pd.DataFrame(recommendations)
            total_savings = round(total_original_cost - total_optimized_cost, 2)

            logging.info(
                f"Unit-normalized optimization complete. Original: ₹{total_original_cost:.2f}, "
                f"Eqv Optimized: ₹{total_optimized_cost:.2f}, Savings: ₹{total_savings:.2f}"
            )

            return (
                summary_df,
                round(total_original_cost, 2),
                round(total_optimized_cost, 2),
                total_savings
            )

        except Exception as e:
            raise CustomException(e, sys)

    def get_drug_alternatives(self, medicine_name: str, full_data: pd.DataFrame = None):
        """Returns exact-strength alternatives sorted by per-unit price."""
        try:
            if full_data is None:
                full_data = self.load_tiered_data()

            target_rows = full_data[full_data['name'] == medicine_name]
            if target_rows.empty:
                return None, pd.DataFrame()

            target = target_rows.iloc[0]
            chem_sig = target['chemical_signature']
            orig_dosage_sig = self._extract_dosage_signature(target)
            orig_form = self._extract_form(medicine_name)
            orig_qty, orig_measure, orig_pack_label = self._parse_pack_size(
                target.get('pack_size_label', None)
            )
            orig_price = float(target['price'])
            orig_unit_price = orig_price / orig_qty

            alts = full_data[full_data['chemical_signature'] == chem_sig].copy()
            alts['_dosage_sig'] = alts.apply(self._extract_dosage_signature, axis=1)
            alts['_form'] = alts['name'].apply(self._extract_form)

            exact_alts = alts[
                (alts['_dosage_sig'] == orig_dosage_sig) &
                (alts['_form'] == orig_form)
            ].drop_duplicates(subset=['name']).copy()

            if not exact_alts.empty:
                if 'pack_size_label' in exact_alts.columns:
                    parsed = exact_alts['pack_size_label'].apply(self._parse_pack_size)
                else:
                    parsed = pd.Series([(1.0, "PACK", "1 Pack")] * len(exact_alts), index=exact_alts.index)

                exact_alts['pack_label'] = [p[2] for p in parsed]
                exact_alts['_qty'] = [p[0] for p in parsed]
                exact_alts['_measure'] = [p[1] for p in parsed]
                exact_alts['unit_price'] = (exact_alts['price'] / exact_alts['_qty']).round(2)
                exact_alts['eqv_dose_price'] = exact_alts.apply(
                    lambda r: round((r['price'] / r['_qty']) * orig_qty, 2)
                    if r['_measure'] == orig_measure and orig_measure != "PACK"
                    else round(float(r['price']), 2),
                    axis=1
                )
                exact_alts = exact_alts.sort_values(by=['unit_price', 'price'], ascending=True)

            target_info = {
                "name": target['name'],
                "manufacturer": target['manufacturer_name'],
                "price": round(orig_price, 2),
                "pack_label": orig_pack_label,
                "unit_qty": orig_qty,
                "unit_measure": orig_measure,
                "unit_price": round(orig_unit_price, 2),
                "chemical_signature": chem_sig,
                "dosage_signature": orig_dosage_sig,
                "form": orig_form,
                "trust_tier": target['trust_tier']
            }

            result_cols = [
                'name', 'manufacturer_name', 'pack_label',
                'price', 'unit_price', 'eqv_dose_price', 'trust_tier'
            ]
            available_cols = [c for c in result_cols if c in exact_alts.columns]
            return target_info, exact_alts[available_cols].head(25)

        except Exception as e:
            raise CustomException(e, sys)

    def get_cluster_analytics(self, full_data: pd.DataFrame = None):
        """Aggregates manufacturer and price statistics across K-Means Trust Tiers."""
        try:
            if full_data is None:
                full_data = self.load_tiered_data()

            tier_summary = (
                full_data.groupby('trust_tier')
                .agg(
                    total_medicines=('name', 'count'),
                    unique_manufacturers=('manufacturer_name', 'nunique'),
                    avg_price=('price', 'mean'),
                    median_price=('price', 'median')
                )
                .reset_index()
                .sort_values(by='trust_tier')
            )
            tier_summary['avg_price'] = tier_summary['avg_price'].round(2)
            tier_summary['median_price'] = tier_summary['median_price'].round(2)

            return tier_summary.to_dict(orient='records')

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
    interactions = predictor.detect_clinical_interactions(basket_details)
    opt_df, orig_cost, new_cost, savings = predictor.optimize_basket_cost(basket_details, df_all)

    print("Predict Pipeline v2.1 Executed Successfully!")
    print(f"Exact Salt Overlaps: {overlaps}")
    print(f"Cross-Salt Clinical Alerts: {len(interactions)}")
    print(f"Original Cost: ₹{orig_cost:.2f} | Eqv. Optimized Cost: ₹{new_cost:.2f} | Net Savings: ₹{savings:.2f}\n")
    print(opt_df[[
        'Original Medicine', 'Orig Unit Cost (₹)',
        'Recommended Generic', 'Gen Unit Cost (₹)',
        'Eqv. Dose Cost (₹)', 'Savings (₹)'
    ]].to_string(index=False))