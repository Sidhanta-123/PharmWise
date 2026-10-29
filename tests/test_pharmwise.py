import unittest
import pandas as pd
from fastapi import HTTPException

from src.pipeline.predict_pipeline import PredictPipeline
from api import health_check, analyze_basket, BasketRequest


class TestPharmWiseEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = PredictPipeline()

    def test_pack_size_parser(self):
        qty1, unit1, _ = self.pipeline._parse_pack_size("strip of 15 tablets")
        self.assertEqual((qty1, unit1), (15.0, "UNIT"))

        qty2, unit2, _ = self.pipeline._parse_pack_size("bottle of 100 ml Syrup")
        self.assertEqual((qty2, unit2), (100.0, "ML"))

        qty3, unit3, _ = self.pipeline._parse_pack_size("3 x 10 tablets")
        self.assertEqual((qty3, unit3), (30.0, "UNIT"))

    def test_clinical_ddi_and_duplication(self):
        mock_basket = pd.DataFrame([
            {"name": "Drug A", "chemical_signature": "ACECLOFENAC + PARACETAMOL"},
            {"name": "Drug B", "chemical_signature": "IBUPROFEN"},
            {"name": "Drug C", "chemical_signature": "NIMESULIDE"}
        ])
        alerts = self.pipeline.detect_clinical_interactions(mock_basket)
        categories = [a["category"] for a in alerts]
        self.assertIn("NSAID (Pain / Anti-inflammatory)", categories)
        self.assertIn("Hepatotoxic Analgesic Combination", categories)

    def test_api_health_and_validation(self):
        self.assertEqual(health_check()["status"], "Active")
        with self.assertRaises(HTTPException) as ctx:
            analyze_basket(BasketRequest(medicines=["Augmentin 625 Duo Tablet"]))
        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()