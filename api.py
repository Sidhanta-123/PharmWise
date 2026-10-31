from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List
import uvicorn
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline

app = FastAPI(
    title="PharmWise API",
    description="Backend ML & Pharmacological Safety API for PharmWise",
    version="2.1"
)

predictor = PredictPipeline()
df_all = predictor.load_tiered_data()

class BasketRequest(BaseModel):
    medicines: List[str]

@app.get("/")
def health_check():
    return {"status": "Active", "message": "PharmWise API v2.1 is running."}

@app.get("/medicines")
def get_medicine_list():
    medicines = df_all['name'].dropna().unique().tolist()
    return {"medicines": medicines}

@app.get("/medicines/search")
def search_medicines(
    q: str = Query(..., min_length=2, description="Brand name or active salt keyword"),
    limit: int = Query(25, ge=1, le=100)
):
    query_upper = q.strip().upper()
    matches = df_all[
        df_all['name'].str.upper().str.contains(query_upper, na=False, regex=False) |
        df_all['chemical_signature'].str.upper().str.contains(query_upper, na=False, regex=False)
    ].drop_duplicates(subset=['name']).head(limit)

    cols = ['name', 'manufacturer_name', 'chemical_signature', 'price', 'trust_tier']
    # Safely fill missing values so JSON doesn't crash
    safe_matches = matches[cols].fillna("")
    
    return {
        "query": q,
        "count": len(matches),
        "results": safe_matches.to_dict(orient="records")
    }

@app.post("/analyze")
def analyze_basket(request: BasketRequest):
    if len(request.medicines) < 2:
        raise HTTPException(status_code=400, detail="Please provide at least 2 medicines to analyze.")

    try:
        basket_df, overlaps = predictor.detect_redundancy(request.medicines, df_all)
        clinical_interactions = predictor.detect_clinical_interactions(basket_df)
        opt_df, orig_cost, new_cost, savings = predictor.optimize_basket_cost(basket_df, df_all)

        safe_basket = basket_df.fillna("")
        safe_opt = opt_df.fillna("")

        return {
            "overlaps": overlaps,
            "clinical_interactions": clinical_interactions,
            "basket_details": safe_basket.to_dict(orient="records"),
            "optimization_table": safe_opt.to_dict(orient="records"),
            "metrics": {
                "original_cost": orig_cost,
                "optimized_cost": new_cost,
                "total_savings": savings,
                "interaction_count": len(clinical_interactions)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alternatives")
def get_alternatives(medicine_name: str = Query(..., description="Exact name of the medicine")):
    try:
        target_info, alts_df = predictor.get_drug_alternatives(medicine_name, df_all)
        if target_info is None:
            raise HTTPException(status_code=404, detail=f"Medicine '{medicine_name}' not found.")
            
        safe_alts = alts_df.fillna("")
        return {
            "selected_medicine": target_info,
            "alternatives": safe_alts.to_dict(orient="records")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics")
def get_analytics():
    try:
        stats = predictor.get_cluster_analytics(df_all)
        return {
            "total_catalog_size": int(len(df_all)),
            "total_manufacturers": int(df_all['manufacturer_name'].nunique()),
            "tier_breakdown": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)