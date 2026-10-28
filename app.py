import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from src.pipeline.predict_pipeline import PredictPipeline

# 1. Page Configuration
st.set_page_config(page_title="PharmWise Optimizer", page_icon="💊", layout="wide")
st.title("💊 PharmWise: Prescription Basket Optimizer")
st.markdown(
    "Detect hidden chemical redundancies and find safe, cost-effective generic "
    "alternatives from **Tier 1** & **Tier 2** manufacturers."
)

# 2. Initialize Pipeline & Cache Dataset
@st.cache_resource
def get_pipeline():
    return PredictPipeline()

@st.cache_data
def load_medicine_data():
    pipeline = get_pipeline()
    return pipeline.load_tiered_data()

try:
    predictor = get_pipeline()
    df = load_medicine_data()
except Exception as e:
    st.error(f"Failed to load application artifacts: {e}")
    st.stop()

# 3. Frontend UI: Search & Select
st.subheader("🛒 Build Your Prescription Basket")
medicine_list = df['name'].dropna().unique().tolist()

selected_medicines = st.multiselect(
    "Search and select medicines (type to filter):",
    options=medicine_list,
    default=[
        "Augmentin 625 Duo Tablet",
        "Aldigesic P 100mg/325mg Tablet",
        "ACTION OR 1000mg Tablet"
    ]
)

if st.button("Analyze Basket", type="primary"):
    if len(selected_medicines) < 2:
        st.warning("Please select at least 2 medicines to analyze for redundancies.")
    else:
        with st.spinner("Running PredictPipeline..."):
            basket_df, overlaps = predictor.detect_redundancy(selected_medicines, df)
            opt_df, orig_cost, new_cost, savings = predictor.optimize_basket_cost(basket_df, df)

            col1, col2 = st.columns([1, 1])

            # Left Column: Redundancy Alert & Bipartite Network Graph
            with col1:
                st.subheader("⚠️ Chemical Redundancy Check")
                if overlaps:
                    overlap_str = ", ".join([f"{salt} ({count}x)" for salt, count in overlaps.items()])
                    st.error(f"**Overdose Risk Detected!** Overlapping active salts: {overlap_str}")
                else:
                    st.success("No overlapping active chemicals detected. Basket is safe.")

                # Build Bipartite Network
                B = nx.Graph()
                medicines = basket_df['name'].tolist()

                for _, row in basket_df.iterrows():
                    med = row['name']
                    B.add_node(med, bipartite=0)
                    salts = [s.strip() for s in str(row['chemical_signature']).split('+')]
                    for salt in salts:
                        B.add_node(salt, bipartite=1)
                        B.add_edge(med, salt)

                fig, ax = plt.subplots(figsize=(9, 5.5))
                pos = nx.bipartite_layout(B, medicines)

                node_colors = [
                    '#1f77b4' if node in medicines
                    else '#d62728' if node in overlaps
                    else '#2ca02c'
                    for node in B.nodes()
                ]

                nx.draw(
                    B, pos, ax=ax, with_labels=True,
                    node_color=node_colors, node_size=2200,
                    font_size=8, font_weight="bold", edge_color="gray", width=1.5
                )
                ax.set_title("Basket Redundancy Network (Red Node = Overdose Risk)", fontsize=11)
                st.pyplot(fig)

            # Right Column: Cost Optimizer Metrics & Table
            with col2:
                st.subheader("💰 Generic Cost Optimization")
                m1, m2, m3 = st.columns(3)
                m1.metric("Original Basket Cost", f"₹{orig_cost:.2f}")
                m2.metric("Optimized Cost", f"₹{new_cost:.2f}")
                m3.metric("Total Patient Savings", f"₹{savings:.2f}")

                st.dataframe(opt_df, use_container_width=True, hide_index=True)