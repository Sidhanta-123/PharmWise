import textwrap
import streamlit as st
import requests
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(page_title="PharmWise Optimizer", page_icon="💊", layout="wide")

API_URL = "http://127.0.0.1:8000"

def check_api_health():
    try:
        res = requests.get(f"{API_URL}/", timeout=3)
        return res.status_code == 200
    except requests.exceptions.RequestException:
        return False

@st.cache_data(ttl=300)
def fetch_medicines():
    res = requests.get(f"{API_URL}/medicines", timeout=15)
    res.raise_for_status()
    return res.json()["medicines"]

@st.cache_data(ttl=300)
def search_catalog(query: str):
    res = requests.get(f"{API_URL}/medicines/search", params={"q": query, "limit": 30}, timeout=10)
    res.raise_for_status()
    return res.json()["results"]

@st.cache_data(ttl=300)
def fetch_alternatives(medicine_name: str):
    res = requests.get(
        f"{API_URL}/alternatives",
        params={"medicine_name": medicine_name},
        timeout=10
    )
    res.raise_for_status()
    return res.json()

@st.cache_data(ttl=300)
def fetch_analytics():
    res = requests.get(f"{API_URL}/analytics", timeout=10)
    res.raise_for_status()
    return res.json()

# Sidebar
with st.sidebar:
    st.header("⚙️ System Architecture")
    api_online = check_api_health()
    if api_online:
        st.success("Backend API v2.1: Online (Port 8000)")
    else:
        st.error("Backend API: Offline")
        st.info("Start the backend in a terminal:\n`python api.py`")

    st.markdown("---")
    st.markdown(
        "**Core Engine Modules:**\n"
        "- **Redundancy Check:** Bipartite Salt Graph\n"
        "- **Clinical Safety:** Cross-Salt DDI & Class Duplication\n"
        "- **Trust Tiers:** Log-Transformed K-Means ($K=3$)\n"
        "- **Cost Engine:** Per-Unit (`₹/tab`, `₹/ml`) Exact-Dose Matcher\n"
        "- **API Docs:** [Swagger UI](http://127.0.0.1:8000/docs)"
    )
    st.markdown("---")
    st.caption(
        "⚕️ **Clinical Note:** PharmWise is an academic decision-support system. "
        "Always consult a licensed physician or pharmacist before altering prescriptions."
    )

if not api_online:
    st.title("💊 PharmWise: Prescription Basket Optimizer")
    st.warning("Cannot connect to the FastAPI backend on `http://127.0.0.1:8000`. Please run `python api.py` and refresh.")
    st.stop()

medicine_list = fetch_medicines()

st.title("💊 PharmWise: Prescription Basket Optimizer")
st.markdown(
    "Detect duplicate chemical salts, screen cross-drug clinical interactions, and compute "
    "**unit-normalized** generic savings from **Tier 1** & **Tier 2** manufacturers."
)

tab1, tab2, tab3 = st.tabs([
    "🛒 Prescription Basket Optimizer",
    "🔍 Generic & Salt Explorer",
    "📊 K-Means Trust Tier Analytics"
])


# TAB 1: BASKET OPTIMIZER & CLINICAL SAFETY

with tab1:
    st.subheader("Build Your Prescription Basket")

    default_candidates = [
        "Augmentin 625 Duo Tablet",
        "Aldigesic P 100mg/325mg Tablet",
        "ACTION OR 1000mg Tablet"
    ]
    valid_defaults = [m for m in default_candidates if m in medicine_list]

    selected_medicines = st.multiselect(
        "Search and select prescribed medicines (minimum 2):",
        options=medicine_list,
        default=valid_defaults
    )

    if st.button("Analyze Prescription Basket", type="primary"):
        if len(selected_medicines) < 2:
            st.warning("Please select at least 2 medicines to analyze.")
        else:
            with st.spinner("Running Redundancy, DDI & Unit-Price Optimization..."):
                response = requests.post(
                    f"{API_URL}/analyze",
                    json={"medicines": selected_medicines},
                    timeout=15
                )
                if response.status_code == 200:
                    st.session_state["analysis_result"] = response.json()
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")

    if "analysis_result" in st.session_state:
        data = st.session_state["analysis_result"]
        overlaps = data["overlaps"]
        interactions = data.get("clinical_interactions", [])
        basket_details = data["basket_details"]
        opt_df = pd.DataFrame(data["optimization_table"])
        metrics = data["metrics"]

        orig_cost = metrics["original_cost"]
        new_cost = metrics["optimized_cost"]
        savings = metrics["total_savings"]
        pct_saved = (savings / orig_cost * 100) if orig_cost > 0 else 0.0

        st.markdown("---")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Original Basket Cost", f"₹{orig_cost:.2f}")
        k2.metric("Eqv. Dose Generic Cost", f"₹{new_cost:.2f}", delta=f"-{pct_saved:.1f}%", delta_color="inverse")
        k3.metric("Net Patient Savings", f"₹{savings:.2f}")
        k4.metric("Safety Alerts", f"{len(overlaps)} Overlap | {len(interactions)} DDI")

        col1, col2 = st.columns([1.0, 1.25])

        with col1:
            st.subheader("⚠️ Pharmacological Safety Network")
            if overlaps:
                overlap_str = ", ".join([f"**{salt}** ({count}x)" for salt, count in overlaps.items()])
                st.error(f"**Duplicate Salt Overdose Risk:** {overlap_str}")
            else:
                st.success("**No Exact Salt Duplicates:** Active chemicals are distinct.")

            if interactions:
                for alert in interactions:
                    st.warning(
                        f"**{alert['type']} ({alert['severity']} Risk) — {alert['category']}**\n\n"
                        f"- **Involved Medicines:** {', '.join(alert['medicines_involved'])}\n"
                        f"- **Interacting Salts:** {', '.join(alert['salts_involved'])}\n"
                        f"- *{alert['clinical_note']}*"
                    )

            ddi_salts = {s for alert in interactions for s in alert.get("salts_involved", [])}

            B = nx.Graph()
            med_nodes = []
            label_map = {}

            for row in basket_details:
                med = row["name"]
                med_nodes.append(med)
                B.add_node(med, bipartite=0)
                label_map[med] = textwrap.fill(med, width=20)

                salts = [s.strip() for s in str(row["chemical_signature"]).split("+")]
                for salt in salts:
                    B.add_node(salt, bipartite=1)
                    B.add_edge(med, salt)
                    label_map[salt] = textwrap.fill(salt, width=18)

            fig, ax = plt.subplots(figsize=(8.5, 5.2))
            pos = nx.bipartite_layout(B, med_nodes)

            node_colors = [
                "#1f77b4" if node in med_nodes
                else "#d62728" if node in overlaps
                else "#ff7f0e" if node in ddi_salts
                else "#2ca02c"
                for node in B.nodes()
            ]

            nx.draw(
                B, pos, ax=ax, labels=label_map, with_labels=True,
                node_color=node_colors, node_size=2000,
                font_size=7.5, font_weight="bold", edge_color="#888888", width=1.6
            )
            ax.margins(x=0.35, y=0.18)

            legend_handles = [
                mpatches.Patch(color="#1f77b4", label="Medicine"),
                mpatches.Patch(color="#2ca02c", label="Safe Salt"),
                mpatches.Patch(color="#d62728", label="Duplicate Salt"),
                mpatches.Patch(color="#ff7f0e", label="Cross-Salt DDI")
            ]
            ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=7.5, frameon=False)
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.subheader("💰 Unit-Normalized Generic Substitutes")
            st.caption("Generics are matched on exact active salt strength, delivery form, and lowest per-unit (`₹/tablet` or `₹/ml`) cost.")
            st.dataframe(opt_df, use_container_width=True, hide_index=True)

            csv_bytes = opt_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 Download Optimized Prescription (CSV)",
                data=csv_bytes,
                file_name="pharmwise_optimized_basket.csv",
                mime="text/csv"
            )


# TAB 2: SINGLE-DRUG & SALT EXPLORER
with tab2:
    st.subheader("Search by Active Salt or Compare Brand Generics")

    search_col, select_col = st.columns([1, 1.5])
    with search_col:
        salt_query = st.text_input("Quick Filter by Salt or Brand (Server-Side Search):", placeholder="e.g., Ibuprofen, Azithromycin...")

    filtered_options = medicine_list
    if salt_query and len(salt_query.strip()) >= 2:
        matches = search_catalog(salt_query.strip())
        if matches:
            filtered_options = [m["name"] for m in matches]
        else:
            st.info(f"No medicines matched '{salt_query}'. Showing full catalog.")

    default_single_idx = (
        filtered_options.index("Augmentin 625 Duo Tablet")
        if "Augmentin 625 Duo Tablet" in filtered_options else 0
    )

    with select_col:
        target_drug = st.selectbox(
            "Select a medicine to inspect unit-price disparity:",
            options=filtered_options,
            index=default_single_idx
        )

    only_reliable = st.checkbox("Show only Tier 1 & Tier 2 manufacturers", value=False)

    if target_drug:
        alt_data = fetch_alternatives(target_drug)
        info = alt_data["selected_medicine"]
        alts_df = pd.DataFrame(alt_data["alternatives"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Selected Pack Price", f"₹{info['price']:.2f}", help=f"Pack: {info['pack_label']}")
        c2.metric("Unit Cost", f"₹{info['unit_price']:.2f} / {info['unit_measure'].lower()}")
        c3.metric("Exact Dosage Key", info["dosage_signature"])
        c4.metric("Manufacturer Tier", " ".join(info["trust_tier"].split(" ")[:2]))

        if not alts_df.empty:
            if only_reliable:
                alts_df = alts_df[alts_df["trust_tier"].str.contains("Tier 1|Tier 2", na=False)].copy()

            if not alts_df.empty:
                alts_df["Eqv. Savings (₹)"] = (info["price"] - alts_df["eqv_dose_price"]).round(2)

                plot_df = alts_df.head(8).copy()
                if info["name"] not in plot_df["name"].values:
                    selected_row = pd.DataFrame([{
                        "name": info["name"],
                        "manufacturer_name": info["manufacturer"],
                        "pack_label": info["pack_label"],
                        "price": info["price"],
                        "unit_price": info["unit_price"],
                        "eqv_dose_price": info["price"],
                        "trust_tier": info["trust_tier"],
                        "Eqv. Savings (₹)": 0.0
                    }])
                    plot_df = pd.concat([plot_df, selected_row], ignore_index=True)

                plot_df = plot_df.sort_values(by="unit_price", ascending=True)

                left_chart, right_table = st.columns([1, 1.2])

                with left_chart:
                    fig2, ax2 = plt.subplots(figsize=(7.5, 4.2))
                    bar_colors = [
                        "#ff7f0e" if n == info["name"]
                        else "#1f77b4" if "Tier 1" in str(t)
                        else "#2ca02c" if "Tier 2" in str(t)
                        else "#7f7f7f"
                        for n, t in zip(plot_df["name"], plot_df["trust_tier"])
                    ]
                    bars = ax2.barh(plot_df["name"], plot_df["unit_price"], color=bar_colors)
                    ax2.set_xlabel(f"Per-{info['unit_measure'].title()} Price (₹)")
                    ax2.set_title("Per-Unit Cost Comparison (Orange = Selected Drug)", fontsize=10)
                    ax2.invert_yaxis()

                    max_val = plot_df["unit_price"].max() if len(plot_df) > 0 else 1.0
                    for bar in bars:
                        width = bar.get_width()
                        ax2.text(width + (max_val * 0.015), bar.get_y() + bar.get_height()/2,
                                 f"₹{width:.2f}", va="center", fontsize=8)
                    ax2.margins(x=0.2)
                    st.pyplot(fig2)
                    plt.close(fig2)

                with right_table:
                    display_df = alts_df.rename(columns={
                        "name": "Medicine Name",
                        "manufacturer_name": "Manufacturer",
                        "pack_label": "Pack Size",
                        "price": "Pack Price (₹)",
                        "unit_price": "Unit Cost (₹)",
                        "eqv_dose_price": "Eqv. Dose Cost (₹)",
                        "trust_tier": "Trust Tier"
                    })
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("No Tier 1 or Tier 2 alternatives found for this exact dosage.")
        else:
            st.info("No exact-dosage alternatives found in the catalog.")

# TAB 3: K-MEANS TRUST TIER ANALYTICS

with tab3:
    st.subheader("Unsupervised Manufacturer Clustering Overview (Log-K-Means, K=3)")

    analytics = fetch_analytics()
    tier_df = pd.DataFrame(analytics["tier_breakdown"])

    a1, a2, a3 = st.columns(3)
    a1.metric("Total Catalog Medicines", f"{analytics['total_catalog_size']:,}")
    a2.metric("Unique Pharmaceutical Manufacturers", f"{analytics['total_manufacturers']:,}")
    a3.metric("Validated Trust Clusters", "3 Tiers (Silhouette: 0.43)")

    ch1, ch2 = st.columns(2)

    with ch1:
        fig3, ax3 = plt.subplots(figsize=(6.5, 3.8))
        short_labels = [t.split("(")[0].strip() for t in tier_df["trust_tier"]]
        ax3.bar(short_labels, tier_df["total_medicines"], color=["#1f77b4", "#2ca02c", "#7f7f7f"])
        ax3.set_ylabel("Number of Medicines")
        ax3.set_title("Catalog Volume Across Manufacturer Tiers", fontsize=10)
        st.pyplot(fig3)
        plt.close(fig3)

    with ch2:
        fig4, ax4 = plt.subplots(figsize=(6.5, 3.8))
        x_idx = range(len(short_labels))
        width = 0.35
        ax4.bar([i - width/2 for i in x_idx], tier_df["avg_price"], width=width, label="Mean Price (₹)", color="#1f77b4")
        ax4.bar([i + width/2 for i in x_idx], tier_df["median_price"], width=width, label="Median Price (₹)", color="#ff7f0e")
        ax4.set_xticks(list(x_idx))
        ax4.set_xticklabels(short_labels)
        ax4.set_ylabel("Price (₹)")
        ax4.set_title("Mean vs. Median Price Disparity per Tier", fontsize=10)
        ax4.legend(fontsize=8)
        st.pyplot(fig4)
        plt.close(fig4)

    st.markdown("**Cluster Aggregation Summary**")
    formatted_tiers = tier_df.rename(columns={
        "trust_tier": "Manufacturer Trust Tier",
        "total_medicines": "Total Products",
        "unique_manufacturers": "Manufacturer Count",
        "avg_price": "Mean Price (₹)",
        "median_price": "Median Price (₹)"
    })
    st.dataframe(formatted_tiers, use_container_width=True, hide_index=True)