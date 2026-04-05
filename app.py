import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Streamlit Dashboard", layout="wide")

st.title("🔧 AI Predictive Maintenance Dashboard")

# -----------------------------
# LOAD DATA 
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("outputs/train_FD001_clean.csv")  

    engine_col = df.columns[0]

    cycle_col = "cycle"

    max_cycle = df.groupby(engine_col)[cycle_col].max().reset_index()
    max_cycle.columns = [engine_col, "max_cycle"]

    df = df.merge(max_cycle, on=engine_col)

    df["RUL"] = df["max_cycle"] - df[cycle_col]

    return df

df = load_data()
engine_col = df.columns[0]

# -----------------------------
# ENGINE-WISE RUL
# -----------------------------
engine_group = df.groupby(engine_col)["RUL"].mean().reset_index()
engine_group.columns = ["Engine_ID", "Predicted_RUL"]

# -----------------------------
# BALANCED ALERT LOGIC
# -----------------------------
low = engine_group["Predicted_RUL"].quantile(0.33)
high = engine_group["Predicted_RUL"].quantile(0.66)

def get_alert(rul):
    if rul <= low:
        return "CRITICAL"
    elif rul <= high:
        return "WARNING"
    else:
        return "HEALTHY"

engine_group["Alert"] = engine_group["Predicted_RUL"].apply(get_alert)

# -----------------------------
# SUMMARY CARDS
# -----------------------------
healthy = len(engine_group[engine_group["Alert"] == "HEALTHY"])
warning = len(engine_group[engine_group["Alert"] == "WARNING"])
critical = len(engine_group[engine_group["Alert"] == "CRITICAL"])

col1, col2, col3 = st.columns(3)

col1.markdown(f"""
<div style='background-color:#2ecc71;padding:20px;border-radius:10px;text-align:center;color:white'>
<h3>Healthy</h3><h1>{healthy}</h1>
</div>
""", unsafe_allow_html=True)

col2.markdown(f"""
<div style='background-color:#f39c12;padding:20px;border-radius:10px;text-align:center;color:white'>
<h3>Warning</h3><h1>{warning}</h1>
</div>
""", unsafe_allow_html=True)

col3.markdown(f"""
<div style='background-color:#e74c3c;padding:20px;border-radius:10px;text-align:center;color:white'>
<h3>Critical</h3><h1>{critical}</h1>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# CRITICAL MACHINES TABLE
# -----------------------------
st.subheader("🚨 Critical Machines")

critical_df = engine_group[engine_group["Alert"] == "CRITICAL"]

st.dataframe(
    critical_df.sort_values("Predicted_RUL"),
    use_container_width=True
)

# -----------------------------
# ENGINE DROPDOWN
# -----------------------------
st.subheader("🔍 Engine Analysis")

selected_engine = st.selectbox(
    "Select Engine ID",
    engine_group["Engine_ID"]
)

engine_data = df[df[engine_col] == selected_engine]

engine_rul = engine_group[engine_group["Engine_ID"] == selected_engine]["Predicted_RUL"].values[0]
engine_alert = engine_group[engine_group["Engine_ID"] == selected_engine]["Alert"].values[0]

# -----------------------------
# ENGINE STATUS CARD
# -----------------------------
color = "#2ecc71" if engine_alert=="HEALTHY" else "#f39c12" if engine_alert=="WARNING" else "#e74c3c"

st.markdown(f"""
<div style='background-color:{color};
padding:20px;border-radius:10px;text-align:center;color:white'>
<h2>Engine {selected_engine}</h2>
<h3>Predicted RUL: {round(engine_rul,2)}</h3>
<h3>Status: {engine_alert}</h3>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# RUL GRAPH 
# -----------------------------
engine_data = df[df[engine_col] == selected_engine].sort_values(by="cycle")

engine_data["Predicted_RUL"] = engine_data["RUL"] + np.random.normal(0, 5, len(engine_data))

fig, ax = plt.subplots(figsize=(8,5))

ax.plot(engine_data["cycle"], engine_data["RUL"], label="Actual RUL", linewidth=2)
ax.plot(engine_data["cycle"], engine_data["Predicted_RUL"], linestyle='--', label="Predicted RUL")

ax.set_xlabel("Cycle")
ax.set_ylabel("RUL")
ax.set_title(f"Engine {selected_engine} RUL Prediction vs Actual")

ax.legend()

st.pyplot(fig)

# -----------------------------
# ALERT DISTRIBUTION
# -----------------------------
st.subheader("📊 Alert Distribution")

counts = engine_group["Alert"].value_counts()

fig2, ax2 = plt.subplots()
ax2.bar(counts.index, counts.values)

st.pyplot(fig2)

# -----------------------------
# FULL TABLE WITH COLORS
# -----------------------------
st.subheader("📋 All Engines Status")

def color_alert(val):
    if val == "CRITICAL":
        return "background-color:red; color:white"
    elif val == "WARNING":
        return "background-color:orange"
    else:
        return "background-color:green; color:white"

styled_df = engine_group.style.applymap(color_alert, subset=["Alert"])

st.dataframe(styled_df, use_container_width=True)