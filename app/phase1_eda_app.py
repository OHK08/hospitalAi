import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------
# Page Config
# --------------------------
st.set_page_config(page_title="Healthcare Wait Time - Phase 1 EDA", layout="wide")
sns.set_style("whitegrid")

st.title("🏥 Healthcare Wait Time Optimization")
st.markdown("### **Phase 1: Define & Measure** *(Six Sigma DMAIC)*")

# --------------------------
# File Upload Section
# --------------------------
uploaded_file = st.file_uploader("📤 Upload your ER Wait Time CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Using default dataset from `data/raw/ErWaitTime.csv`")
    df = pd.read_csv("../data/raw/ErWaitTime.csv")  # adjust if needed

df.columns = df.columns.str.strip()

# --------------------------
# Quick Stats
# --------------------------
col1, col2, col3 = st.columns(3)
col1.metric("📄 Rows", f"{df.shape[0]:,}")
col2.metric("📊 Columns", f"{df.shape[1]}")
col3.metric("🕒 Missing Values", df.isnull().sum().sum())

# --------------------------
# Dataset Preview
# --------------------------
with st.expander("🔍 Preview Dataset"):
    st.dataframe(df.head(10), use_container_width=True)

# --------------------------
# Data Types & Missing Values
# --------------------------
with st.expander("📋 Data Info"):
    st.write("**Data Types:**")
    st.write(df.dtypes)
    st.write("**Missing Values per Column:**")
    st.write(df.isnull().sum())

# --------------------------
# Summary Statistics
# --------------------------
with st.expander("📊 Summary Statistics"):
    st.dataframe(df.describe().T, use_container_width=True)

# --------------------------
# Numerical Distributions
# --------------------------
st.subheader("📈 Distributions")
num_cols = df.select_dtypes(include=np.number).columns.tolist()

if num_cols:
    col_a, col_b = st.columns(2)
    for idx, col in enumerate(num_cols[:6]):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[col], kde=True, ax=ax, color="skyblue")
        ax.set_title(f"Histogram of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        if idx % 2 == 0:
            col_a.pyplot(fig)
        else:
            col_b.pyplot(fig)

# --------------------------
# Correlation Heatmap
# --------------------------
# if len(num_cols) >= 2:
#     st.subheader("🔗 Correlation Matrix")
#     fig, ax = plt.subplots(figsize=(8, 5))
#     sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
#     st.pyplot(fig)

if len(num_cols) >= 2:
    st.subheader("🔗 Correlation Matrix")
    corr_matrix = df[num_cols].corr()
    # Keep a fixed compact size that fits well on most screens
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax,
        cbar=True,
        square=True,
        linewidths=0.5
    )
    # Center the heatmap in the Streamlit layout
    center_col = st.columns([1, 6, 1])[1]  # middle column is wider
    with center_col:
        st.pyplot(fig)

# --------------------------
# Categorical Distributions
# --------------------------
# cat_cols = df.select_dtypes(include="object").columns.tolist()
# if cat_cols:
#     st.subheader("🔤 Categorical Distributions")
#     for col in cat_cols[:4]:
#         fig, ax = plt.subplots(figsize=(6, 4))
#         df[col].value_counts().head(10).plot(kind="bar", ax=ax, color="lightgreen")
#         ax.set_title(f"Top 10 {col} Values")
#         ax.set_ylabel("Count")
#         ax.set_xlabel(col)
#         st.pyplot(fig)

# --------------------------
# Project Charter & SIPOC
# --------------------------
with st.expander("📜 Project Charter & SIPOC"):
    st.markdown("""
    **Project Title:** Healthcare Wait Time Optimization using Six Sigma (DMAIC) & AIML
    
    **Problem Statement:**  
    Emergency Department (ED) wait times are above acceptable thresholds, causing reduced patient satisfaction and potential adverse outcomes.
    
    **Goal:**  
    Reduce average ED wait time by X% within Y months.
    
    **Scope:**  
    - **In-scope:** ED triage to initial provider contact  
    - **Out-of-scope:** Inpatient bed assignment processes
    
    **CTQs:**  
    - Average wait time  
    - % patients seen within target time  
    - Throughput per hour
    
    **Stakeholders:** ED staff, hospital administrators, patients
    
    ---
    
    **SIPOC Diagram:**  
    - **Suppliers:** EMS, patients, registration staff  
    - **Inputs:** Arrival time, triage info, staffing, beds  
    - **Process:** Registration → Triage → Assessment → Treatment → Disposition  
    - **Outputs:** Patient seen, discharged/admitted  
    - **Customers:** Patients, hospital mgmt, insurers
    """)


# --------------------------
# 1. Baseline Performance Dashboard
# --------------------------
st.subheader("📌 Baseline Performance Metrics")
wait_col = "Total Wait Time (min)"  # Change if different
if wait_col in df.columns:
    avg_wait = df[wait_col].mean()
    median_wait = df[wait_col].median()
    pct_under_target = (df[wait_col] <= 30).mean() * 100  # Example: target 30 mins

    c1, c2, c3 = st.columns(3)
    c1.metric("Average Wait Time", f"{avg_wait:.1f} min")
    c2.metric("Median Wait Time", f"{median_wait:.1f} min")
    c3.metric("% Under 30 min", f"{pct_under_target:.1f} %")
else:
    st.warning(f"'{wait_col}' column not found. Please update column name for metrics.")

# --------------------------
# 2. Segment Analysis (Compact)
# --------------------------
st.subheader("📊 Average Wait Time by Segments")
segment_cols = ["Urgency Level", "Day of Week", "Time of Day", "Region", "Hospital Name"]

for seg in segment_cols:
    if seg in df.columns:
        st.write(f"**By {seg}**")
        seg_df = df.groupby(seg)[wait_col].mean().reset_index().sort_values(wait_col, ascending=False)
        st.dataframe(seg_df, use_container_width=True)

        # Center the chart and make it small
        col_center = st.columns([1, 4, 1])[1]  # Middle column is wider
        with col_center:
            fig, ax = plt.subplots(figsize=(5, 3))  # Smaller figure size
            sns.barplot(data=seg_df, x=wait_col, y=seg, palette="viridis", ax=ax)
            ax.set_title(f"Average Wait Time by {seg}", fontsize=10)
            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            plt.tight_layout()
            st.pyplot(fig)


# --------------------------
# 3. Outlier Detection
# --------------------------
st.subheader("🚨 Longest Wait Time Cases")
if wait_col in df.columns:
    outliers_df = df.nlargest(10, wait_col)
    st.dataframe(outliers_df, use_container_width=True)

# --------------------------
# 4. Auto Hypothesis Generator
# --------------------------
st.subheader("💡 Potential Root Cause Hypotheses")
hypotheses = []

# Example correlation-based hypotheses
if "Specialist Availability" in df.columns and wait_col in df.columns:
    corr_val = df["Specialist Availability"].corr(df[wait_col])
    if corr_val < -0.3:
        hypotheses.append("Higher specialist availability is linked to shorter wait times.")
    elif corr_val > 0.3:
        hypotheses.append("Higher specialist availability is unexpectedly linked to longer wait times — possible scheduling mismatch.")

if "Nurse-to-Patient Ratio" in df.columns and wait_col in df.columns:
    corr_val = df["Nurse-to-Patient Ratio"].corr(df[wait_col])
    if corr_val < -0.3:
        hypotheses.append("Better nurse-to-patient ratios reduce wait times.")
    elif corr_val > 0.3:
        hypotheses.append("Higher nurse-to-patient ratios may still lead to longer waits — possible inefficiency.")

# Day of week effect
if "Day of Week" in df.columns and wait_col in df.columns:
    day_avg = df.groupby("Day of Week")[wait_col].mean()
    if day_avg.max() - day_avg.min() > 10:
        hypotheses.append("Certain days (e.g., weekends) have significantly higher wait times.")

# Urgency effect
if "Urgency Level" in df.columns and wait_col in df.columns:
    urg_avg = df.groupby("Urgency Level")[wait_col].mean()
    if urg_avg.max() - urg_avg.min() > 15:
        hypotheses.append("High urgency cases may be delaying lower urgency patients, causing longer waits.")

# Show hypotheses
if hypotheses:
    for h in hypotheses:
        st.markdown(f"- {h}")
else:
    st.info("No strong patterns detected for hypotheses.")
