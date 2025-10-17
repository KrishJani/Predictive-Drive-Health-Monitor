"""
app.py

A Streamlit web application for the Predictive Drive Health Monitor.
This app provides an interactive interface to upload data, run anomaly
detection, and visualize the results.
"""

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import seaborn as sns
from drive_analyzer import load_and_preprocess_data, detect_anomalies, get_evaluation_report

# --- Page Configuration ---
st.set_page_config(
    page_title="Predictive Drive Health Monitor",
    page_icon="🤖",
    layout="wide"
)

# --- App Title and Description ---
st.title("🤖 Predictive Drive Health Monitor")
st.write("""
This tool uses an **Isolation Forest** machine learning model to proactively detect
potentially failing SSDs and HDDs from operational data.

**Ready to analyze!** The app is configured to use your local 2013 Backblaze data.
Simply adjust the settings in the sidebar and click "Run Analysis" to start.
""")

# Check if data folder exists
import os
data_exists = os.path.exists("training_data/") and len(os.listdir("training_data/")) > 0 if os.path.exists("training_data/") else False

# Debug info
st.sidebar.write("**Debug Info:**")
st.sidebar.write(f"Streamlit version: {st.__version__}")
st.sidebar.write(f"Matplotlib backend: {matplotlib.get_backend()}")
st.sidebar.write(f"Data folder exists: {'✅ Yes' if data_exists else '❌ No'}")

# --- Sidebar for User Inputs ---
st.sidebar.header("⚙️ Analysis Settings")

# Input for the data folder path
folder_path = st.sidebar.text_input(
    "Enter the path to your Backblaze data folder",
    value="training_data/",  # Default to local data
    placeholder="training_data/"
)

contamination_rate = st.sidebar.slider(
    'Expected Anomaly Rate (Contamination)',
    min_value=0.0001,
    max_value=0.05,
    value=0.01,
    step=0.0001,
    format="%.4f",
    help="Controls recall vs precision trade-off. Lower = higher precision, lower recall. Higher = higher recall, lower precision."
)

# Show expected recall based on contamination
if contamination_rate <= 0.001:
    expected_recall = "~10% (Very selective)"
elif contamination_rate <= 0.005:
    expected_recall = "~30% (Selective)"
elif contamination_rate <= 0.01:
    expected_recall = "~50% (Balanced)"
elif contamination_rate <= 0.02:
    expected_recall = "~70% (Inclusive)"
else:
    expected_recall = "~90% (Very inclusive)"

st.sidebar.write(f"**Expected Recall:** {expected_recall}")
st.sidebar.write("**💡 Tip:** For maximum recall, use 0.02-0.05. For balanced performance, use 0.01.")

run_button = st.sidebar.button("Run Analysis", type="primary")

# --- Main App Logic ---
if run_button and folder_path:
    try:
        with st.spinner('Loading and combining data... This may take a few minutes for a full quarter.'):
            drive_data = load_and_preprocess_data(folder_path)
        st.success(f"Successfully loaded {len(drive_data):,} drive records from the folder.")
        
        # Show actual failure rate
        actual_failures = int(drive_data['failure'].sum())
        actual_failure_rate = drive_data['failure'].mean()
        st.info(f"📊 **Data Statistics:** {actual_failures:,} actual failures out of {len(drive_data):,} drives ({actual_failure_rate:.4f}% failure rate)")

        with st.spinner('Running anomaly detection model...'):
            df_with_anomalies = detect_anomalies(drive_data, contamination=contamination_rate)

        st.header("📊 Analysis Results")

        # Display key metrics
        anomalies_found = len(df_with_anomalies[df_with_anomalies['anomaly'] == -1])
        actual_failures = int(df_with_anomalies['failure'].sum())

        col1, col2, col3 = st.columns(3)
        col1.metric("Anomalies Detected by Model", f"{anomalies_found:,}")
        col2.metric("Actual Failures in Data", f"{actual_failures:,}")
        col3.metric("Anomaly Rate Setting", f"{contamination_rate:.3f}")

        # Display top anomalies in an expandable section
        with st.expander("View Top 10 Most Anomalous Drives", expanded=False):
            st.write("These are the drives the model found most likely to be anomalous, sorted by score.")
            top_anomalies = df_with_anomalies.sort_values('anomaly_score').head(10)
            st.dataframe(top_anomalies[['serial_number', 'failure', 'anomaly_score', 'reallocated_sectors', 'uncorrectable_errors', 'pending_sectors']])

        # Display evaluation report
        st.subheader("📝 Model Performance Evaluation")
        report = get_evaluation_report(df_with_anomalies)
        st.text(report)
        st.info("""
        **How to read this report:**
        - **Precision (Failed):** Of all drives we flagged as anomalies, what percentage *actually* failed?
        - **Recall (Failed):** Of all drives that *actually* failed, what percentage did our model successfully catch?
        """)

        # Display visualizations
        st.subheader("📈 Data Visualizations")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.histplot(data=df_with_anomalies, x='anomaly_score', hue='anomaly', kde=True, palette={1: '#4B8BBE', -1: '#D9534F'}, ax=ax1)
        ax1.set_title('Distribution of Anomaly Scores (Red = Anomaly)', fontsize=16)
        st.pyplot(fig1)

        anomalies_df = df_with_anomalies[df_with_anomalies['anomaly'] == -1]
        if not anomalies_df.empty:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            sns.scatterplot(data=anomalies_df, x='temperature', y='reallocated_sectors', color='red', alpha=0.5, ax=ax2)
            ax2.set_title('Anomalous Drives: Temperature vs. Reallocated Sectors', fontsize=16)
            ax2.set_xlabel('Temperature (Celsius)')
            ax2.set_ylabel('Reallocated Sectors Count')
            st.pyplot(fig2)

    except FileNotFoundError as e:
        st.error(f"Error: Could not find the folder. Please check the path: {e}")
    except ValueError as e:
        st.error(f"Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

elif run_button and not folder_path:
    st.warning("Please enter a folder path to begin the analysis.")

else:
    st.info("✅ Ready to analyze! The data path is set to 'training_data/'. Click 'Run Analysis' in the sidebar to start.")