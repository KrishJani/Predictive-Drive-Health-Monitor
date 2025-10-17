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

**To get started:**
1.  Download and unzip a data quarter from the [Backblaze Drive Stats](https://www.backblaze.com/b2/hard-drive-test-data.html). (I have used 2013 data for this project)
2.  Enter the path to the unzipped folder below.
""")

# Debug info
st.sidebar.write("**Debug Info:**")
st.sidebar.write(f"Streamlit version: {st.__version__}")
st.sidebar.write(f"Matplotlib backend: {matplotlib.get_backend()}")

# --- Sidebar for User Inputs ---
st.sidebar.header("⚙️ Analysis Settings")

# Input for the data folder path
folder_path = st.sidebar.text_input(
    "Enter the path to your Backblaze data folder",
    placeholder="training_data/"
)

contamination_rate = st.sidebar.slider(
    'Expected Anomaly Rate (Contamination)',
    min_value=0.001,
    max_value=0.1,
    value=0.01,
    step=0.001,
    format="%.3f",
    help="Adjust this based on how rare you believe failures are in the dataset."
)

run_button = st.sidebar.button("Run Analysis", type="primary")

# --- Main App Logic ---
if run_button and folder_path:
    try:
        with st.spinner('Loading and combining data... This may take a few minutes for a full quarter.'):
            drive_data = load_and_preprocess_data(folder_path)
        st.success(f"Successfully loaded {len(drive_data):,} drive records from the folder.")

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
    st.info("Enter a data folder path and click 'Run Analysis' in the sidebar to start.")