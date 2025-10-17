#!/usr/bin/env python3
"""
Command-line version of the Predictive Drive Health Monitor.
This script runs the analysis without requiring Streamlit.
"""

import sys
import os
from drive_analyzer import load_and_preprocess_data, detect_anomalies, get_evaluation_report

def main():
    print("🤖 Predictive Drive Health Monitor - Command Line Version")
    print("=" * 60)
    
    # Default data path
    data_path = "training_data"
    
    # Check if data path exists
    if not os.path.exists(data_path):
        print(f"❌ Error: Data folder '{data_path}' not found!")
        print("Please make sure the training_data folder exists with CSV files.")
        sys.exit(1)
    
    try:
        print(f"📁 Loading data from: {data_path}")
        print("⏳ This may take a few minutes for a full dataset...")
        
        # Load and preprocess data
        drive_data = load_and_preprocess_data(data_path)
        print(f"✅ Successfully loaded {len(drive_data):,} drive records")
        
        # Run anomaly detection
        print("🔍 Running anomaly detection...")
        contamination_rate = 0.01  # 1% expected anomaly rate
        df_with_anomalies = detect_anomalies(drive_data, contamination=contamination_rate)
        
        # Display results
        print("\n📊 Analysis Results")
        print("-" * 30)
        
        anomalies_found = len(df_with_anomalies[df_with_anomalies['anomaly'] == -1])
        actual_failures = int(df_with_anomalies['failure'].sum())
        
        print(f"Anomalies Detected: {anomalies_found:,}")
        print(f"Actual Failures: {actual_failures:,}")
        print(f"Anomaly Rate Setting: {contamination_rate:.3f}")
        
        # Show top anomalies
        print(f"\n🔍 Top 10 Most Anomalous Drives:")
        print("-" * 50)
        top_anomalies = df_with_anomalies.sort_values('anomaly_score').head(10)
        for idx, row in top_anomalies.iterrows():
            status = "FAILED" if row['failure'] == 1 else "Normal"
            print(f"Serial: {row['serial_number']} | Status: {status} | Score: {row['anomaly_score']:.4f}")
        
        # Evaluation report
        print(f"\n📝 Model Performance Evaluation:")
        print("-" * 40)
        report = get_evaluation_report(df_with_anomalies)
        print(report)
        
        print(f"\n✅ Analysis complete!")
        print(f"💡 To run the web interface, install Streamlit and run: streamlit run app.py")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
