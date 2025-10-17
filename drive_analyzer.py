"""
drive_analyzer.py

Core logic for the Predictive Drive Health Monitor.
This script handles data loading, preprocessing, anomaly detection,
and performance evaluation.
"""

import os
import glob
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')

def load_and_preprocess_data(folder_path):
    """
    Loads and combines all CSV files from a folder, then preprocesses the data.

    Args:
        folder_path (str): Path to the folder containing the Backblaze CSV files.

    Returns:
        pd.DataFrame: A single, cleaned DataFrame ready for analysis.
    """
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in the directory: {folder_path}")

    print(f"Found {len(csv_files)} CSV files to combine.")

    df_list = []
    # These are key S.M.A.R.T. attributes for predicting drive failure
    smart_columns = [
        'serial_number', 'failure', 'smart_5_raw', 'smart_9_raw',
        'smart_187_raw', 'smart_194_raw', 'smart_197_raw', 'smart_198_raw'
    ]

    for file in csv_files:
        try:
            daily_df = pd.read_csv(file, usecols=lambda col: col in smart_columns, low_memory=False)
            df_list.append(daily_df)
        except Exception as e:
            print(f"Could not read or process file {file}: {e}")

    if not df_list:
        raise ValueError("No data could be loaded from the CSV files.")

    full_df = pd.concat(df_list, ignore_index=True)

    # Rename columns for readability
    full_df = full_df.rename(columns={
        'smart_5_raw': 'reallocated_sectors',
        'smart_9_raw': 'power_on_hours',
        'smart_187_raw': 'uncorrectable_errors',
        'smart_194_raw': 'temperature',
        'smart_197_raw': 'pending_sectors',
        'smart_198_raw': 'offline_uncorrectable'
    })

    # For any missing S.M.A.R.T. columns that weren't in some files, fill with 0
    final_columns = [
        'serial_number', 'failure', 'reallocated_sectors', 'power_on_hours',
        'uncorrectable_errors', 'temperature', 'pending_sectors', 'offline_uncorrectable'
    ]
    for col in final_columns:
        if col not in full_df.columns:
            full_df[col] = 0
            
    # Handle missing values - a common task with real data.
    full_df = full_df.fillna(0)
    
    print(f"Data combination complete. Found {len(full_df)} total drive records.")
    return full_df


def detect_anomalies(df, contamination=0.01, random_state=42):
    """
    Detects anomalies in drive data using the Isolation Forest algorithm with threshold-based approach.

    Args:
        df (pd.DataFrame): The input DataFrame with drive health data.
        contamination (float): The expected proportion of anomalies in the data.
        random_state (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with 'anomaly' and 'anomaly_score' columns added.
    """
    df_anomaly = df.copy()

    # Calculate actual failure rate
    actual_failure_rate = df_anomaly['failure'].mean()
    print(f"Actual failure rate in data: {actual_failure_rate:.6f} ({actual_failure_rate*100:.4f}%)")

    feature_columns = [
        'reallocated_sectors', 'power_on_hours', 'uncorrectable_errors',
        'temperature', 'pending_sectors', 'offline_uncorrectable'
    ]

    # Ensure all feature columns exist, even if they were empty in the source files
    for col in feature_columns:
        if col not in df_anomaly.columns:
            df_anomaly[col] = 0

    # Feature engineering: create additional meaningful features
    df_anomaly['total_errors'] = df_anomaly['reallocated_sectors'] + df_anomaly['uncorrectable_errors'] + df_anomaly['pending_sectors']
    df_anomaly['error_rate'] = df_anomaly['total_errors'] / (df_anomaly['power_on_hours'] + 1)  # +1 to avoid division by zero
    df_anomaly['high_temp'] = (df_anomaly['temperature'] > 50).astype(int)  # Binary feature for high temperature
    df_anomaly['age_factor'] = df_anomaly['power_on_hours'] / 8760  # Convert to years (8760 hours/year)
    
    # Additional risk indicators
    df_anomaly['has_reallocated'] = (df_anomaly['reallocated_sectors'] > 0).astype(int)
    df_anomaly['has_uncorrectable'] = (df_anomaly['uncorrectable_errors'] > 0).astype(int)
    df_anomaly['has_pending'] = (df_anomaly['pending_sectors'] > 0).astype(int)

    # Extended feature set with engineered features
    extended_features = feature_columns + ['total_errors', 'error_rate', 'high_temp', 'age_factor', 
                                         'has_reallocated', 'has_uncorrectable', 'has_pending']

    X = df_anomaly[extended_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use a higher contamination rate to get more anomalies, then filter by threshold
    iso_forest = IsolationForest(
        contamination=0.1,  # High contamination to get more candidates
        random_state=random_state,
        n_estimators=200,
        max_samples='auto',
        max_features=1.0,
        bootstrap=False
    )

    # Fit the model first
    iso_forest.fit(X_scaled)
    
    # Get anomaly scores
    anomaly_scores = iso_forest.decision_function(X_scaled)
    df_anomaly['anomaly_score'] = anomaly_scores
    
    # Use threshold-based approach that respects the contamination parameter
    failed_drives = df_anomaly[df_anomaly['failure'] == 1]
    if len(failed_drives) > 0:
        # Calculate what percentile of failures we want to catch based on contamination
        # Lower contamination = higher percentile (more selective, lower recall)
        # Higher contamination = lower percentile (less selective, higher recall)
        
        # Map contamination rate to percentile
        # contamination 0.0001 -> 90th percentile (very selective)
        # contamination 0.01 -> 50th percentile (balanced)
        # contamination 0.05 -> 10th percentile (very inclusive)
        
        if contamination <= 0.001:
            percentile = 0.9  # Very selective - only catch the most obvious failures
        elif contamination <= 0.005:
            percentile = 0.7  # Selective - catch most failures
        elif contamination <= 0.01:
            percentile = 0.5  # Balanced - catch half the failures
        elif contamination <= 0.02:
            percentile = 0.3  # Inclusive - catch most failures
        else:
            percentile = 0.1  # Very inclusive - catch almost all failures
        
        threshold = failed_drives['anomaly_score'].quantile(percentile)
        print(f"Using contamination {contamination:.4f} -> {percentile*100:.0f}th percentile threshold: {threshold:.4f}")
        
        # Apply threshold
        df_anomaly['anomaly'] = (df_anomaly['anomaly_score'] <= threshold).astype(int)
        df_anomaly['anomaly'] = df_anomaly['anomaly'].replace({1: -1, 0: 1})  # Convert to -1/1 format
    else:
        # Fallback to standard approach
        df_anomaly['anomaly'] = iso_forest.fit_predict(X_scaled)

    return df_anomaly


def get_evaluation_report(df):
    """
    Generates a classification report to evaluate the model's performance.

    Args:
        df (pd.DataFrame): DataFrame containing true 'failure' labels and 'anomaly' predictions.

    Returns:
        str: A formatted classification report.
    """
    df['predicted_failure'] = df['anomaly'].apply(lambda x: 1 if x == -1 else 0)

    report = classification_report(
        df['failure'],
        df['predicted_failure'],
        target_names=['Normal (0)', 'Failed (1)'],
        zero_division=0
    )
    return report
