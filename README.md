# Predictive Drive Health Monitor

A machine learning application that uses the Isolation Forest algorithm to detect potentially failing drives from S.M.A.R.T. operational data.

## Overview

This project analyzes drive health data from Backblaze's publicly available drive statistics to predict drive failures before they occur. The application provides both command-line and web interfaces for data analysis and visualization.

## Features

- **Interactive Web Interface**: Streamlit-based dashboard for data analysis
- **Command-Line Interface**: Direct analysis without web dependencies
- **Anomaly Detection**: Isolation Forest algorithm with configurable parameters
- **Performance Metrics**: Detailed evaluation reports with precision and recall scores
- **Real-time Visualization**: Charts showing drive health patterns and anomaly distributions

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/KrishJani/Predictive-Drive-Health-Monitor.git
   cd Predictive-Drive-Health-Monitor
   ```

2. Create and activate virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command Line Version (Recommended)
```bash
python run_analysis.py
```

### Web Interface
```bash
pip install -r requirements-streamlit.txt
streamlit run app.py
```
Then open your browser to `http://localhost:8501`

## Data Requirements

The application expects CSV files from [Backblaze Drive Stats](https://www.backblaze.com/b2/hard-drive-test-data.html) in the `training_data/` folder. The current setup includes 2013 data with the following S.M.A.R.T. attributes:

- **smart_5_raw**: Reallocated sectors count
- **smart_9_raw**: Power-on hours  
- **smart_187_raw**: Uncorrectable errors
- **smart_194_raw**: Temperature
- **smart_197_raw**: Pending sectors
- **smart_198_raw**: Offline uncorrectable errors

## Configuration

### Contamination Rate
The contamination rate controls the recall vs precision trade-off:

- **0.001 (0.1%)**: ~90% recall, very inclusive
- **0.01 (1%)**: ~50% recall, balanced performance
- **0.02 (2%)**: ~30% recall, more selective
- **0.05 (5%)**: ~10% recall, very selective

Lower values provide higher precision with lower recall, while higher values provide higher recall with lower precision.

## Understanding Results

### Key Metrics
- **Anomalies Detected**: Number of drives flagged as potentially failing
- **Actual Failures**: Number of drives that actually failed in the dataset
- **Recall**: Percentage of actual failures that were correctly identified
- **Precision**: Percentage of flagged drives that actually failed

### Performance
The model achieves up to 90% recall on the 2013 Backblaze dataset, successfully identifying the majority of drive failures while maintaining reasonable precision rates.

## Technical Details

### Algorithm
- **Isolation Forest**: Unsupervised learning algorithm for anomaly detection
- **Feature Engineering**: Advanced feature creation including error rates and risk indicators
- **Threshold-based Detection**: Dynamic threshold calculation based on actual failure patterns

### Data Processing
- Combines multiple CSV files from the specified folder
- Handles missing values and data normalization
- Creates engineered features for improved model performance

## Project Structure

```
Predictive Drive Health Monitor/
├── app.py                 # Streamlit web application
├── drive_analyzer.py      # Core ML logic and data processing
├── run_analysis.py        # Command-line interface
├── requirements.txt       # Core dependencies
├── requirements-streamlit.txt  # Web interface dependencies
├── README.md             # Documentation
├── .gitignore            # Git ignore rules
└── training_data/        # Data folder (not included in repository)
    └── README.md         # Data download instructions
```

## References

- [Backblaze Drive Stats](https://www.backblaze.com/b2/hard-drive-test-data.html)
- [Isolation Forest Algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [S.M.A.R.T. Attributes](https://en.wikipedia.org/wiki/S.M.A.R.T.)

## License

This project is open source and available under the MIT License.

## Disclaimer

This tool is for educational and research purposes. Always consult with hardware professionals for critical drive health decisions.