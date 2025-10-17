# 🤖 Predictive Drive Health Monitor

A machine learning-powered web application that uses **Isolation Forest** algorithm to proactively detect potentially failing SSDs and HDDs from operational S.M.A.R.T. data.

## 🎯 Overview

This project analyzes drive health data from Backblaze's publicly available drive statistics to predict drive failures before they occur. The application provides an interactive web interface for uploading data, running anomaly detection, and visualizing results.

## ✨ Features

- **Interactive Web Interface**: Built with Streamlit for easy data analysis
- **Anomaly Detection**: Uses Isolation Forest algorithm to identify potentially failing drives
- **Real-time Visualization**: Charts and graphs showing drive health patterns
- **Performance Metrics**: Detailed evaluation reports with precision and recall scores
- **Configurable Parameters**: Adjustable contamination rate for different failure scenarios

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd "Predictive Drive Health Monitor"
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

#### Option 1: Command Line Version (Recommended)
```bash
python run_analysis.py
```

#### Option 2: Web Interface (Requires Streamlit)
```bash
pip install -r requirements-streamlit.txt
streamlit run app.py
```
Then open your browser to `http://localhost:8501`

**Note:** If you encounter PyArrow installation issues, the command-line version works without Streamlit and provides the same analysis functionality.

## 📊 Data Requirements

The application expects CSV files from [Backblaze Drive Stats](https://www.backblaze.com/b2/hard-drive-test-data.html) in the `training_data/` folder. The current setup includes 2013 data with the following S.M.A.R.T. attributes:

- **smart_5_raw**: Reallocated sectors count
- **smart_9_raw**: Power-on hours
- **smart_187_raw**: Uncorrectable errors
- **smart_194_raw**: Temperature
- **smart_197_raw**: Pending sectors
- **smart_198_raw**: Offline uncorrectable errors

## 🎛️ Usage

1. **Set Data Path**: Enter the path to your data folder (default: `training_data/`)
2. **Adjust Contamination Rate**: Set the expected anomaly rate (default: 0.01 or 1%)
3. **Run Analysis**: Click "Run Analysis" to start the anomaly detection
4. **Review Results**: Examine the detected anomalies and performance metrics
5. **Explore Visualizations**: View charts showing drive health patterns

## 📈 Understanding the Results

### Key Metrics
- **Anomalies Detected**: Number of drives flagged as potentially failing
- **Actual Failures**: Number of drives that actually failed in the dataset
- **Anomaly Rate**: The contamination parameter you set

### Performance Evaluation
- **Precision (Failed)**: Of all drives flagged as anomalies, what percentage actually failed?
- **Recall (Failed)**: Of all drives that actually failed, what percentage did the model catch?

### Visualizations
- **Anomaly Score Distribution**: Histogram showing the distribution of anomaly scores
- **Temperature vs Reallocated Sectors**: Scatter plot for anomalous drives

## 🔧 Technical Details

### Algorithm
- **Isolation Forest**: An unsupervised learning algorithm that isolates anomalies by randomly selecting features and splitting values
- **StandardScaler**: Normalizes features for better model performance
- **100 Estimators**: Uses 100 decision trees for robust anomaly detection

### Data Processing
- Combines multiple CSV files from the specified folder
- Handles missing values by filling with zeros
- Renames S.M.A.R.T. attributes for better readability
- Filters to essential drive health metrics

## 📁 Project Structure

```
Predictive Drive Health Monitor/
├── app.py                 # Main Streamlit application
├── drive_analyzer.py      # Core ML logic and data processing
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore            # Git ignore rules
└── training_data/        # CSV files from Backblaze (2013 data)
    ├── 2013-04-10.csv
    ├── 2013-04-11.csv
    └── ... (266 files total)
```

## 🛠️ Customization

### Adjusting Contamination Rate
- **Lower values (0.001-0.005)**: More conservative, fewer false positives
- **Higher values (0.05-0.1)**: More aggressive, catches more potential failures

### Adding New Features
You can extend the model by:
1. Adding new S.M.A.R.T. attributes to `smart_columns` in `drive_analyzer.py`
2. Including them in the `feature_columns` list
3. Updating the column renaming dictionary

## 📚 References

- [Backblaze Drive Stats](https://www.backblaze.com/b2/hard-drive-test-data.html)
- [Isolation Forest Algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [S.M.A.R.T. Attributes](https://en.wikipedia.org/wiki/S.M.A.R.T.)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## ⚠️ Disclaimer

This tool is for educational and research purposes. Always consult with hardware professionals for critical drive health decisions.
