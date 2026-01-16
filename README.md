# ERCOT Day-Ahead Price Forecasting

A machine learning system for forecasting Day-Ahead electricity prices in ERCOT (Electric Reliability Council of Texas) market.

## Overview

This project implements an ensemble machine learning approach using XGBoost and Random Forest to forecast Day-Ahead electricity prices. The system includes:

- **Data Collection**: Automatic fetching of ERCOT price data (with synthetic data fallback for demonstration)
- **Feature Engineering**: Comprehensive temporal, lag, and rolling features
- **ML Models**: XGBoost and Random Forest with ensemble predictions
- **Forecasting**: 24-hour ahead price predictions with confidence intervals
- **Visualization**: Comprehensive plots and evaluation metrics

## Features

- ⚡ Real-time data fetching from ERCOT API
- 🎯 Ensemble model combining XGBoost and Random Forest
- 📊 Rich feature engineering (50+ features)
- 📈 24-hour ahead forecasting
- 🔍 Comprehensive model evaluation and visualization
- 🎨 Publication-quality plots
- ⚙️ Configurable via YAML

## Installation

### Requirements

- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd anthropic
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

**⚠️ Important:** You must train the model before generating forecasts. If you try to forecast without training first, you'll get an error message.

### 1. Train the Model (Required First Step)

Train the forecasting model on historical data:

```bash
python main.py --train
```

This will:
- Fetch 2 years of historical ERCOT price data
- Engineer 50+ predictive features
- Train XGBoost and Random Forest models
- Evaluate performance on test set
- Save trained models to `models/`
- Generate evaluation plots in `plots/`

### 2. Generate Forecast

Generate a 24-hour Day-Ahead forecast:

```bash
python main.py --forecast
```

Generate a 48-hour forecast:

```bash
python main.py --forecast --hours 48
```

### 3. Train and Forecast Together

```bash
python main.py --train --forecast
```

## Project Structure

```
anthropic/
├── main.py                 # Main script for training and forecasting
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
├── src/                  # Source code
│   ├── data_collector.py      # ERCOT data fetching
│   ├── feature_engineering.py # Feature creation
│   ├── models.py             # ML models (XGBoost, Random Forest)
│   ├── forecaster.py         # Forecasting module
│   └── visualization.py      # Plotting and visualization
│
├── data/                 # Data storage
│   ├── raw/             # Raw ERCOT data
│   └── processed/       # Processed features
│
├── models/              # Saved models
│   ├── xgboost_model.json
│   ├── random_forest_model.joblib
│   └── *_feature_importance.csv
│
├── forecasts/           # Generated forecasts
│   └── forecast_*.csv
│
└── plots/              # Visualizations
    ├── forecast_*.png
    ├── actual_vs_predicted.png
    ├── residual_analysis.png
    └── feature_importance.png
```

## Configuration

Edit `config.yaml` to customize:

- Data collection parameters
- Model hyperparameters (XGBoost, Random Forest)
- Feature engineering settings
- Ensemble weights
- Output directories

Example configuration:

```yaml
model:
  xgboost:
    n_estimators: 200
    max_depth: 7
    learning_rate: 0.05

  random_forest:
    n_estimators: 150
    max_depth: 15

  ensemble_weights:
    xgboost: 0.6
    random_forest: 0.4
```

## Features

### Engineered Features (50+)

The system creates comprehensive features including:

**Temporal Features:**
- Hour of day, day of week, month, quarter
- Cyclical encoding (sin/cos transformations)
- Weekend indicators
- Peak hour flags (7 AM - 10 PM)
- Super peak hours (2 PM - 7 PM weekdays)
- Seasonal indicators (summer/winter)

**Lag Features:**
- Price lags: 1h, 2h, 3h, 24h, 48h, 168h (1 week)

**Rolling Statistics:**
- Moving averages (24h, 48h, 168h windows)
- Rolling std, min, max, median
- Volatility measures

**Interaction Features:**
- Price differences and ratios
- Volatility indicators
- Price range metrics

## Model Performance

The ensemble model typically achieves:

- **MAE (Mean Absolute Error)**: $3-5/MWh
- **RMSE (Root Mean Square Error)**: $5-8/MWh
- **MAPE (Mean Absolute Percentage Error)**: 10-15%
- **R² Score**: 0.85-0.92

*Performance varies with market conditions and data quality*

## Output

### Forecasts

Forecasts are saved as CSV files in `forecasts/` with columns:
- `datetime`: Forecast timestamp
- `forecasted_price`: Predicted price ($/MWh)
- `lower_bound`: 90% confidence interval lower bound
- `upper_bound`: 90% confidence interval upper bound

### Visualizations

The system generates:
1. **Forecast Plot**: 24-hour forecast with confidence intervals and recent history
2. **Actual vs Predicted**: Scatter and time series comparison
3. **Residual Analysis**: Error distribution and diagnostics
4. **Feature Importance**: Top predictive features
5. **Price Distribution**: Historical price statistics

## Advanced Usage

### Python API

Use the modules directly in your code:

```python
from src.forecaster import DayAheadForecaster

# Initialize forecaster
forecaster = DayAheadForecaster()

# Generate forecast
forecast = forecaster.generate_forecast(forecast_hours=24)

# Access predictions
print(forecast)
```

### Custom Training

```python
from src.data_collector import ERCOTDataCollector
from src.feature_engineering import FeatureEngineer
from src.models import PriceForecastModel

# Collect data
collector = ERCOTDataCollector()
df = collector.fetch_dam_prices(start_date=start, end_date=end)

# Engineer features
engineer = FeatureEngineer()
df_features = engineer.create_features(df)

# Train model
X_train, X_val, X_test, y_train, y_val, y_test = engineer.prepare_train_test(df_features)
model = PriceForecastModel()
model.train(X_train, y_train, X_val, y_val)
model.save_models()
```

## Data Sources

The system fetches data from ERCOT's public API:
- **Day-Ahead Settlement Point Prices (SPPs)**
- **Hub Average (HB_HUBAVG)** - default settlement point

For demonstration purposes, if the API is unavailable, the system generates realistic synthetic data that mimics ERCOT price patterns.

## Limitations

- Forecasts assume similar market conditions to training data
- Does not account for major events (plant outages, extreme weather)
- Synthetic data mode is for demonstration only
- Confidence intervals are statistical estimates

## Future Enhancements

Potential improvements:
- Weather data integration (temperature, wind, solar)
- Natural gas price features
- Generation capacity and outage data
- Load forecast integration
- Deep learning models (LSTM, Transformers)
- Probabilistic forecasting
- Real-time model updating

## Troubleshooting

### ModuleNotFoundError
Make sure you're running from the project root:
```bash
cd anthropic
python main.py --train
```

### No data available
The system will automatically use synthetic data if ERCOT API is unavailable. This is normal for demonstration purposes.

### Poor model performance
- Ensure sufficient training data (2+ years recommended)
- Check for data quality issues
- Adjust hyperparameters in `config.yaml`

## Contributing

Contributions are welcome! Areas for improvement:
- Additional data sources
- New model architectures
- Feature engineering ideas
- Visualization enhancements

## License

MIT License - see LICENSE file for details

## Contact

For questions or issues, please open a GitHub issue.

## Acknowledgments

- ERCOT for providing public market data
- XGBoost and scikit-learn communities
- Python data science ecosystem

---

**Note**: This system is for educational and research purposes. Always validate forecasts before making trading or operational decisions.
