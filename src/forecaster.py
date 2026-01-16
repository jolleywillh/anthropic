"""
Forecaster Module
Generates Day-Ahead price forecasts using trained models
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from typing import Optional, Tuple
import yaml

from models import PriceForecastModel
from feature_engineering import FeatureEngineer
from data_collector import ERCOTDataCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DayAheadForecaster:
    """Generates 24-hour Day-Ahead price forecasts"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the forecaster

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.model = PriceForecastModel(config_path)
        self.engineer = FeatureEngineer(
            lag_features=self.config['model']['lag_features'],
            rolling_windows=self.config['model']['rolling_windows']
        )
        self.collector = ERCOTDataCollector(
            data_dir=self.config['data']['raw_data_dir']
        )

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Config file {config_path} not found")
            return {}

    def load_historical_data(self) -> pd.DataFrame:
        """
        Load historical data for feature creation

        Returns:
            DataFrame with historical price data
        """
        logger.info("Loading historical data")

        # Try to load from file
        df = self.collector.load_data()

        if df.empty:
            # Fetch fresh data
            logger.info("No cached data found, fetching from ERCOT")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config['data']['lookback_days'])
            df = self.collector.fetch_and_save(start_date=start_date, end_date=end_date)

        return df

    def prepare_forecast_features(
        self,
        historical_data: pd.DataFrame,
        forecast_start: Optional[datetime] = None,
        forecast_hours: int = 24
    ) -> pd.DataFrame:
        """
        Prepare features for forecasting

        Args:
            historical_data: Historical price data
            forecast_start: Start time for forecast (default: next hour)
            forecast_hours: Number of hours to forecast

        Returns:
            DataFrame with features for forecasting
        """
        logger.info(f"Preparing features for {forecast_hours}-hour forecast")

        if forecast_start is None:
            # Default: start from next hour
            last_time = historical_data['datetime'].max()
            forecast_start = last_time + timedelta(hours=1)

        # Create datetime range for forecast
        forecast_dates = pd.date_range(
            start=forecast_start,
            periods=forecast_hours,
            freq='h'
        )

        # Create features from historical data
        df_features = self.engineer.create_features(historical_data)

        # For forecasting, we need to iteratively predict and append
        # to build features for future timestamps
        forecast_features = []

        for i, forecast_time in enumerate(forecast_dates):
            # Create a row for this forecast time
            row = {'datetime': forecast_time}

            # Add temporal features
            row['hour'] = forecast_time.hour
            row['day_of_week'] = forecast_time.dayofweek
            row['is_weekend'] = int(forecast_time.dayofweek >= 5)
            row['month'] = forecast_time.month
            row['quarter'] = forecast_time.quarter
            row['day_of_year'] = forecast_time.dayofyear
            row['week_of_year'] = forecast_time.isocalendar()[1]

            # Cyclical features
            row['hour_sin'] = np.sin(2 * np.pi * row['hour'] / 24)
            row['hour_cos'] = np.cos(2 * np.pi * row['hour'] / 24)
            row['day_of_week_sin'] = np.sin(2 * np.pi * row['day_of_week'] / 7)
            row['day_of_week_cos'] = np.cos(2 * np.pi * row['day_of_week'] / 7)
            row['month_sin'] = np.sin(2 * np.pi * row['month'] / 12)
            row['month_cos'] = np.cos(2 * np.pi * row['month'] / 12)

            # Peak hours
            row['is_peak_hour'] = int((row['hour'] >= 7) and (row['hour'] <= 22))
            row['is_super_peak'] = int(
                (row['hour'] >= 14) and (row['hour'] <= 19) and (row['day_of_week'] < 5)
            )

            # Seasonal
            row['is_summer'] = int(row['month'] in [6, 7, 8])
            row['is_winter'] = int(row['month'] in [12, 1, 2])

            # For lag and rolling features, we need to look back at historical + previously forecasted
            # Combine historical data with any forecasted points so far
            if i == 0:
                combined_data = df_features
            else:
                # Append previously forecasted points
                prev_forecast_df = pd.DataFrame(forecast_features)
                prev_forecast_df['datetime'] = pd.to_datetime(prev_forecast_df['datetime'])
                prev_forecast_df = prev_forecast_df.set_index('datetime')

                combined_data = pd.concat([df_features, prev_forecast_df])
                combined_data = combined_data.sort_index()

            # Calculate lag features
            for lag in self.engineer.lag_features:
                lag_time = forecast_time - timedelta(hours=lag)
                if lag_time in combined_data.index:
                    row[f'price_lag_{lag}h'] = combined_data.loc[lag_time, 'price']
                else:
                    # Use last available value if lag is not available
                    row[f'price_lag_{lag}h'] = combined_data['price'].iloc[-1]

            # Calculate rolling features
            for window in self.engineer.rolling_windows:
                window_start = forecast_time - timedelta(hours=window + 1)
                window_end = forecast_time - timedelta(hours=1)

                window_data = combined_data[
                    (combined_data.index >= window_start) &
                    (combined_data.index <= window_end)
                ]['price']

                if len(window_data) > 0:
                    row[f'price_rolling_mean_{window}h'] = window_data.mean()
                    row[f'price_rolling_std_{window}h'] = window_data.std()
                    row[f'price_rolling_min_{window}h'] = window_data.min()
                    row[f'price_rolling_max_{window}h'] = window_data.max()
                    row[f'price_rolling_median_{window}h'] = window_data.median()
                else:
                    # Use last available values
                    row[f'price_rolling_mean_{window}h'] = combined_data['price'].iloc[-window:].mean()
                    row[f'price_rolling_std_{window}h'] = combined_data['price'].iloc[-window:].std()
                    row[f'price_rolling_min_{window}h'] = combined_data['price'].iloc[-window:].min()
                    row[f'price_rolling_max_{window}h'] = combined_data['price'].iloc[-window:].max()
                    row[f'price_rolling_median_{window}h'] = combined_data['price'].iloc[-window:].median()

            # Interaction features
            if f'price_lag_1h' in row and f'price_lag_24h' in row:
                row['price_diff_1h'] = row['price_lag_1h'] - row['price_lag_24h']

            if f'price_lag_24h' in row and f'price_lag_168h' in row:
                row['price_diff_24h'] = row['price_lag_24h'] - row['price_lag_168h']

            if f'price_lag_24h' in row and f'price_rolling_mean_168h' in row:
                row['price_ratio_24h_168h'] = row['price_lag_24h'] / (row['price_rolling_mean_168h'] + 1e-6)

            if f'price_rolling_std_24h' in row and f'price_rolling_mean_24h' in row:
                row['price_volatility_24h'] = row['price_rolling_std_24h'] / (row['price_rolling_mean_24h'] + 1e-6)

            if f'price_rolling_max_24h' in row and f'price_rolling_min_24h' in row:
                row['price_range_24h'] = row['price_rolling_max_24h'] - row['price_rolling_min_24h']

            # Make prediction for this timestamp
            row_df = pd.DataFrame([row])
            row_df = row_df.set_index('datetime')

            # Ensure all expected features are present
            feature_cols = [col for col in df_features.columns if col != 'price']
            for col in feature_cols:
                if col not in row_df.columns:
                    row_df[col] = 0  # Fill missing features with 0

            # Reorder columns to match training data
            row_df = row_df[feature_cols]

            # Predict
            if len(self.model.models) > 0:
                price_pred = self.model.predict(row_df, use_ensemble=True)[0]
            else:
                # If model not loaded, use a placeholder
                price_pred = row.get('price_lag_24h', 30.0)

            row['price'] = price_pred
            forecast_features.append(row)

        # Convert to DataFrame
        forecast_df = pd.DataFrame(forecast_features)
        forecast_df['datetime'] = pd.to_datetime(forecast_df['datetime'])
        forecast_df = forecast_df.set_index('datetime')

        logger.info(f"Forecast features prepared for {len(forecast_df)} hours")

        return forecast_df

    def generate_forecast(
        self,
        forecast_start: Optional[datetime] = None,
        forecast_hours: int = 24,
        include_confidence: bool = True
    ) -> pd.DataFrame:
        """
        Generate Day-Ahead price forecast

        Args:
            forecast_start: Start time for forecast
            forecast_hours: Number of hours to forecast
            include_confidence: Whether to include confidence intervals

        Returns:
            DataFrame with forecast
        """
        logger.info(f"Generating {forecast_hours}-hour Day-Ahead forecast")

        # Load model if not already loaded
        if len(self.model.models) == 0:
            logger.info("Loading trained model")
            self.model.load_models()

        # Load historical data
        historical_data = self.load_historical_data()

        # Prepare features
        forecast_df = self.prepare_forecast_features(
            historical_data,
            forecast_start=forecast_start,
            forecast_hours=forecast_hours
        )

        # Create result DataFrame
        result = pd.DataFrame({
            'datetime': forecast_df.index,
            'forecasted_price': forecast_df['price']
        })

        # Add confidence intervals (simple approach using historical error)
        if include_confidence:
            # Estimate prediction uncertainty from historical data
            # For simplicity, use 1.5 * RMSE for 90% CI
            historical_std = historical_data['price'].std()
            ci_width = 1.5 * historical_std

            result['lower_bound'] = result['forecasted_price'] - ci_width
            result['upper_bound'] = result['forecasted_price'] + ci_width

            # Ensure non-negative bounds
            result['lower_bound'] = result['lower_bound'].clip(lower=0)

        logger.info(f"Forecast generated successfully")
        logger.info(f"Forecast range: ${result['forecasted_price'].min():.2f} - ${result['forecasted_price'].max():.2f}")
        logger.info(f"Mean forecast: ${result['forecasted_price'].mean():.2f}")

        return result

    def save_forecast(self, forecast_df: pd.DataFrame, filename: Optional[str] = None):
        """
        Save forecast to file

        Args:
            forecast_df: Forecast DataFrame
            filename: Output filename (default: forecast_YYYYMMDD_HHMMSS.csv)
        """
        forecast_dir = self.config['output']['forecast_dir']
        os.makedirs(forecast_dir, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"forecast_{timestamp}.csv"

        filepath = os.path.join(forecast_dir, filename)
        forecast_df.to_csv(filepath, index=False)
        logger.info(f"Forecast saved to {filepath}")

        return filepath


if __name__ == "__main__":
    # Example usage
    forecaster = DayAheadForecaster()

    # Generate 24-hour forecast
    forecast = forecaster.generate_forecast(forecast_hours=24)

    print("\n24-Hour Day-Ahead Price Forecast:")
    print(forecast.to_string())

    # Save forecast
    forecaster.save_forecast(forecast)
