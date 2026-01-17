"""
Feature Engineering Module
Creates features from raw ERCOT Day-Ahead price data
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates features for price forecasting"""

    def __init__(
        self,
        lag_features: List[int] = [1, 2, 3, 24, 48, 168],
        rolling_windows: List[int] = [24, 48, 168]
    ):
        """
        Initialize feature engineer

        Args:
            lag_features: List of lag periods (in hours) to create
            rolling_windows: List of rolling window sizes (in hours)
        """
        self.lag_features = lag_features
        self.rolling_windows = rolling_windows

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features from raw data

        Args:
            df: DataFrame with datetime and price columns

        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering")

        df = df.copy()

        # Ensure datetime is the index
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')

        df = df.sort_index()

        # Encode settlement_point as categorical features if present
        if 'settlement_point' in df.columns:
            df = self._encode_settlement_point(df)

        # Create temporal features
        df = self._create_temporal_features(df)

        # Create lag features
        df = self._create_lag_features(df)

        # Create rolling statistics
        df = self._create_rolling_features(df)

        # Create interaction features
        df = self._create_interaction_features(df)

        # Drop rows with NaN values (from lag/rolling features)
        initial_rows = len(df)
        df = df.dropna()
        logger.info(f"Dropped {initial_rows - len(df)} rows with NaN values")

        logger.info(f"Feature engineering complete. Created {len(df.columns)} features")

        return df

    def _encode_settlement_point(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode settlement point as one-hot encoded features

        Args:
            df: DataFrame with settlement_point column

        Returns:
            DataFrame with encoded settlement point features
        """
        logger.info("Encoding settlement point as categorical features")

        # Define the standard ERCOT hubs
        known_hubs = ['HB_HUBAVG', 'HB_HOUSTON', 'HB_NORTH', 'HB_SOUTH', 'HB_WEST']

        # One-hot encode the settlement point
        for hub in known_hubs:
            df[f'hub_{hub}'] = (df['settlement_point'] == hub).astype(int)

        # Drop the original settlement_point column
        df = df.drop(columns=['settlement_point'])

        return df

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features

        Args:
            df: DataFrame with datetime index

        Returns:
            DataFrame with temporal features
        """
        logger.info("Creating temporal features")

        # Hour of day
        df['hour'] = df.index.hour

        # Day of week (0 = Monday, 6 = Sunday)
        df['day_of_week'] = df.index.dayofweek

        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Month
        df['month'] = df.index.month

        # Quarter
        df['quarter'] = df.index.quarter

        # Day of year
        df['day_of_year'] = df.index.dayofyear

        # Week of year
        df['week_of_year'] = df.index.isocalendar().week

        # Cyclical encoding for hour (captures continuity between hour 23 and 0)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Cyclical encoding for day of week
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Cyclical encoding for month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Peak hours indicator (7 AM - 10 PM)
        df['is_peak_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 22)).astype(int)

        # Super peak hours (2 PM - 7 PM on weekdays)
        df['is_super_peak'] = (
            (df['hour'] >= 14) & (df['hour'] <= 19) & (df['day_of_week'] < 5)
        ).astype(int)

        # Summer months (June, July, August - high demand in Texas)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)

        # Winter months (December, January, February)
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)

        return df

    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lagged price features

        Args:
            df: DataFrame with price column

        Returns:
            DataFrame with lag features
        """
        logger.info(f"Creating lag features: {self.lag_features}")

        for lag in self.lag_features:
            df[f'price_lag_{lag}h'] = df['price'].shift(lag)

        return df

    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling statistics features

        Args:
            df: DataFrame with price column

        Returns:
            DataFrame with rolling features
        """
        logger.info(f"Creating rolling features: {self.rolling_windows}")

        for window in self.rolling_windows:
            # Rolling mean
            df[f'price_rolling_mean_{window}h'] = (
                df['price'].shift(1).rolling(window=window).mean()
            )

            # Rolling std
            df[f'price_rolling_std_{window}h'] = (
                df['price'].shift(1).rolling(window=window).std()
            )

            # Rolling min
            df[f'price_rolling_min_{window}h'] = (
                df['price'].shift(1).rolling(window=window).min()
            )

            # Rolling max
            df[f'price_rolling_max_{window}h'] = (
                df['price'].shift(1).rolling(window=window).max()
            )

            # Rolling median
            df[f'price_rolling_median_{window}h'] = (
                df['price'].shift(1).rolling(window=window).median()
            )

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features

        Args:
            df: DataFrame with existing features

        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features")

        # Price differences
        if 'price_lag_1h' in df.columns:
            df['price_diff_1h'] = df['price_lag_1h'] - df['price_lag_24h']

        if 'price_lag_24h' in df.columns:
            df['price_diff_24h'] = df['price_lag_24h'] - df['price_lag_168h']

        # Ratio of recent to historical prices
        if 'price_lag_24h' in df.columns and 'price_rolling_mean_168h' in df.columns:
            df['price_ratio_24h_168h'] = (
                df['price_lag_24h'] / (df['price_rolling_mean_168h'] + 1e-6)
            )

        # Volatility measure
        if 'price_rolling_std_24h' in df.columns and 'price_rolling_mean_24h' in df.columns:
            df['price_volatility_24h'] = (
                df['price_rolling_std_24h'] / (df['price_rolling_mean_24h'] + 1e-6)
            )

        # Price range in last 24 hours
        if 'price_rolling_max_24h' in df.columns and 'price_rolling_min_24h' in df.columns:
            df['price_range_24h'] = (
                df['price_rolling_max_24h'] - df['price_rolling_min_24h']
            )

        return df

    def prepare_train_test(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        validation_size: float = 0.1
    ) -> tuple:
        """
        Split data into train, validation, and test sets

        Args:
            df: DataFrame with features
            test_size: Fraction for test set
            validation_size: Fraction for validation set (from train)

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Splitting data into train/validation/test sets")

        # Separate features and target
        feature_cols = [col for col in df.columns if col != 'price']
        X = df[feature_cols]
        y = df['price']

        # Time-based split (important for time series)
        test_idx = int(len(df) * (1 - test_size))
        val_idx = int(test_idx * (1 - validation_size))

        X_train = X.iloc[:val_idx]
        X_val = X.iloc[val_idx:test_idx]
        X_test = X.iloc[test_idx:]

        y_train = y.iloc[:val_idx]
        y_val = y.iloc[val_idx:test_idx]
        y_test = y.iloc[test_idx:]

        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Validation set: {len(X_val)} samples")
        logger.info(f"Test set: {len(X_test)} samples")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature names (excluding target)

        Args:
            df: DataFrame with features

        Returns:
            List of feature names
        """
        return [col for col in df.columns if col != 'price']


if __name__ == "__main__":
    # Example usage
    from data_collector import ERCOTDataCollector

    # Load data
    collector = ERCOTDataCollector()
    df = collector.load_data()

    if not df.empty:
        # Create features
        engineer = FeatureEngineer()
        df_features = engineer.create_features(df)

        print(f"\nFeature engineering complete:")
        print(f"Total features: {len(engineer.get_feature_names(df_features))}")
        print(f"\nFeature list:")
        for feature in engineer.get_feature_names(df_features)[:20]:
            print(f"  - {feature}")
        if len(engineer.get_feature_names(df_features)) > 20:
            print(f"  ... and {len(engineer.get_feature_names(df_features)) - 20} more")

        print(f"\nData shape: {df_features.shape}")
        print(f"\nFirst few rows:")
        print(df_features.head())
