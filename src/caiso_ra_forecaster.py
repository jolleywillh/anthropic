"""
CAISO Resource Adequacy Price Forecaster

This module provides forecasting capabilities for CAISO Resource Adequacy (RA) prices
for years 2026-2030. RA prices represent the cost of capacity to ensure grid reliability.

Key features:
- Multi-year forecasting (2026-2030)
- Monthly granularity for RA prices
- Zone-specific forecasts (System, Local Capacity Areas)
- Trend-based projections incorporating load growth, renewables, and retirements
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CAISORAForecaster:
    """
    Forecaster for CAISO Resource Adequacy prices across multiple years.

    RA prices are influenced by:
    - Peak load growth
    - Renewable energy penetration
    - Generator retirements
    - Transmission constraints
    - Seasonal demand patterns
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the CAISO RA Forecaster.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self._default_config()
        self.forecast_data = None
        self.historical_data = None

    def _default_config(self) -> Dict:
        """Default configuration for CAISO RA forecasting."""
        return {
            # Base year for projections (2025 as reference)
            'base_year': 2025,
            'forecast_years': [2026, 2027, 2028, 2029, 2030],

            # Base RA prices ($/kW-month) for 2025 by zone and month
            # Based on recent CAISO RA market trends
            'base_prices': {
                'System': {
                    'Jan': 4.50, 'Feb': 4.25, 'Mar': 4.00, 'Apr': 4.75,
                    'May': 5.50, 'Jun': 6.75, 'Jul': 8.50, 'Aug': 9.25,
                    'Sep': 8.75, 'Oct': 6.50, 'Nov': 5.25, 'Dec': 4.75
                },
                'Big Creek/Ventura': {
                    'Jan': 6.50, 'Feb': 6.00, 'Mar': 5.75, 'Apr': 6.50,
                    'May': 7.50, 'Jun': 9.50, 'Jul': 12.00, 'Aug': 13.50,
                    'Sep': 12.50, 'Oct': 9.00, 'Nov': 7.25, 'Dec': 6.75
                },
                'LA Basin': {
                    'Jan': 7.00, 'Feb': 6.50, 'Mar': 6.25, 'Apr': 7.00,
                    'May': 8.25, 'Jun': 10.50, 'Jul': 13.50, 'Aug': 15.00,
                    'Sep': 14.00, 'Oct': 10.00, 'Nov': 8.00, 'Dec': 7.25
                },
                'San Diego/Imperial Valley': {
                    'Jan': 5.75, 'Feb': 5.50, 'Mar': 5.25, 'Apr': 6.00,
                    'May': 7.00, 'Jun': 8.75, 'Jul': 11.00, 'Aug': 12.00,
                    'Sep': 11.25, 'Oct': 8.25, 'Nov': 6.75, 'Dec': 6.00
                },
                'Sierra': {
                    'Jan': 4.25, 'Feb': 4.00, 'Mar': 3.75, 'Apr': 4.50,
                    'May': 5.25, 'Jun': 6.50, 'Jul': 8.00, 'Aug': 8.75,
                    'Sep': 8.25, 'Oct': 6.25, 'Nov': 5.00, 'Dec': 4.50
                }
            },

            # Annual growth factors
            'annual_load_growth': 0.015,  # 1.5% per year
            'renewable_penetration_impact': 0.025,  # 2.5% price increase per year (firming capacity)
            'retirement_impact': 0.020,  # 2.0% price increase (gas plant retirements)
            'battery_storage_impact': -0.015,  # -1.5% price decrease (new battery storage)

            # Volatility and uncertainty
            'base_volatility': 0.10,  # 10% standard deviation
            'volatility_growth': 0.02,  # Volatility increases over time

            # Zones to forecast
            'zones': ['System', 'Big Creek/Ventura', 'LA Basin',
                     'San Diego/Imperial Valley', 'Sierra']
        }

    def generate_historical_baseline(self) -> pd.DataFrame:
        """
        Generate a baseline of historical RA prices for context.
        Creates synthetic historical data for 2023-2025.

        Returns:
            DataFrame with historical RA prices
        """
        logger.info("Generating historical baseline data (2023-2025)")

        historical_records = []
        base_prices = self.config['base_prices']
        zones = self.config['zones']

        # Generate data for 2023-2025
        for year in [2023, 2024, 2025]:
            # Apply backward trend adjustment (prices were lower in past)
            years_back = 2025 - year
            trend_factor = 1 - (years_back * 0.05)  # 5% lower per year back

            for zone in zones:
                for month_idx, month in enumerate(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 1):
                    base_price = base_prices[zone][month]
                    adjusted_price = base_price * trend_factor

                    # Add some historical noise
                    noise = np.random.normal(0, adjusted_price * 0.08)
                    final_price = max(1.0, adjusted_price + noise)

                    historical_records.append({
                        'year': year,
                        'month': month,
                        'month_num': month_idx,
                        'zone': zone,
                        'ra_price': round(final_price, 2),
                        'date': pd.Timestamp(year=year, month=month_idx, day=1)
                    })

        self.historical_data = pd.DataFrame(historical_records)
        logger.info(f"Generated {len(historical_records)} historical data points")
        return self.historical_data

    def generate_forecast(self, include_uncertainty: bool = True) -> pd.DataFrame:
        """
        Generate RA price forecasts for 2026-2030.

        Args:
            include_uncertainty: Whether to include confidence intervals

        Returns:
            DataFrame with forecasted RA prices
        """
        logger.info("Generating CAISO RA price forecast for 2026-2030")

        forecast_records = []
        base_prices = self.config['base_prices']
        zones = self.config['zones']
        forecast_years = self.config['forecast_years']

        # Growth factors
        load_growth = self.config['annual_load_growth']
        renewable_impact = self.config['renewable_penetration_impact']
        retirement_impact = self.config['retirement_impact']
        battery_impact = self.config['battery_storage_impact']

        # Net annual price growth rate
        net_growth_rate = (load_growth + renewable_impact +
                          retirement_impact + battery_impact)

        logger.info(f"Net annual growth rate: {net_growth_rate:.2%}")

        for year in forecast_years:
            years_ahead = year - self.config['base_year']

            # Compound growth factor
            growth_factor = (1 + net_growth_rate) ** years_ahead

            # Volatility increases over time (more uncertainty in distant future)
            volatility = self.config['base_volatility'] + (
                years_ahead * self.config['volatility_growth']
            )

            for zone in zones:
                for month_idx, month in enumerate(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 1):
                    base_price = base_prices[zone][month]

                    # Apply growth factor
                    forecasted_price = base_price * growth_factor

                    # Calculate confidence intervals
                    if include_uncertainty:
                        std_dev = forecasted_price * volatility
                        lower_bound = forecasted_price - (1.96 * std_dev)  # 95% CI
                        upper_bound = forecasted_price + (1.96 * std_dev)
                    else:
                        lower_bound = forecasted_price
                        upper_bound = forecasted_price

                    # Ensure prices don't go negative
                    lower_bound = max(1.0, lower_bound)

                    forecast_records.append({
                        'year': year,
                        'month': month,
                        'month_num': month_idx,
                        'zone': zone,
                        'ra_price': round(forecasted_price, 2),
                        'lower_bound': round(lower_bound, 2),
                        'upper_bound': round(upper_bound, 2),
                        'growth_factor': round(growth_factor, 3),
                        'date': pd.Timestamp(year=year, month=month_idx, day=1)
                    })

        self.forecast_data = pd.DataFrame(forecast_records)
        logger.info(f"Generated {len(forecast_records)} forecast data points")

        return self.forecast_data

    def get_annual_summary(self) -> pd.DataFrame:
        """
        Generate annual summary statistics for each zone.

        Returns:
            DataFrame with annual averages and ranges
        """
        if self.forecast_data is None:
            raise ValueError("No forecast data available. Run generate_forecast() first.")

        summary = self.forecast_data.groupby(['zone', 'year']).agg({
            'ra_price': ['mean', 'min', 'max', 'std'],
            'lower_bound': 'mean',
            'upper_bound': 'mean'
        }).round(2)

        summary.columns = ['avg_price', 'min_price', 'max_price',
                          'std_dev', 'avg_lower_bound', 'avg_upper_bound']
        summary = summary.reset_index()

        return summary

    def save_forecast(self, output_dir: str = 'forecasts') -> str:
        """
        Save forecast data to CSV file.

        Args:
            output_dir: Directory to save forecast

        Returns:
            Path to saved file
        """
        if self.forecast_data is None:
            raise ValueError("No forecast data available. Run generate_forecast() first.")

        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'caiso_ra_forecast_{timestamp}.csv'
        filepath = os.path.join(output_dir, filename)

        self.forecast_data.to_csv(filepath, index=False)
        logger.info(f"Forecast saved to {filepath}")

        # Also save annual summary
        summary_filename = f'caiso_ra_annual_summary_{timestamp}.csv'
        summary_filepath = os.path.join(output_dir, summary_filename)
        summary = self.get_annual_summary()
        summary.to_csv(summary_filepath, index=False)
        logger.info(f"Annual summary saved to {summary_filepath}")

        return filepath

    def save_metadata(self, output_dir: str = 'forecasts') -> str:
        """
        Save forecast metadata and configuration.

        Args:
            output_dir: Directory to save metadata

        Returns:
            Path to saved metadata file
        """
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'caiso_ra_metadata_{timestamp}.json'
        filepath = os.path.join(output_dir, filename)

        metadata = {
            'forecast_type': 'CAISO Resource Adequacy Prices',
            'generated_at': datetime.now().isoformat(),
            'forecast_years': self.config['forecast_years'],
            'zones': self.config['zones'],
            'methodology': 'Trend-based projection with compound growth factors',
            'growth_factors': {
                'annual_load_growth': self.config['annual_load_growth'],
                'renewable_penetration_impact': self.config['renewable_penetration_impact'],
                'retirement_impact': self.config['retirement_impact'],
                'battery_storage_impact': self.config['battery_storage_impact'],
                'net_growth_rate': (
                    self.config['annual_load_growth'] +
                    self.config['renewable_penetration_impact'] +
                    self.config['retirement_impact'] +
                    self.config['battery_storage_impact']
                )
            },
            'units': '$/kW-month',
            'confidence_interval': '95%'
        }

        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved to {filepath}")
        return filepath
