"""
CAISO Resource Adequacy Forecast Visualization

This module provides visualization capabilities for CAISO RA price forecasts,
including multi-year trends, zone comparisons, and seasonal patterns.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class CAISORAVisualizer:
    """Visualizer for CAISO Resource Adequacy forecasts."""

    def __init__(self, forecast_data: pd.DataFrame, historical_data: Optional[pd.DataFrame] = None):
        """
        Initialize visualizer with forecast data.

        Args:
            forecast_data: DataFrame with forecast data
            historical_data: Optional DataFrame with historical data
        """
        self.forecast_data = forecast_data
        self.historical_data = historical_data

    def plot_multi_year_trends(self, output_dir: str = 'plots', filename: str = None) -> str:
        """
        Create comprehensive multi-year trend plot for all zones.

        Args:
            output_dir: Directory to save plot
            filename: Optional custom filename

        Returns:
            Path to saved plot
        """
        logger.info("Generating multi-year trends plot")

        zones = self.forecast_data['zone'].unique()
        n_zones = len(zones)

        fig, axes = plt.subplots(n_zones, 1, figsize=(16, 4 * n_zones))
        if n_zones == 1:
            axes = [axes]

        colors = sns.color_palette("husl", n_zones)

        for idx, zone in enumerate(zones):
            ax = axes[idx]
            zone_data = self.forecast_data[self.forecast_data['zone'] == zone]

            # Plot main forecast line
            ax.plot(zone_data['date'], zone_data['ra_price'],
                   label='Forecast', color=colors[idx], linewidth=2, marker='o', markersize=3)

            # Plot confidence intervals
            ax.fill_between(zone_data['date'],
                          zone_data['lower_bound'],
                          zone_data['upper_bound'],
                          alpha=0.2, color=colors[idx], label='95% Confidence Interval')

            # Add historical data if available
            if self.historical_data is not None:
                hist_zone = self.historical_data[self.historical_data['zone'] == zone]
                ax.plot(hist_zone['date'], hist_zone['ra_price'],
                       label='Historical', color='gray', linewidth=1.5,
                       linestyle='--', marker='s', markersize=2)

            ax.set_title(f'{zone} - RA Price Forecast (2026-2030)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=11)
            ax.set_ylabel('RA Price ($/kW-month)', fontsize=11)
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)

            # Add year separators
            years = sorted(zone_data['year'].unique())
            for year in years[1:]:
                year_start = pd.Timestamp(year=year, month=1, day=1)
                ax.axvline(x=year_start, color='red', linestyle=':', alpha=0.5, linewidth=1)

        plt.tight_layout()

        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'caiso_ra_multi_year_trends_{timestamp}.png'

        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Multi-year trends plot saved to {filepath}")
        return filepath

    def plot_zone_comparison(self, output_dir: str = 'plots', filename: str = None) -> str:
        """
        Create zone comparison plot showing average annual prices.

        Args:
            output_dir: Directory to save plot
            filename: Optional custom filename

        Returns:
            Path to saved plot
        """
        logger.info("Generating zone comparison plot")

        # Calculate annual averages
        annual_avg = self.forecast_data.groupby(['zone', 'year'])['ra_price'].mean().reset_index()

        fig, ax = plt.subplots(figsize=(14, 8))

        zones = annual_avg['zone'].unique()
        x = np.arange(len(annual_avg['year'].unique()))
        width = 0.15
        colors = sns.color_palette("husl", len(zones))

        for idx, zone in enumerate(zones):
            zone_data = annual_avg[annual_avg['zone'] == zone]
            offset = (idx - len(zones) / 2) * width + width / 2
            ax.bar(x + offset, zone_data['ra_price'], width,
                  label=zone, color=colors[idx], alpha=0.8)

        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average RA Price ($/kW-month)', fontsize=12, fontweight='bold')
        ax.set_title('CAISO RA Price Forecast - Zone Comparison (2026-2030)',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(annual_avg['year'].unique())
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'caiso_ra_zone_comparison_{timestamp}.png'

        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Zone comparison plot saved to {filepath}")
        return filepath

    def plot_seasonal_patterns(self, output_dir: str = 'plots', filename: str = None) -> str:
        """
        Create seasonal pattern plot showing monthly variations.

        Args:
            output_dir: Directory to save plot
            filename: Optional custom filename

        Returns:
            Path to saved plot
        """
        logger.info("Generating seasonal patterns plot")

        zones = self.forecast_data['zone'].unique()
        years = sorted(self.forecast_data['year'].unique())

        fig, axes = plt.subplots(len(years), 1, figsize=(16, 4 * len(years)))
        if len(years) == 1:
            axes = [axes]

        colors = sns.color_palette("husl", len(zones))
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        for idx, year in enumerate(years):
            ax = axes[idx]
            year_data = self.forecast_data[self.forecast_data['year'] == year]

            for zone_idx, zone in enumerate(zones):
                zone_year_data = year_data[year_data['zone'] == zone].sort_values('month_num')
                ax.plot(zone_year_data['month_num'], zone_year_data['ra_price'],
                       label=zone, color=colors[zone_idx], linewidth=2.5,
                       marker='o', markersize=6)

            ax.set_title(f'Seasonal RA Price Patterns - {year}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Month', fontsize=11)
            ax.set_ylabel('RA Price ($/kW-month)', fontsize=11)
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(months)
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'caiso_ra_seasonal_patterns_{timestamp}.png'

        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Seasonal patterns plot saved to {filepath}")
        return filepath

    def plot_price_heatmap(self, output_dir: str = 'plots', filename: str = None) -> str:
        """
        Create heatmap showing price variations across zones and years.

        Args:
            output_dir: Directory to save plot
            filename: Optional custom filename

        Returns:
            Path to saved plot
        """
        logger.info("Generating price heatmap")

        # Calculate annual averages for heatmap
        annual_avg = self.forecast_data.groupby(['zone', 'year'])['ra_price'].mean().reset_index()
        heatmap_data = annual_avg.pivot(index='zone', columns='year', values='ra_price')

        fig, ax = plt.subplots(figsize=(12, 6))

        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd',
                   cbar_kws={'label': 'RA Price ($/kW-month)'}, ax=ax,
                   linewidths=1, linecolor='white')

        ax.set_title('CAISO RA Price Forecast Heatmap - Average Annual Prices',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Zone', fontsize=12, fontweight='bold')

        plt.tight_layout()

        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'caiso_ra_price_heatmap_{timestamp}.png'

        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Price heatmap saved to {filepath}")
        return filepath

    def plot_growth_trends(self, output_dir: str = 'plots', filename: str = None) -> str:
        """
        Create plot showing price growth trends over time.

        Args:
            output_dir: Directory to save plot
            filename: Optional custom filename

        Returns:
            Path to saved plot
        """
        logger.info("Generating growth trends plot")

        zones = self.forecast_data['zone'].unique()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        colors = sns.color_palette("husl", len(zones))

        # Plot 1: Average annual price by zone
        annual_avg = self.forecast_data.groupby(['zone', 'year'])['ra_price'].mean().reset_index()

        for idx, zone in enumerate(zones):
            zone_data = annual_avg[annual_avg['zone'] == zone]
            ax1.plot(zone_data['year'], zone_data['ra_price'],
                    label=zone, color=colors[idx], linewidth=2.5,
                    marker='o', markersize=8)

        ax1.set_title('Average Annual RA Prices by Zone (2026-2030)',
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Year', fontsize=11)
        ax1.set_ylabel('Average RA Price ($/kW-month)', fontsize=11)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Year-over-year growth rates
        growth_data = []
        for zone in zones:
            zone_annual = annual_avg[annual_avg['zone'] == zone].sort_values('year')
            prices = zone_annual['ra_price'].values
            years = zone_annual['year'].values[1:]

            growth_rates = [(prices[i] - prices[i-1]) / prices[i-1] * 100
                          for i in range(1, len(prices))]

            for year, growth in zip(years, growth_rates):
                growth_data.append({'zone': zone, 'year': year, 'growth_rate': growth})

        growth_df = pd.DataFrame(growth_data)

        x = np.arange(len(growth_df['year'].unique()))
        width = 0.15

        for idx, zone in enumerate(zones):
            zone_growth = growth_df[growth_df['zone'] == zone]
            offset = (idx - len(zones) / 2) * width + width / 2
            ax2.bar(x + offset, zone_growth['growth_rate'], width,
                   label=zone, color=colors[idx], alpha=0.8)

        ax2.set_title('Year-over-Year RA Price Growth Rates by Zone',
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Year', fontsize=11)
        ax2.set_ylabel('Growth Rate (%)', fontsize=11)
        ax2.set_xticks(x)
        ax2.set_xticklabels(growth_df['year'].unique())
        ax2.legend(loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()

        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'caiso_ra_growth_trends_{timestamp}.png'

        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Growth trends plot saved to {filepath}")
        return filepath

    def generate_all_plots(self, output_dir: str = 'plots') -> dict:
        """
        Generate all visualization plots.

        Args:
            output_dir: Directory to save plots

        Returns:
            Dictionary with plot names and file paths
        """
        logger.info("Generating all visualization plots")

        plots = {
            'multi_year_trends': self.plot_multi_year_trends(output_dir),
            'zone_comparison': self.plot_zone_comparison(output_dir),
            'seasonal_patterns': self.plot_seasonal_patterns(output_dir),
            'price_heatmap': self.plot_price_heatmap(output_dir),
            'growth_trends': self.plot_growth_trends(output_dir)
        }

        logger.info(f"Generated {len(plots)} visualization plots")
        return plots
