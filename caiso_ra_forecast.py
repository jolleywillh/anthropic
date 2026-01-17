#!/usr/bin/env python3
"""
CAISO Resource Adequacy Price Forecast Generator

This script generates multi-year RA price forecasts for CAISO zones (2026-2030)
and creates comprehensive visualizations of the forecast results.

Usage:
    python caiso_ra_forecast.py

Output:
    - Forecast data (CSV)
    - Annual summary statistics (CSV)
    - Forecast metadata (JSON)
    - Comprehensive visualizations (PNG)
"""

import sys
import os
from datetime import datetime
import logging

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from caiso_ra_forecaster import CAISORAForecaster
from caiso_ra_visualization import CAISORAVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""

    print("=" * 80)
    print("CAISO Resource Adequacy Price Forecast")
    print("Forecast Period: 2026-2030")
    print("=" * 80)
    print()

    # Initialize forecaster
    logger.info("Initializing CAISO RA Forecaster")
    forecaster = CAISORAForecaster()

    # Display configuration
    print("Configuration:")
    print(f"  Base Year: {forecaster.config['base_year']}")
    print(f"  Forecast Years: {', '.join(map(str, forecaster.config['forecast_years']))}")
    print(f"  Zones: {', '.join(forecaster.config['zones'])}")
    print()

    # Display growth factors
    print("Growth Factors:")
    print(f"  Annual Load Growth: {forecaster.config['annual_load_growth']:.1%}")
    print(f"  Renewable Penetration Impact: {forecaster.config['renewable_penetration_impact']:.1%}")
    print(f"  Generator Retirement Impact: {forecaster.config['retirement_impact']:.1%}")
    print(f"  Battery Storage Impact: {forecaster.config['battery_storage_impact']:.1%}")

    net_growth = (
        forecaster.config['annual_load_growth'] +
        forecaster.config['renewable_penetration_impact'] +
        forecaster.config['retirement_impact'] +
        forecaster.config['battery_storage_impact']
    )
    print(f"  Net Growth Rate: {net_growth:.1%}")
    print()

    # Generate historical baseline
    print("Step 1: Generating historical baseline (2023-2025)...")
    historical_data = forecaster.generate_historical_baseline()
    print(f"  ✓ Generated {len(historical_data)} historical data points")
    print()

    # Generate forecast
    print("Step 2: Generating RA price forecast (2026-2030)...")
    forecast_data = forecaster.generate_forecast(include_uncertainty=True)
    print(f"  ✓ Generated {len(forecast_data)} forecast data points")
    print()

    # Generate annual summary
    print("Step 3: Computing annual summary statistics...")
    annual_summary = forecaster.get_annual_summary()
    print(f"  ✓ Summary generated for {len(annual_summary)} zone-year combinations")
    print()

    # Display key forecast results
    print("=" * 80)
    print("FORECAST SUMMARY")
    print("=" * 80)
    print()

    for zone in forecaster.config['zones']:
        zone_summary = annual_summary[annual_summary['zone'] == zone]
        print(f"\n{zone}:")
        print(f"{'Year':<6} {'Avg Price':<12} {'Min Price':<12} {'Max Price':<12} {'Std Dev':<10}")
        print("-" * 60)
        for _, row in zone_summary.iterrows():
            print(f"{int(row['year']):<6} "
                  f"${row['avg_price']:<11.2f} "
                  f"${row['min_price']:<11.2f} "
                  f"${row['max_price']:<11.2f} "
                  f"${row['std_dev']:<9.2f}")

    print()
    print("=" * 80)
    print()

    # Save forecast data
    print("Step 4: Saving forecast data...")
    forecast_file = forecaster.save_forecast(output_dir='forecasts')
    print(f"  ✓ Forecast data saved to: {forecast_file}")

    # Save metadata
    metadata_file = forecaster.save_metadata(output_dir='forecasts')
    print(f"  ✓ Metadata saved to: {metadata_file}")
    print()

    # Generate visualizations
    print("Step 5: Generating visualizations...")
    visualizer = CAISORAVisualizer(forecast_data, historical_data)

    plots = visualizer.generate_all_plots(output_dir='plots')

    print("  ✓ Generated visualizations:")
    for plot_name, plot_path in plots.items():
        print(f"    - {plot_name}: {plot_path}")
    print()

    # Summary statistics by year (system-wide average)
    print("=" * 80)
    print("SYSTEM-WIDE AVERAGE PRICES BY YEAR")
    print("=" * 80)
    print()

    yearly_avg = annual_summary.groupby('year')['avg_price'].mean()
    print(f"{'Year':<6} {'System Avg Price':<20} {'YoY Change':<15}")
    print("-" * 45)

    prev_price = None
    for year, avg_price in yearly_avg.items():
        yoy_change = ""
        if prev_price is not None:
            change_pct = ((avg_price - prev_price) / prev_price) * 100
            yoy_change = f"+{change_pct:.1f}%" if change_pct > 0 else f"{change_pct:.1f}%"

        print(f"{int(year):<6} ${avg_price:<19.2f} {yoy_change:<15}")
        prev_price = avg_price

    print()
    print("=" * 80)
    print()

    # Final summary
    print("✓ CAISO RA forecast generation complete!")
    print()
    print("Output files:")
    print(f"  Forecasts: forecasts/")
    print(f"  Plots: plots/")
    print()
    print("Next steps:")
    print("  1. Review the forecast data CSV files")
    print("  2. Examine the visualization plots")
    print("  3. Check the metadata file for methodology details")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error during forecast generation: {e}", exc_info=True)
        sys.exit(1)
