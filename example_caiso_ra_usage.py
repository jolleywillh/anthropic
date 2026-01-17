#!/usr/bin/env python3
"""
Example usage of CAISO RA Forecasting API

This script demonstrates how to use the CAISO RA forecasting module
programmatically for custom analyses and integrations.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from caiso_ra_forecaster import CAISORAForecaster
from caiso_ra_visualization import CAISORAVisualizer
import pandas as pd


def example_1_basic_forecast():
    """Example 1: Generate a basic forecast."""
    print("\n" + "=" * 80)
    print("Example 1: Basic Forecast Generation")
    print("=" * 80 + "\n")

    # Initialize forecaster
    forecaster = CAISORAForecaster()

    # Generate forecast
    forecast = forecaster.generate_forecast()

    # Display first few rows
    print("First 10 forecast records:")
    print(forecast.head(10))

    # Save to CSV
    output_file = forecaster.save_forecast()
    print(f"\n✓ Forecast saved to: {output_file}")


def example_2_custom_configuration():
    """Example 2: Forecast with custom configuration."""
    print("\n" + "=" * 80)
    print("Example 2: Custom Configuration")
    print("=" * 80 + "\n")

    # Custom configuration
    custom_config = {
        'base_year': 2025,
        'forecast_years': [2026, 2027, 2028],  # Only 3 years
        'annual_load_growth': 0.02,  # Higher load growth (2%)
        'renewable_penetration_impact': 0.03,  # Higher renewable impact (3%)
        'retirement_impact': 0.025,  # Higher retirement impact (2.5%)
        'battery_storage_impact': -0.02,  # Stronger battery impact (-2%)
        'zones': ['System', 'LA Basin']  # Only two zones
    }

    # Initialize with custom config
    forecaster = CAISORAForecaster(config=custom_config)

    # Generate forecast
    forecast = forecaster.generate_forecast()

    print("Custom forecast configuration:")
    print(f"  Years: {custom_config['forecast_years']}")
    print(f"  Zones: {custom_config['zones']}")
    print(f"  Net growth rate: {(custom_config['annual_load_growth'] + custom_config['renewable_penetration_impact'] + custom_config['retirement_impact'] + custom_config['battery_storage_impact']):.1%}")

    # Get annual summary
    summary = forecaster.get_annual_summary()
    print("\nAnnual Summary:")
    print(summary)


def example_3_zone_specific_analysis():
    """Example 3: Analyze a specific zone."""
    print("\n" + "=" * 80)
    print("Example 3: Zone-Specific Analysis")
    print("=" * 80 + "\n")

    # Generate forecast
    forecaster = CAISORAForecaster()
    forecast = forecaster.generate_forecast()

    # Analyze LA Basin
    zone = 'LA Basin'
    la_basin = forecast[forecast['zone'] == zone]

    print(f"Analysis for {zone}:")
    print(f"  Total forecast records: {len(la_basin)}")

    # Calculate statistics per year
    print(f"\n  Yearly Statistics:")
    for year in sorted(la_basin['year'].unique()):
        year_data = la_basin[la_basin['year'] == year]
        print(f"    {year}:")
        print(f"      Average: ${year_data['ra_price'].mean():.2f}/kW-month")
        print(f"      Min: ${year_data['ra_price'].min():.2f}/kW-month")
        print(f"      Max: ${year_data['ra_price'].max():.2f}/kW-month")
        print(f"      Summer Avg (Jun-Sep): ${year_data[year_data['month_num'].isin([6, 7, 8, 9])]['ra_price'].mean():.2f}/kW-month")


def example_4_visualizations():
    """Example 4: Generate custom visualizations."""
    print("\n" + "=" * 80)
    print("Example 4: Custom Visualizations")
    print("=" * 80 + "\n")

    # Generate forecast and historical data
    forecaster = CAISORAForecaster()
    historical = forecaster.generate_historical_baseline()
    forecast = forecaster.generate_forecast()

    # Create visualizer
    visualizer = CAISORAVisualizer(forecast, historical)

    # Generate individual plots
    print("Generating visualizations...")

    plot1 = visualizer.plot_multi_year_trends(output_dir='plots')
    print(f"  ✓ Multi-year trends: {plot1}")

    plot2 = visualizer.plot_zone_comparison(output_dir='plots')
    print(f"  ✓ Zone comparison: {plot2}")

    plot3 = visualizer.plot_seasonal_patterns(output_dir='plots')
    print(f"  ✓ Seasonal patterns: {plot3}")

    plot4 = visualizer.plot_price_heatmap(output_dir='plots')
    print(f"  ✓ Price heatmap: {plot4}")

    plot5 = visualizer.plot_growth_trends(output_dir='plots')
    print(f"  ✓ Growth trends: {plot5}")


def example_5_monthly_analysis():
    """Example 5: Analyze specific months across years."""
    print("\n" + "=" * 80)
    print("Example 5: Monthly Analysis Across Years")
    print("=" * 80 + "\n")

    # Generate forecast
    forecaster = CAISORAForecaster()
    forecast = forecaster.generate_forecast()

    # Analyze summer months (June, July, August)
    summer_months = [6, 7, 8]
    print("Summer Peak RA Prices (Jun-Aug):")
    print(f"{'Zone':<25} {'2026':<10} {'2027':<10} {'2028':<10} {'2029':<10} {'2030':<10}")
    print("-" * 80)

    for zone in forecast['zone'].unique():
        zone_data = forecast[
            (forecast['zone'] == zone) &
            (forecast['month_num'].isin(summer_months))
        ]

        row = [zone]
        for year in sorted(forecast['year'].unique()):
            year_data = zone_data[zone_data['year'] == year]
            avg_price = year_data['ra_price'].mean()
            row.append(f"${avg_price:.2f}")

        print(f"{row[0]:<25} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<10} {row[5]:<10}")


def example_6_export_specific_zone_year():
    """Example 6: Export specific zone and year data."""
    print("\n" + "=" * 80)
    print("Example 6: Export Specific Zone/Year Data")
    print("=" * 80 + "\n")

    # Generate forecast
    forecaster = CAISORAForecaster()
    forecast = forecaster.generate_forecast()

    # Export LA Basin 2028 data
    zone = 'LA Basin'
    year = 2028

    filtered_data = forecast[
        (forecast['zone'] == zone) &
        (forecast['year'] == year)
    ][['month', 'ra_price', 'lower_bound', 'upper_bound']]

    # Save to custom CSV
    output_file = f'forecasts/{zone.replace("/", "_").replace(" ", "_")}_{year}.csv'
    os.makedirs('forecasts', exist_ok=True)
    filtered_data.to_csv(output_file, index=False)

    print(f"Exported {zone} {year} data:")
    print(filtered_data)
    print(f"\n✓ Saved to: {output_file}")


def example_7_calculate_total_cost():
    """Example 7: Calculate total RA procurement cost."""
    print("\n" + "=" * 80)
    print("Example 7: Calculate Total RA Procurement Cost")
    print("=" * 80 + "\n")

    # Generate forecast
    forecaster = CAISORAForecaster()
    forecast = forecaster.generate_forecast()

    # Assume a load-serving entity needs to procure capacity
    capacity_mw = 500  # 500 MW capacity requirement

    zone = 'LA Basin'
    year = 2026

    zone_year_data = forecast[
        (forecast['zone'] == zone) &
        (forecast['year'] == year)
    ]

    # Calculate annual cost
    monthly_costs = []
    print(f"Estimated RA Procurement Cost for {capacity_mw} MW in {zone} ({year}):")
    print(f"\n{'Month':<10} {'RA Price':<15} {'Monthly Cost':<20}")
    print("-" * 50)

    for _, row in zone_year_data.iterrows():
        # Convert $/kW-month to total monthly cost
        monthly_cost = row['ra_price'] * capacity_mw * 1000  # Convert MW to kW
        monthly_costs.append(monthly_cost)
        print(f"{row['month']:<10} ${row['ra_price']:<14.2f} ${monthly_cost:>18,.2f}")

    total_annual_cost = sum(monthly_costs)
    print("-" * 50)
    print(f"{'TOTAL':<10} {'':15} ${total_annual_cost:>18,.2f}")
    print(f"\nAverage monthly cost: ${total_annual_cost/12:,.2f}")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("CAISO RA Forecasting - Example Usage")
    print("=" * 80)

    # Run examples
    example_1_basic_forecast()
    example_2_custom_configuration()
    example_3_zone_specific_analysis()
    example_4_visualizations()
    example_5_monthly_analysis()
    example_6_export_specific_zone_year()
    example_7_calculate_total_cost()

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
