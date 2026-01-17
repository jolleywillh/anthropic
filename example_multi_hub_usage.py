#!/usr/bin/env python3
"""
Example Usage: Multi-Hub ERCOT Day-Ahead Price Forecasting
Demonstrates how to use the forecasting system for multiple ERCOT hubs
"""

import sys
sys.path.insert(0, 'src')

from datetime import datetime, timedelta
import pandas as pd
from data_collector import ERCOTDataCollector
from forecaster import DayAheadForecaster
from feature_engineering import FeatureEngineer

def example_1_fetch_data_for_multiple_hubs():
    """
    Example 1: Fetch historical data for multiple ERCOT hubs
    """
    print("=" * 80)
    print("EXAMPLE 1: Fetch Data for Multiple Hubs")
    print("=" * 80)

    collector = ERCOTDataCollector()

    # Define the hubs we want to fetch
    hubs = ['HB_HOUSTON', 'HB_NORTH', 'HB_SOUTH', 'HB_WEST']

    # Fetch last 30 days of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    # Fetch data for all hubs
    df_all = collector.fetch_dam_prices_multi_hub(
        hubs=hubs,
        start_date=start_date,
        end_date=end_date
    )

    print(f"\nFetched {len(df_all)} total records across all hubs")
    print("\nRecords per hub:")
    print(df_all['settlement_point'].value_counts())

    print("\nSample data:")
    print(df_all.head(10))

    return df_all


def example_2_forecast_single_hub():
    """
    Example 2: Generate forecast for a single hub (Houston)
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Generate Forecast for Houston Hub")
    print("=" * 80)

    # Initialize forecaster for Houston hub
    forecaster = DayAheadForecaster(hub='HB_HOUSTON')

    # Generate 24-hour forecast
    forecast = forecaster.generate_forecast(forecast_hours=24)

    print("\nHouston Hub 24-Hour Forecast:")
    print(forecast[['datetime', 'hub', 'forecasted_price']].head(10))

    # Save the forecast
    filepath = forecaster.save_forecast(forecast)
    print(f"\nForecast saved to: {filepath}")

    return forecast


def example_3_forecast_all_hubs():
    """
    Example 3: Generate forecasts for all major ERCOT hubs
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Generate Forecasts for All Hubs")
    print("=" * 80)

    hubs = ['HB_HOUSTON', 'HB_NORTH', 'HB_SOUTH', 'HB_WEST']
    all_forecasts = []

    for hub in hubs:
        print(f"\nGenerating forecast for {hub}...")

        # Initialize forecaster for this hub
        forecaster = DayAheadForecaster(hub=hub)

        # Generate 24-hour forecast
        forecast = forecaster.generate_forecast(forecast_hours=24, include_confidence=True)

        # Add to collection
        all_forecasts.append(forecast)

        print(f"  Mean price: ${forecast['forecasted_price'].mean():.2f}/MWh")
        print(f"  Peak price: ${forecast['forecasted_price'].max():.2f}/MWh")

    # Combine all forecasts
    combined_forecasts = pd.concat(all_forecasts, ignore_index=True)

    print("\n" + "=" * 80)
    print("All Hub Forecasts Generated Successfully")
    print("=" * 80)

    # Summary comparison
    print("\nPrice Comparison Across Hubs:")
    summary = combined_forecasts.groupby('hub')['forecasted_price'].agg(['mean', 'min', 'max', 'std'])
    summary.columns = ['Mean ($/MWh)', 'Min ($/MWh)', 'Max ($/MWh)', 'Std Dev']
    print(summary)

    return combined_forecasts


def example_4_compare_hub_prices():
    """
    Example 4: Compare historical prices across hubs
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Compare Historical Prices Across Hubs")
    print("=" * 80)

    collector = ERCOTDataCollector()
    hubs = ['HB_HOUSTON', 'HB_NORTH', 'HB_SOUTH', 'HB_WEST']

    # Fetch last 7 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    df_all = collector.fetch_dam_prices_multi_hub(
        hubs=hubs,
        start_date=start_date,
        end_date=end_date
    )

    # Calculate statistics per hub
    print("\nHistorical Price Statistics (Last 7 Days):")
    stats = df_all.groupby('settlement_point')['price'].agg(['mean', 'min', 'max', 'std'])
    stats.columns = ['Mean ($/MWh)', 'Min ($/MWh)', 'Max ($/MWh)', 'Std Dev']
    print(stats)

    # Find most volatile hub
    most_volatile = stats['Std Dev'].idxmax()
    print(f"\nMost volatile hub: {most_volatile} (Std Dev: ${stats.loc[most_volatile, 'Std Dev']:.2f})")

    # Find highest average price hub
    highest_avg = stats['Mean ($/MWh)'].idxmax()
    print(f"Highest average price hub: {highest_avg} (Mean: ${stats.loc[highest_avg, 'Mean ($/MWh)']:.2f}/MWh)")

    return df_all, stats


def example_5_feature_engineering_with_hubs():
    """
    Example 5: Create features for multi-hub data
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Feature Engineering with Hub Information")
    print("=" * 80)

    collector = ERCOTDataCollector()

    # Fetch data for Houston hub
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    df = collector.fetch_dam_prices(
        start_date=start_date,
        end_date=end_date,
        settlement_point='HB_HOUSTON'
    )

    print(f"\nFetched {len(df)} records for Houston hub")

    # Create features
    engineer = FeatureEngineer()
    df_features = engineer.create_features(df)

    print(f"\nCreated {len(df_features.columns)} features")

    # Show hub-related features
    hub_features = [col for col in df_features.columns if 'hub_' in col.lower()]
    print(f"\nHub-related features ({len(hub_features)}):")
    for feature in hub_features:
        print(f"  - {feature}: {df_features[feature].iloc[0]}")

    print(f"\nTotal features created: {len(df_features.columns)}")

    return df_features


def example_6_48hour_forecast_west_hub():
    """
    Example 6: Generate 48-hour forecast for West hub
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Generate 48-Hour Forecast for West Hub")
    print("=" * 80)

    # Initialize forecaster for West hub
    forecaster = DayAheadForecaster(hub='HB_WEST')

    # Generate 48-hour forecast with confidence intervals
    forecast = forecaster.generate_forecast(
        forecast_hours=48,
        include_confidence=True
    )

    print("\nWest Hub 48-Hour Forecast Summary:")
    print(f"  Total hours: {len(forecast)}")
    print(f"  Mean price: ${forecast['forecasted_price'].mean():.2f}/MWh")
    print(f"  Min price:  ${forecast['forecasted_price'].min():.2f}/MWh")
    print(f"  Max price:  ${forecast['forecasted_price'].max():.2f}/MWh")

    # Show first 12 hours
    print("\nFirst 12 Hours:")
    print(forecast[['datetime', 'hub', 'forecasted_price', 'lower_bound', 'upper_bound']].head(12).to_string(index=False))

    # Find peak price hour
    peak_idx = forecast['forecasted_price'].idxmax()
    peak_row = forecast.loc[peak_idx]
    print(f"\nPeak price hour: {peak_row['datetime']}")
    print(f"  Price: ${peak_row['forecasted_price']:.2f}/MWh")
    print(f"  Confidence interval: ${peak_row['lower_bound']:.2f} - ${peak_row['upper_bound']:.2f}")

    # Save forecast
    filepath = forecaster.save_forecast(forecast)
    print(f"\nForecast saved to: {filepath}")

    return forecast


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("ERCOT MULTI-HUB FORECASTING EXAMPLES")
    print("=" * 80)
    print("\nThis script demonstrates various use cases for multi-hub forecasting")
    print("Note: Examples use synthetic data for demonstration purposes")
    print("=" * 80)

    try:
        # Run examples
        example_1_fetch_data_for_multiple_hubs()

        # Note: Examples 2-6 require trained models
        # Uncomment these after training models for each hub

        # example_2_forecast_single_hub()
        # example_3_forecast_all_hubs()
        # example_4_compare_hub_prices()
        # example_5_feature_engineering_with_hubs()
        # example_6_48hour_forecast_west_hub()

        print("\n" + "=" * 80)
        print("EXAMPLES COMPLETE")
        print("=" * 80)
        print("\nTo run forecasting examples (2-6), first train models for each hub:")
        print("  python main.py --train --hub HB_HOUSTON")
        print("  python main.py --train --hub HB_NORTH")
        print("  python main.py --train --hub HB_SOUTH")
        print("  python main.py --train --hub HB_WEST")
        print("\nOr train all hubs at once:")
        print("  python main.py --train --all-hubs")

    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
