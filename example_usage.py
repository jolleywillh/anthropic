#!/usr/bin/env python3
"""
Example Usage of ERCOT Day-Ahead Price Forecasting System
This script demonstrates various ways to use the forecasting system
"""

import sys
sys.path.insert(0, 'src')

from datetime import datetime, timedelta
import pandas as pd

from data_collector import ERCOTDataCollector
from feature_engineering import FeatureEngineer
from models import PriceForecastModel
from forecaster import DayAheadForecaster
from visualization import ForecastVisualizer


def example_1_quick_forecast():
    """Example 1: Quick 24-hour forecast using existing model"""
    print("=" * 80)
    print("EXAMPLE 1: Quick 24-hour forecast")
    print("=" * 80)

    forecaster = DayAheadForecaster()
    forecast = forecaster.generate_forecast(forecast_hours=24)

    print("\nForecast Summary:")
    print(f"Mean Price: ${forecast['forecasted_price'].mean():.2f}/MWh")
    print(f"Peak Price: ${forecast['forecasted_price'].max():.2f}/MWh")
    print(f"Min Price:  ${forecast['forecasted_price'].min():.2f}/MWh")

    print("\nFirst 5 hours:")
    print(forecast.head())


def example_2_custom_training():
    """Example 2: Train model with custom parameters"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Custom model training")
    print("=" * 80)

    # Collect data
    collector = ERCOTDataCollector()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year
    df = collector.fetch_dam_prices(start_date=start_date, end_date=end_date)

    print(f"\nCollected {len(df)} records")

    # Engineer features
    engineer = FeatureEngineer(
        lag_features=[1, 24, 168],  # Custom lag features
        rolling_windows=[24, 168]    # Custom rolling windows
    )
    df_features = engineer.create_features(df)

    print(f"Created {len(engineer.get_feature_names(df_features))} features")

    # Train model
    X_train, X_val, X_test, y_train, y_val, y_test = engineer.prepare_train_test(
        df_features, test_size=0.2
    )

    model = PriceForecastModel()
    model.build_models()
    model.train(X_train, y_train, X_val, y_val)

    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    print(f"\nTest MAE: ${metrics['ensemble']['mae']:.2f}")


def example_3_visualization():
    """Example 3: Create custom visualizations"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Custom visualizations")
    print("=" * 80)

    # Generate forecast
    forecaster = DayAheadForecaster()
    forecast = forecaster.generate_forecast(forecast_hours=48)

    # Load historical data
    historical_df = forecaster.load_historical_data()

    # Create visualizations
    visualizer = ForecastVisualizer(output_dir="custom_plots")

    # Plot forecast
    visualizer.plot_forecast(
        forecast_df=forecast,
        historical_df=historical_df,
        title="Custom 48-Hour ERCOT Forecast"
    )

    # Plot price distribution
    visualizer.plot_price_distribution(historical_df, forecast)

    print("\nVisualizations saved to custom_plots/")


def example_4_feature_importance():
    """Example 4: Analyze feature importance"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Feature importance analysis")
    print("=" * 80)

    # Load trained model
    model = PriceForecastModel()
    model.load_models()

    # Get feature importance
    importance = model.get_feature_importance(model_name='xgboost', top_n=10)

    print("\nTop 10 Most Important Features:")
    print("-" * 50)
    for idx, row in importance.iterrows():
        print(f"{row['feature']:30s} {row['importance']:.4f}")

    # Visualize
    visualizer = ForecastVisualizer()
    visualizer.plot_feature_importance(importance, top_n=10)


def example_5_batch_forecasting():
    """Example 5: Generate forecasts for multiple horizons"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Batch forecasting")
    print("=" * 80)

    forecaster = DayAheadForecaster()

    horizons = [6, 12, 24, 48]
    results = {}

    for hours in horizons:
        forecast = forecaster.generate_forecast(forecast_hours=hours)
        results[hours] = {
            'mean': forecast['forecasted_price'].mean(),
            'max': forecast['forecasted_price'].max(),
            'min': forecast['forecasted_price'].min()
        }

    print("\nForecast Summary by Horizon:")
    print("-" * 70)
    print(f"{'Horizon':<10} {'Mean Price':<15} {'Max Price':<15} {'Min Price':<15}")
    print("-" * 70)
    for hours, stats in results.items():
        print(f"{hours}h{' '*6} ${stats['mean']:>6.2f}/MWh    "
              f"${stats['max']:>6.2f}/MWh    ${stats['min']:>6.2f}/MWh")


def example_6_api_usage():
    """Example 6: Using the system as an API"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: API-style usage")
    print("=" * 80)

    # Initialize forecaster
    forecaster = DayAheadForecaster()

    # Generate forecast
    forecast = forecaster.generate_forecast(
        forecast_start=datetime.now() + timedelta(hours=1),
        forecast_hours=24,
        include_confidence=True
    )

    # Convert to dictionary for API response
    forecast_dict = {
        'timestamp': datetime.now().isoformat(),
        'forecast_start': forecast['datetime'].min().isoformat(),
        'forecast_end': forecast['datetime'].max().isoformat(),
        'prices': [
            {
                'datetime': row['datetime'].isoformat(),
                'price': float(row['forecasted_price']),
                'confidence_interval': {
                    'lower': float(row['lower_bound']),
                    'upper': float(row['upper_bound'])
                }
            }
            for _, row in forecast.iterrows()
        ]
    }

    print("\nAPI Response Sample (first 3 hours):")
    print("-" * 70)
    for price_point in forecast_dict['prices'][:3]:
        print(f"Time: {price_point['datetime']}")
        print(f"  Price: ${price_point['price']:.2f}/MWh")
        print(f"  CI: [${price_point['confidence_interval']['lower']:.2f}, "
              f"${price_point['confidence_interval']['upper']:.2f}]")
        print()


def main():
    """Run all examples"""
    examples = [
        ("Quick Forecast", example_1_quick_forecast),
        ("Custom Training", example_2_custom_training),
        ("Visualization", example_3_visualization),
        ("Feature Importance", example_4_feature_importance),
        ("Batch Forecasting", example_5_batch_forecasting),
        ("API Usage", example_6_api_usage),
    ]

    print("\nERCOT Day-Ahead Price Forecasting - Example Usage")
    print("=" * 80)
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print("  0. Run all examples")

    try:
        choice = input("\nSelect example (0-6): ")
        choice = int(choice)

        if choice == 0:
            # Run all examples
            for name, example_func in examples:
                try:
                    example_func()
                except Exception as e:
                    print(f"\nError in {name}: {str(e)}")
        elif 1 <= choice <= len(examples):
            # Run selected example
            examples[choice - 1][1]()
        else:
            print("Invalid choice")

    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")

    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
