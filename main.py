#!/usr/bin/env python3
"""
ERCOT Day-Ahead Price Forecasting System
Main script for training models and generating forecasts
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
import pandas as pd

# Add src to path
sys.path.insert(0, 'src')

from data_collector import ERCOTDataCollector
from feature_engineering import FeatureEngineer
from models import PriceForecastModel
from forecaster import DayAheadForecaster
from visualization import ForecastVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(config_path: str = "config.yaml", hub: str = "HB_HUBAVG"):
    """
    Train the forecasting model

    Args:
        config_path: Path to configuration file
        hub: ERCOT hub to train model for
    """
    logger.info("=" * 80)
    logger.info(f"STARTING MODEL TRAINING FOR HUB: {hub}")
    logger.info("=" * 80)

    # Step 1: Collect data
    logger.info(f"\n[1/6] Collecting ERCOT Day-Ahead price data for {hub}...")
    collector = ERCOTDataCollector()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years

    hub_filename = f"ercot_dam_prices_{hub}.csv"
    df = collector.fetch_and_save(
        start_date=start_date,
        end_date=end_date,
        settlement_point=hub,
        filename=hub_filename
    )

    logger.info(f"Collected {len(df)} records from {df['datetime'].min()} to {df['datetime'].max()}")

    # Step 2: Feature engineering
    logger.info("\n[2/6] Engineering features...")
    engineer = FeatureEngineer()
    df_features = engineer.create_features(df)

    logger.info(f"Created {len(engineer.get_feature_names(df_features))} features")

    # Step 3: Split data
    logger.info("\n[3/6] Splitting data into train/validation/test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = engineer.prepare_train_test(
        df_features, test_size=0.2, validation_size=0.1
    )

    # Step 4: Train models
    logger.info("\n[4/6] Training ML models...")
    model = PriceForecastModel(config_path)
    model.train(X_train, y_train, X_val, y_val)

    # Step 5: Evaluate
    logger.info("\n[5/6] Evaluating model performance...")
    test_metrics = model.evaluate(X_test, y_test, dataset_name="test")

    # Step 6: Save model and create visualizations
    logger.info("\n[6/6] Saving model and creating visualizations...")
    model.save_models()

    # Create visualizations
    visualizer = ForecastVisualizer()

    # Get predictions for visualization
    y_pred = model.predict(X_test, use_ensemble=True)

    # Create evaluation report
    feature_importance = model.get_feature_importance(model_name='xgboost', top_n=20)
    visualizer.create_evaluation_report(
        metrics=test_metrics,
        y_true=y_test,
        y_pred=y_pred,
        feature_importance=feature_importance
    )

    # Plot price distribution
    visualizer.plot_price_distribution(df)

    logger.info("\n" + "=" * 80)
    logger.info("MODEL TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nEnsemble Model Performance (Test Set):")
    logger.info(f"  MAE:  ${test_metrics['ensemble']['mae']:.2f}")
    logger.info(f"  RMSE: ${test_metrics['ensemble']['rmse']:.2f}")
    logger.info(f"  MAPE: {test_metrics['ensemble']['mape']:.2f}%")
    logger.info(f"  R²:   {test_metrics['ensemble']['r2']:.4f}")
    logger.info(f"\nModel saved to: models/")
    logger.info(f"Plots saved to: plots/")


def generate_forecast(
    forecast_hours: int = 24,
    config_path: str = "config.yaml",
    save_plot: bool = True,
    hub: str = "HB_HUBAVG"
):
    """
    Generate Day-Ahead forecast

    Args:
        forecast_hours: Number of hours to forecast
        config_path: Path to configuration file
        save_plot: Whether to save forecast plot
        hub: ERCOT hub to forecast for
    """
    logger.info("=" * 80)
    logger.info(f"GENERATING {forecast_hours}-HOUR DAY-AHEAD FORECAST FOR {hub}")
    logger.info("=" * 80)

    # Initialize forecaster
    forecaster = DayAheadForecaster(config_path, hub=hub)

    # Generate forecast
    logger.info("\nGenerating forecast...")
    forecast = forecaster.generate_forecast(forecast_hours=forecast_hours, hub=hub)

    # Save forecast
    filepath = forecaster.save_forecast(forecast, hub=hub)

    # Create visualization
    if save_plot:
        logger.info("\nCreating forecast visualization...")
        visualizer = ForecastVisualizer()

        # Load historical data for context
        historical_df = forecaster.load_historical_data()

        visualizer.plot_forecast(
            forecast_df=forecast,
            historical_df=historical_df,
            title=f"ERCOT Day-Ahead Price Forecast - {hub} ({forecast_hours} Hours)"
        )

    logger.info("\n" + "=" * 80)
    logger.info("FORECAST GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nForecast Summary:")
    logger.info(f"  Hub:        {hub}")
    logger.info(f"  Time Range: {forecast['datetime'].min()} to {forecast['datetime'].max()}")
    logger.info(f"  Mean Price: ${forecast['forecasted_price'].mean():.2f}/MWh")
    logger.info(f"  Min Price:  ${forecast['forecasted_price'].min():.2f}/MWh")
    logger.info(f"  Max Price:  ${forecast['forecasted_price'].max():.2f}/MWh")
    logger.info(f"\nForecast saved to: {filepath}")
    if save_plot:
        logger.info(f"Plot saved to: plots/")

    # Display forecast table
    logger.info("\nForecast Details:")
    print("\n" + forecast.to_string(index=False))


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="ERCOT Day-Ahead Price Forecasting System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the model for Houston hub
  python main.py --train --hub HB_HOUSTON

  # Generate 24-hour forecast for North hub
  python main.py --forecast --hub HB_NORTH

  # Generate 48-hour forecast for South hub
  python main.py --forecast --hours 48 --hub HB_SOUTH

  # Train and then forecast for West hub
  python main.py --train --forecast --hub HB_WEST

  # Train models for all hubs
  python main.py --train --all-hubs

  # Generate forecasts for all hubs
  python main.py --forecast --all-hubs
        """
    )

    parser.add_argument(
        '--train',
        action='store_true',
        help='Train the forecasting model'
    )

    parser.add_argument(
        '--forecast',
        action='store_true',
        help='Generate Day-Ahead price forecast'
    )

    parser.add_argument(
        '--hours',
        type=int,
        default=24,
        help='Number of hours to forecast (default: 24)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )

    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Do not create visualization plots'
    )

    parser.add_argument(
        '--hub',
        type=str,
        default='HB_HUBAVG',
        help='ERCOT hub to train/forecast for (default: HB_HUBAVG). Options: HB_HOUSTON, HB_NORTH, HB_SOUTH, HB_WEST, HB_HUBAVG'
    )

    parser.add_argument(
        '--all-hubs',
        action='store_true',
        help='Train or forecast for all ERCOT hubs (Houston, North, South, West)'
    )

    args = parser.parse_args()

    # If no action specified, show help
    if not args.train and not args.forecast:
        parser.print_help()
        sys.exit(0)

    try:
        # Determine which hubs to process
        if args.all_hubs:
            hubs = ['HB_HOUSTON', 'HB_NORTH', 'HB_SOUTH', 'HB_WEST']
        else:
            hubs = [args.hub]

        # Process each hub
        for hub in hubs:
            # Train model
            if args.train:
                train_model(config_path=args.config, hub=hub)

            # Generate forecast
            if args.forecast:
                generate_forecast(
                    forecast_hours=args.hours,
                    config_path=args.config,
                    save_plot=not args.no_plot,
                    hub=hub
                )

            # Add separator between hubs if processing multiple
            if len(hubs) > 1 and hub != hubs[-1]:
                logger.info("\n" + "=" * 80 + "\n")

    except FileNotFoundError as e:
        logger.error(f"\n{str(e)}\n")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
