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


def train_model(config_path: str = "config.yaml"):
    """
    Train the forecasting model

    Args:
        config_path: Path to configuration file
    """
    logger.info("=" * 80)
    logger.info("STARTING MODEL TRAINING")
    logger.info("=" * 80)

    # Step 1: Collect data
    logger.info("\n[1/6] Collecting ERCOT Day-Ahead price data...")
    collector = ERCOTDataCollector()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years

    df = collector.fetch_and_save(start_date=start_date, end_date=end_date)

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
    save_plot: bool = True
):
    """
    Generate Day-Ahead forecast

    Args:
        forecast_hours: Number of hours to forecast
        config_path: Path to configuration file
        save_plot: Whether to save forecast plot
    """
    logger.info("=" * 80)
    logger.info(f"GENERATING {forecast_hours}-HOUR DAY-AHEAD FORECAST")
    logger.info("=" * 80)

    # Initialize forecaster
    forecaster = DayAheadForecaster(config_path)

    # Generate forecast
    logger.info("\nGenerating forecast...")
    forecast = forecaster.generate_forecast(forecast_hours=forecast_hours)

    # Save forecast
    filepath = forecaster.save_forecast(forecast)

    # Create visualization
    if save_plot:
        logger.info("\nCreating forecast visualization...")
        visualizer = ForecastVisualizer()

        # Load historical data for context
        historical_df = forecaster.load_historical_data()

        visualizer.plot_forecast(
            forecast_df=forecast,
            historical_df=historical_df,
            title=f"ERCOT Day-Ahead Price Forecast ({forecast_hours} Hours)"
        )

    logger.info("\n" + "=" * 80)
    logger.info("FORECAST GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nForecast Summary:")
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
  # Train the model
  python main.py --train

  # Generate 24-hour forecast
  python main.py --forecast

  # Generate 48-hour forecast
  python main.py --forecast --hours 48

  # Train and then forecast
  python main.py --train --forecast
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
"""Teams Chat Summarization Tool

This script fetches messages from a Microsoft Teams chat and generates
a summary using Claude API.
"""
import argparse
import sys
from datetime import datetime
from config import Config
from teams_client import TeamsClient
from summarizer import ChatSummarizer


def save_output(content: str, filename: str):
    """Save content to a file.

    Args:
        content: Content to save
        filename: Output filename
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Saved output to: {filename}")
    except Exception as e:
        print(f"✗ Error saving file: {e}")


def main():
    """Main entry point for the chat summarization tool."""
    parser = argparse.ArgumentParser(
        description="Summarize Microsoft Teams chat using Claude API"
    )
    parser.add_argument(
        "--chat-name",
        type=str,
        help="Name of the Teams chat to summarize (default: from .env)"
    )
    parser.add_argument(
        "--chat-id",
        type=str,
        help="Specific Teams chat ID (optional, will search by name if not provided)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days of chat history to fetch (default: 7)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of messages to fetch (default: 50)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Claude model to use (default: claude-sonnet-4-20250514)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save summary to file (optional)"
    )
    parser.add_argument(
        "--save-messages",
        type=str,
        help="Save raw messages to file (optional)"
    )
    parser.add_argument(
        "--custom-prompt",
        type=str,
        help="Custom instructions for summarization (optional)"
    )

    args = parser.parse_args()

    # If no action specified, show help
    if not args.train and not args.forecast:
        parser.print_help()
        sys.exit(0)

    try:
        # Train model
        if args.train:
            train_model(config_path=args.config)

        # Generate forecast
        if args.forecast:
            generate_forecast(
                forecast_hours=args.hours,
                config_path=args.config,
                save_plot=not args.no_plot
            )

    except FileNotFoundError as e:
        logger.error(f"\n{str(e)}\n")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
    print("=" * 60)
    print("Teams Chat Summarization Tool")
    print("=" * 60)
    print()

    try:
        # Validate configuration
        Config.validate()

        # Initialize clients
        print("Initializing...")
        teams_client = TeamsClient()
        summarizer = ChatSummarizer()
        print()

        # Fetch chat content
        print("Fetching chat messages...")
        chat_content = teams_client.get_chat_summary_content(
            chat_name=args.chat_name,
            chat_id=args.chat_id,
            days_back=args.days,
            limit=args.limit
        )
        print()

        # Save raw messages if requested
        if args.save_messages:
            save_output(chat_content, args.save_messages)
            print()

        # Generate summary
        print("Generating summary...")
        if args.custom_prompt:
            summary = summarizer.summarize_with_custom_prompt(
                chat_content=chat_content,
                custom_instructions=args.custom_prompt,
                model=args.model
            )
        else:
            summary = summarizer.summarize(
                chat_content=chat_content,
                model=args.model
            )
        print()

        # Output summary
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print()
        print(summary)
        print()
        print("=" * 60)

        # Save summary if requested
        if args.output:
            # Add metadata
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            output_content = f"""Teams Chat Summary
Generated: {timestamp}
Chat: {args.chat_name or Config.TEAMS_CHAT_NAME}
Period: Last {args.days} days
Model: {args.model}

{'=' * 60}

{summary}
"""
            save_output(output_content, args.output)

        print()
        print("✓ Summarization complete!")

    except ValueError as e:
        print(f"\n✗ Configuration Error: {e}")
        print("\nPlease ensure you have:")
        print("  1. Created a .env file (copy from .env.example)")
        print("  2. Set all required credentials")
        print("  3. See README.md for setup instructions")
        sys.exit(1)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
