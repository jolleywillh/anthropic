"""
Visualization Module
Creates plots and visualizations for forecasts and model performance
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
from typing import Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class ForecastVisualizer:
    """Creates visualizations for price forecasts and model evaluation"""

    def __init__(self, output_dir: str = "plots"):
        """
        Initialize visualizer

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_forecast(
        self,
        forecast_df: pd.DataFrame,
        historical_df: Optional[pd.DataFrame] = None,
        title: str = "ERCOT Day-Ahead Price Forecast",
        save_path: Optional[str] = None
    ):
        """
        Plot price forecast with confidence intervals

        Args:
            forecast_df: Forecast DataFrame with datetime, forecasted_price columns
            historical_df: Historical data for context (optional)
            title: Plot title
            save_path: Path to save plot (optional)
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot historical data if provided
        if historical_df is not None:
            # Show last 7 days of historical data
            last_week = historical_df['datetime'].max() - pd.Timedelta(days=7)
            hist_recent = historical_df[historical_df['datetime'] >= last_week]

            ax.plot(
                hist_recent['datetime'],
                hist_recent['price'],
                label='Historical',
                color='#2E86AB',
                linewidth=2,
                alpha=0.8
            )

        # Plot forecast
        ax.plot(
            forecast_df['datetime'],
            forecast_df['forecasted_price'],
            label='Forecast',
            color='#A23B72',
            linewidth=2.5,
            marker='o',
            markersize=5
        )

        # Plot confidence intervals if available
        if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
            ax.fill_between(
                forecast_df['datetime'],
                forecast_df['lower_bound'],
                forecast_df['upper_bound'],
                alpha=0.3,
                color='#A23B72',
                label='90% Confidence Interval'
            )

        ax.set_xlabel('Date/Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price ($/MWh)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Format x-axis
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Forecast plot saved to {save_path}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f'forecast_{timestamp}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Forecast plot saved to {save_path}")

        plt.close()

    def plot_actual_vs_predicted(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        title: str = "Actual vs Predicted Prices",
        save_path: Optional[str] = None
    ):
        """
        Plot actual vs predicted prices

        Args:
            y_true: True prices
            y_pred: Predicted prices
            title: Plot title
            save_path: Path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Scatter plot
        ax1.scatter(y_true, y_pred, alpha=0.5, s=20, color='#2E86AB')
        ax1.plot([y_true.min(), y_true.max()],
                [y_true.min(), y_true.max()],
                'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Price ($/MWh)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Predicted Price ($/MWh)', fontsize=12, fontweight='bold')
        ax1.set_title('Scatter Plot', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Time series plot
        if isinstance(y_true.index, pd.DatetimeIndex):
            # Show last 30 days for clarity
            last_30_days = y_true.index.max() - pd.Timedelta(days=30)
            mask = y_true.index >= last_30_days

            ax2.plot(y_true.index[mask], y_true[mask],
                    label='Actual', color='#2E86AB', linewidth=2, alpha=0.8)
            ax2.plot(y_true.index[mask], y_pred[mask],
                    label='Predicted', color='#A23B72', linewidth=2, alpha=0.8)
        else:
            ax2.plot(y_true.values[:500], label='Actual', color='#2E86AB', linewidth=1.5, alpha=0.8)
            ax2.plot(y_pred[:500], label='Predicted', color='#A23B72', linewidth=1.5, alpha=0.8)

        ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Price ($/MWh)', fontsize=12, fontweight='bold')
        ax2.set_title('Time Series (Last 30 Days)', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')

        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'actual_vs_predicted.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Actual vs predicted plot saved to {save_path}")
        plt.close()

    def plot_residuals(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot residual analysis

        Args:
            y_true: True prices
            y_pred: Predicted prices
            save_path: Path to save plot
        """
        residuals = y_true - y_pred

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=20, color='#2E86AB')
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Predicted Price ($/MWh)', fontweight='bold')
        axes[0, 0].set_ylabel('Residuals ($/MWh)', fontweight='bold')
        axes[0, 0].set_title('Residuals vs Predicted', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Residuals ($/MWh)', fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontweight='bold')
        axes[0, 1].set_title('Distribution of Residuals', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # Residuals over time
        if isinstance(y_true.index, pd.DatetimeIndex):
            axes[1, 1].plot(y_true.index, residuals, alpha=0.6, linewidth=1, color='#2E86AB')
        else:
            axes[1, 1].plot(residuals, alpha=0.6, linewidth=1, color='#2E86AB')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Time', fontweight='bold')
        axes[1, 1].set_ylabel('Residuals ($/MWh)', fontweight='bold')
        axes[1, 1].set_title('Residuals Over Time', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('Residual Analysis', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'residual_analysis.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Residual analysis plot saved to {save_path}")
        plt.close()

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        title: str = "Feature Importance",
        top_n: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Plot feature importance

        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            title: Plot title
            top_n: Number of top features to show
            save_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        top_features = importance_df.head(top_n).sort_values('importance', ascending=True)

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))

        ax.barh(range(len(top_features)), top_features['importance'], color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
        ax.set_title(f'{title} (Top {top_n})', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'feature_importance.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
        plt.close()

    def plot_price_distribution(
        self,
        historical_df: pd.DataFrame,
        forecast_df: Optional[pd.DataFrame] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot price distribution comparison

        Args:
            historical_df: Historical data
            forecast_df: Forecast data (optional)
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Historical distribution
        axes[0].hist(historical_df['price'], bins=50, color='#2E86AB',
                    alpha=0.7, edgecolor='black', density=True)
        axes[0].set_xlabel('Price ($/MWh)', fontweight='bold')
        axes[0].set_ylabel('Density', fontweight='bold')
        axes[0].set_title('Historical Price Distribution', fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Add statistics
        mean_hist = historical_df['price'].mean()
        median_hist = historical_df['price'].median()
        axes[0].axvline(mean_hist, color='r', linestyle='--', linewidth=2, label=f'Mean: ${mean_hist:.2f}')
        axes[0].axvline(median_hist, color='g', linestyle='--', linewidth=2, label=f'Median: ${median_hist:.2f}')
        axes[0].legend()

        # Comparison with forecast if available
        if forecast_df is not None:
            axes[1].hist(historical_df['price'], bins=50, color='#2E86AB',
                        alpha=0.5, edgecolor='black', density=True, label='Historical')
            axes[1].hist(forecast_df['forecasted_price'], bins=30, color='#A23B72',
                        alpha=0.5, edgecolor='black', density=True, label='Forecast')
            axes[1].set_xlabel('Price ($/MWh)', fontweight='bold')
            axes[1].set_ylabel('Density', fontweight='bold')
            axes[1].set_title('Historical vs Forecast Distribution', fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            # Box plot
            axes[1].boxplot(historical_df['price'], vert=True,
                           patch_artist=True,
                           boxprops=dict(facecolor='#2E86AB', alpha=0.7),
                           medianprops=dict(color='red', linewidth=2))
            axes[1].set_ylabel('Price ($/MWh)', fontweight='bold')
            axes[1].set_title('Price Distribution (Box Plot)', fontweight='bold')
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'price_distribution.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Price distribution plot saved to {save_path}")
        plt.close()

    def create_evaluation_report(
        self,
        metrics: dict,
        y_true: pd.Series,
        y_pred: np.ndarray,
        feature_importance: pd.DataFrame,
        save_dir: Optional[str] = None
    ):
        """
        Create comprehensive evaluation report with multiple plots

        Args:
            metrics: Dictionary of metrics
            y_true: True prices
            y_pred: Predicted prices
            feature_importance: Feature importance DataFrame
            save_dir: Directory to save plots
        """
        if save_dir is None:
            save_dir = self.output_dir

        os.makedirs(save_dir, exist_ok=True)

        logger.info("Creating comprehensive evaluation report")

        # Actual vs Predicted
        self.plot_actual_vs_predicted(
            y_true, y_pred,
            save_path=os.path.join(save_dir, 'actual_vs_predicted.png')
        )

        # Residual Analysis
        self.plot_residuals(
            y_true, y_pred,
            save_path=os.path.join(save_dir, 'residual_analysis.png')
        )

        # Feature Importance
        self.plot_feature_importance(
            feature_importance,
            save_path=os.path.join(save_dir, 'feature_importance.png')
        )

        logger.info(f"Evaluation report created in {save_dir}")


if __name__ == "__main__":
    # Example usage
    from data_collector import ERCOTDataCollector

    collector = ERCOTDataCollector()
    df = collector.load_data()

    if not df.empty:
        visualizer = ForecastVisualizer()
        visualizer.plot_price_distribution(df)
        print("Visualization examples created in plots/")
