"""
ML Models Module
Implements XGBoost and Random Forest models for price forecasting
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import logging
from typing import Dict, Tuple, Optional
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PriceForecastModel:
    """Ensemble model for ERCOT Day-Ahead price forecasting"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the forecasting model

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.models = {}
        self.feature_importance = {}
        self.training_metrics = {}

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._default_config()

    def _default_config(self) -> dict:
        """Return default configuration"""
        return {
            'model': {
                'model_dir': 'models',
                'xgboost': {
                    'n_estimators': 200,
                    'max_depth': 7,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                },
                'random_forest': {
                    'n_estimators': 150,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                },
                'ensemble_weights': {
                    'xgboost': 0.6,
                    'random_forest': 0.4
                }
            }
        }

    def build_models(self):
        """Build XGBoost and Random Forest models"""
        logger.info("Building models")

        # XGBoost
        xgb_params = self.config['model']['xgboost']
        self.models['xgboost'] = xgb.XGBRegressor(
            n_estimators=xgb_params['n_estimators'],
            max_depth=xgb_params['max_depth'],
            learning_rate=xgb_params['learning_rate'],
            subsample=xgb_params['subsample'],
            colsample_bytree=xgb_params['colsample_bytree'],
            random_state=xgb_params['random_state'],
            objective='reg:squarederror',
            n_jobs=-1
        )

        # Random Forest
        rf_params = self.config['model']['random_forest']
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=rf_params['n_estimators'],
            max_depth=rf_params['max_depth'],
            min_samples_split=rf_params['min_samples_split'],
            min_samples_leaf=rf_params['min_samples_leaf'],
            random_state=rf_params['random_state'],
            n_jobs=-1
        )

        logger.info("Models built successfully")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ):
        """
        Train all models

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        """
        logger.info("Starting model training")

        if not self.models:
            self.build_models()

        # Train XGBoost
        logger.info("Training XGBoost model")
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.models['xgboost'].fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.models['xgboost'].fit(X_train, y_train)

        # Train Random Forest
        logger.info("Training Random Forest model")
        self.models['random_forest'].fit(X_train, y_train)

        # Store feature importance
        self._extract_feature_importance(X_train.columns)

        # Evaluate on training set
        logger.info("Evaluating on training set")
        train_metrics = self.evaluate(X_train, y_train, dataset_name="train")
        self.training_metrics['train'] = train_metrics

        # Evaluate on validation set
        if X_val is not None and y_val is not None:
            logger.info("Evaluating on validation set")
            val_metrics = self.evaluate(X_val, y_val, dataset_name="validation")
            self.training_metrics['validation'] = val_metrics

        logger.info("Training complete")

    def predict(
        self,
        X: pd.DataFrame,
        use_ensemble: bool = True
    ) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Features for prediction
            use_ensemble: If True, use weighted ensemble; if False, use only XGBoost

        Returns:
            Array of predictions
        """
        if use_ensemble:
            # Ensemble prediction
            weights = self.config['model']['ensemble_weights']

            pred_xgb = self.models['xgboost'].predict(X)
            pred_rf = self.models['random_forest'].predict(X)

            predictions = (
                weights['xgboost'] * pred_xgb +
                weights['random_forest'] * pred_rf
            )
        else:
            # Use only XGBoost
            predictions = self.models['xgboost'].predict(X)

        return predictions

    def evaluate(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        dataset_name: str = "test"
    ) -> Dict[str, float]:
        """
        Evaluate model performance

        Args:
            X: Features
            y_true: True values
            dataset_name: Name of dataset for logging

        Returns:
            Dictionary of metrics
        """
        # Individual model predictions
        pred_xgb = self.models['xgboost'].predict(X)
        pred_rf = self.models['random_forest'].predict(X)

        # Ensemble prediction
        pred_ensemble = self.predict(X, use_ensemble=True)

        # Calculate metrics
        metrics = {
            'xgboost': self._calculate_metrics(y_true, pred_xgb),
            'random_forest': self._calculate_metrics(y_true, pred_rf),
            'ensemble': self._calculate_metrics(y_true, pred_ensemble)
        }

        # Log metrics
        logger.info(f"\n{dataset_name.upper()} SET METRICS:")
        for model_name, model_metrics in metrics.items():
            logger.info(f"\n{model_name.upper()}:")
            logger.info(f"  MAE:  ${model_metrics['mae']:.2f}")
            logger.info(f"  RMSE: ${model_metrics['rmse']:.2f}")
            logger.info(f"  MAPE: {model_metrics['mape']:.2f}%")
            logger.info(f"  R²:   {model_metrics['r2']:.4f}")

        return metrics

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)

        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        }

    def _extract_feature_importance(self, feature_names):
        """Extract and store feature importance from models"""
        # XGBoost feature importance
        xgb_importance = self.models['xgboost'].feature_importances_
        self.feature_importance['xgboost'] = pd.DataFrame({
            'feature': feature_names,
            'importance': xgb_importance
        }).sort_values('importance', ascending=False)

        # Random Forest feature importance
        rf_importance = self.models['random_forest'].feature_importances_
        self.feature_importance['random_forest'] = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_importance
        }).sort_values('importance', ascending=False)

        # Log top features
        logger.info("\nTop 10 features (XGBoost):")
        for idx, row in self.feature_importance['xgboost'].head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    def get_feature_importance(self, model_name: str = 'xgboost', top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance

        Args:
            model_name: Name of model ('xgboost' or 'random_forest')
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if model_name in self.feature_importance:
            return self.feature_importance[model_name].head(top_n)
        else:
            return pd.DataFrame()

    def save_models(self, model_dir: Optional[str] = None):
        """
        Save trained models to disk

        Args:
            model_dir: Directory to save models
        """
        if model_dir is None:
            model_dir = self.config['model']['model_dir']

        os.makedirs(model_dir, exist_ok=True)

        # Save XGBoost
        xgb_path = os.path.join(model_dir, 'xgboost_model.json')
        self.models['xgboost'].save_model(xgb_path)
        logger.info(f"XGBoost model saved to {xgb_path}")

        # Save Random Forest
        rf_path = os.path.join(model_dir, 'random_forest_model.joblib')
        joblib.dump(self.models['random_forest'], rf_path)
        logger.info(f"Random Forest model saved to {rf_path}")

        # Save feature importance
        for model_name, importance_df in self.feature_importance.items():
            importance_path = os.path.join(model_dir, f'{model_name}_feature_importance.csv')
            importance_df.to_csv(importance_path, index=False)

        # Save training metrics
        metrics_path = os.path.join(model_dir, 'training_metrics.yaml')
        with open(metrics_path, 'w') as f:
            yaml.dump(self.training_metrics, f)

        logger.info(f"All models and metadata saved to {model_dir}")

    def load_models(self, model_dir: Optional[str] = None):
        """
        Load trained models from disk

        Args:
            model_dir: Directory containing saved models
        """
        if model_dir is None:
            model_dir = self.config['model']['model_dir']

        # Load XGBoost
        xgb_path = os.path.join(model_dir, 'xgboost_model.json')
        if os.path.exists(xgb_path):
            xgb_params = self.config['model']['xgboost']
            self.models['xgboost'] = xgb.XGBRegressor()
            self.models['xgboost'].load_model(xgb_path)
            logger.info(f"XGBoost model loaded from {xgb_path}")

        # Load Random Forest
        rf_path = os.path.join(model_dir, 'random_forest_model.joblib')
        if os.path.exists(rf_path):
            self.models['random_forest'] = joblib.load(rf_path)
            logger.info(f"Random Forest model loaded from {rf_path}")

        # Load feature importance if available
        for model_name in ['xgboost', 'random_forest']:
            importance_path = os.path.join(model_dir, f'{model_name}_feature_importance.csv')
            if os.path.exists(importance_path):
                self.feature_importance[model_name] = pd.read_csv(importance_path)

        logger.info(f"Models loaded from {model_dir}")


if __name__ == "__main__":
    # Example usage
    from data_collector import ERCOTDataCollector
    from feature_engineering import FeatureEngineer

    # Load and prepare data
    collector = ERCOTDataCollector()
    df = collector.load_data()

    if not df.empty:
        # Create features
        engineer = FeatureEngineer()
        df_features = engineer.create_features(df)

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = engineer.prepare_train_test(
            df_features, test_size=0.2, validation_size=0.1
        )

        # Train model
        model = PriceForecastModel()
        model.train(X_train, y_train, X_val, y_val)

        # Evaluate
        test_metrics = model.evaluate(X_test, y_test, dataset_name="test")

        # Save model
        model.save_models()

        print("\nTraining complete!")
