import logging
import math
from typing import List, Dict, Any

from app.schemas.predict import TimeSeriesTelemetry

logger = logging.getLogger(__name__)

class MetaClassifierRegistry:
    """
    A lightweight, in-memory k-Nearest Neighbors (KNN) style Meta-Classifier.
    It stores the [Telemetry -> Winning Model] mapping updated by background tournaments.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetaClassifierRegistry, cls).__new__(cls)
            # Bootstrapping the brain with our Baseline "Ground Truth" Profiles
            cls._instance.profiles = [
                # High seasonality, low trend → SARIMA
                {"seasonality_score": 1.0, "trend_slope_ratio": 0.0, "model": "SARIMA"},
                # Low seasonality, high trend → Holt-Winters
                {"seasonality_score": 0.0, "trend_slope_ratio": 1.0, "model": "Holt-Winters"},
                # Low seasonality, low trend → ARIMA (stationary)
                {"seasonality_score": 0.0, "trend_slope_ratio": 0.0, "model": "ARIMA"},
                # Moderate seasonality, low trend → SimpleES (mean-reverting)
                {"seasonality_score": 0.4, "trend_slope_ratio": 0.05, "model": "SimpleES"},
                # Very high trend, minimal seasonality → DriftModel
                {"seasonality_score": 0.05, "trend_slope_ratio": 1.0, "model": "DriftModel"},
                # Moderate trend + moderate seasonality → Theta
                {"seasonality_score": 0.5, "trend_slope_ratio": 0.5, "model": "Theta"},
            ]
        return cls._instance

    def update_brain(self, telemetry: TimeSeriesTelemetry, winning_model: str) -> None:
        """
        Retrain the Meta-Classifier: Stores the new mapping from the recent tournament.
        """
        trend_ratio = 0.0
        if abs(telemetry.mean) > 1e-8:
            trend_ratio = abs(telemetry.trend_slope / telemetry.mean)
            
        # Normalize the trend ratio for Euclidean distance calculation (heuristic capping at 1.0)
        norm_trend_ratio = min(1.0, trend_ratio * 100)

        new_profile = {
            "seasonality_score": telemetry.seasonality_score,
            "trend_slope_ratio": norm_trend_ratio,
            "model": winning_model
        }
        
        self.profiles.append(new_profile)
        # Maintain a rolling window of recent brain profiles (prevent memory bloat)
        if len(self.profiles) > 100:
            self.profiles.pop(0)
            
        logger.info(
            f"🧠 Meta-Classifier Brain Updated! Learned mapping: "
            f"[Seasonality: {telemetry.seasonality_score:.2f}, Trend Ratio: {norm_trend_ratio:.2f}] "
            f"-> {winning_model}"
        )

    def route(self, telemetry: TimeSeriesTelemetry) -> str:
        """
        Predicts the best model using 1-Nearest Neighbor distance in the telemetry feature space.
        """
        trend_ratio = 0.0
        if abs(telemetry.mean) > 1e-8:
            trend_ratio = abs(telemetry.trend_slope / telemetry.mean)
            
        norm_trend_ratio = min(1.0, trend_ratio * 100)
        
        best_model = "ARIMA"
        min_distance = float('inf')
        
        for profile in self.profiles:
            # Euclidean distance in the 2D feature space (Seasonality, Trend)
            dist = math.sqrt(
                (telemetry.seasonality_score - profile["seasonality_score"]) ** 2 +
                (norm_trend_ratio - profile["trend_slope_ratio"]) ** 2
            )
            if dist < min_distance:
                min_distance = dist
                best_model = profile["model"]
                
        return best_model

# Singleton instance to be imported across the app
registry = MetaClassifierRegistry()
