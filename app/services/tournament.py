import logging
import numpy as np
from typing import Dict, Any

from app.schemas.predict import TimeSeriesTelemetry
from app.services.registry import registry
from app.services.prediction import PredictionService

logger = logging.getLogger(__name__)

ALL_MODELS = ["ARIMA", "SARIMA", "Holt-Winters", "SimpleES", "DriftModel", "Theta"]


class TournamentRunner:
    @staticmethod
    def run_tournament(
        values: np.ndarray,
        telemetry: TimeSeriesTelemetry,
        seasonal_period: int,
    ) -> Dict[str, Any]:
        """
        Executes the background tournament across all 6 candidate models.
        Evaluates each on a held-out validation slice, selects the winner
        by lowest MAE, and updates the Meta-Classifier brain.

        Returns a full result dict with per-model scores and the winner.
        """
        logger.info("⚔️  Drift Detected! Starting Background Model Tournament (6 candidates)...")

        best_model = "ARIMA"
        best_mae = float("inf")
        n_points = len(values)
        test_size = max(3, min(6, n_points // 4))

        model_scores: Dict[str, Dict[str, float]] = {}

        for model in ALL_MODELS:
            try:
                metrics, _ = PredictionService.run_backtest(
                    values, model, seasonal_period, test_size
                )
                model_scores[model] = {
                    "mape": round(metrics.mape, 4),
                    "mae": round(metrics.mae, 4),
                    "rmse": round(metrics.rmse, 4),
                }
                logger.info(
                    f"   --> [{model}] MAE={metrics.mae:.4f}  "
                    f"MAPE={metrics.mape:.4f}%  RMSE={metrics.rmse:.4f}"
                )
                if metrics.mae < best_mae:
                    best_mae = metrics.mae
                    best_model = model
            except Exception as e:
                logger.warning(f"   --> [{model}] failed: {str(e)}")
                model_scores[model] = {"mape": None, "mae": None, "rmse": None}

        logger.info(f"🏆 Tournament Winner: {best_model} (MAE={best_mae:.4f})")

        # Update the Meta-Classifier Brain with the new empirical mapping
        registry.update_brain(telemetry, best_model)

        return {
            "winner": best_model,
            "scores": model_scores,
            "winning_mae": round(best_mae, 4),
        }
