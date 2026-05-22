import logging
import numpy as np
from typing import List

from app.schemas.predict import TimeSeriesTelemetry, PredictionMetrics
from app.services.tournament import TournamentRunner

logger = logging.getLogger(__name__)

class DriftMonitorService:
    """
    Simulates the background MLOps Drift Monitor.
    Tracks recent error metrics in memory to detect sudden Concept Drift.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DriftMonitorService, cls).__new__(cls)
            cls._instance.recent_errors = []
        return cls._instance

    def process_telemetry(
        self, 
        values: np.ndarray, 
        telemetry: TimeSeriesTelemetry, 
        metrics: PredictionMetrics,
        seasonal_period: int
    ) -> None:
        """
        Ingests the latest prediction telemetry and error metrics asynchronously.
        Checks for drift, and if detected, triggers the Asynchronous Tournament.
        """
        current_error = metrics.mape
        
        # We need a small baseline of at least 2 past requests to detect anomalies
        if len(self.recent_errors) >= 2:
            baseline_error = sum(self.recent_errors) / len(self.recent_errors)
            
            # Simple Drift Rule: If error spikes by more than 50% relative, or absolute 15% jump
            # Ensure we only flag meaningful drift (ignore tiny fluctuations if baseline is near 0)
            drift_detected = (current_error > baseline_error + 15.0) or \
                             (baseline_error > 5.0 and current_error > baseline_error * 1.5)
            
            if drift_detected:
                logger.warning(
                    f"🚨 CONCEPT DRIFT DETECTED! Current MAPE ({current_error:.2f}%) "
                    f"severely exceeds baseline ({baseline_error:.2f}%)."
                )
                
                # Trigger the Tournament to self-heal the Champion mapping
                TournamentRunner.run_tournament(values, telemetry, seasonal_period)
                
                # Clear recent errors after hot-swapping so we establish a fresh baseline
                self.recent_errors = []
                return
        
        # No drift detected, log telemetry cleanly
        logger.info(f"📊 Telemetry logged asynchronously. Current MAPE: {current_error:.2f}% (Healthy)")
        self.recent_errors.append(current_error)
        
        # Keep a rolling window of the last 10 requests to track the baseline
        if len(self.recent_errors) > 10:
            self.recent_errors.pop(0)

# Singleton instance to be used across the app
drift_monitor = DriftMonitorService()
