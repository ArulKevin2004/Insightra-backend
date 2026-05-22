import logging
import math
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats

from app.schemas.predict import (
    PredictionRequest,
    TimeSeriesPoint,
    PredictionMetrics,
    TimeSeriesTelemetry,
    PredictionResponse,
)

logger = logging.getLogger(__name__)


class PredictionService:
    @staticmethod
    def extract_telemetry(
        values: np.ndarray, index_array: np.ndarray, period: int
    ) -> TimeSeriesTelemetry:
        """
        Extract telemetry data characteristics including mean, variance, trend slope,
        and seasonality score (via Autocorrelation at the given period of detrended series).
        """
        mean_val = float(np.mean(values))
        var_val = float(np.var(values))

        # 1. Trend slope using ordinary least squares linear regression
        if len(values) > 1:
            slope, intercept, _, _, _ = stats.linregress(index_array, values)
            trend_slope = float(slope)
            # Detrend the series to isolate periodic behavior
            detrended = values - (intercept + slope * index_array)
        else:
            trend_slope = 0.0
            detrended = values.copy()

        # 2. Seasonality score via Autocorrelation Coefficient at period S of detrended data
        seasonality_score = 0.0
        if len(detrended) > 2 * period and np.std(detrended) > 1e-6:
            # Slice series to align lag S
            y_t = detrended[period:]
            y_lag = detrended[:-period]
            if np.std(y_t) > 1e-8 and np.std(y_lag) > 1e-8:
                corr = np.corrcoef(y_t, y_lag)[0, 1]
                # Map negative correlations as well (anti-seasonal peaks)
                seasonality_score = min(1.0, max(0.0, float(abs(corr))))

        return TimeSeriesTelemetry(
            mean=mean_val,
            variance=var_val,
            seasonality_score=seasonality_score,
            trend_slope=trend_slope,
        )

    @staticmethod
    def detect_seasonal_period(
        values: np.ndarray, requested_period: Optional[int]
    ) -> int:
        """
        Determines the seasonal period. If not provided in the request,
        scans lags to find the highest autocorrelation on detrended data.
        """
        N = len(values)
        if requested_period is not None:
            if requested_period < N / 2:
                return requested_period
            return max(2, int(N // 3))

        # Detrend the series first to prevent trend-induced auto-correlation
        slope, intercept, _, _, _ = stats.linregress(np.arange(N), values)
        detrended = values - (intercept + slope * np.arange(N))
        if np.std(detrended) < 1e-6:
            return 2  # No seasonality, return default minimum

        # Auto-detect period by finding the lag in [2, N/2] with maximum absolute ACF
        max_acf = -1.0
        best_lag = 1
        max_lag = min(30, int(N // 2))  # Scan up to 30 or N/2

        if max_lag < 2:
            return 2  # Default minimum seasonal period

        for lag in range(2, max_lag + 1):
            y_t = detrended[lag:]
            y_lag = detrended[:-lag]
            if np.std(y_t) > 1e-8 and np.std(y_lag) > 1e-8:
                acf = abs(np.corrcoef(y_t, y_lag)[0, 1])
                if acf > max_acf:
                    max_acf = acf
                    best_lag = lag

        return best_lag

    @classmethod
    def fit_and_predict_ar(
        cls,
        values: np.ndarray,
        horizon: int,
        p: int = 2,
        seasonal_period: Optional[int] = None,
    ) -> np.ndarray:
        """
        Fits an Autoregressive Model with standard lag (p) and optional seasonal lag
        using ordinary least squares (OLS) linear regression.
        Extrapolates recursively to forecast out-of-sample values.
        """
        N = len(values)
        # Handle cases with tiny histories
        if N <= p + 2:
            # Fallback to mean/trend extrapolation
            slope, intercept, _, _, _ = stats.linregress(np.arange(N), values)
            return intercept + slope * (np.arange(N, N + horizon))

        # Build design matrix X and target z
        # Feature columns: Intercept, Trend index (t), Lag 1, Lag 2, [Seasonal Lag]
        has_seasonal = seasonal_period is not None and N > seasonal_period + 2
        start_idx = seasonal_period if has_seasonal else p

        rows = []
        targets = []
        for t in range(start_idx, N):
            row = [1.0, float(t), float(values[t - 1]), float(values[t - 2])]
            if has_seasonal:
                row.append(float(values[t - seasonal_period]))
            rows.append(row)
            targets.append(values[t])

        X = np.array(rows)
        z = np.array(targets)

        # Fit OLS: coefficients w = (X^T X)^{-1} X^T z
        try:
            w, _, _, _ = np.linalg.lstsq(X, z, rcond=None)
        except np.linalg.LinAlgError:
            # Fallback to simple drift model
            slope, intercept, _, _, _ = stats.linregress(np.arange(N), values)
            return intercept + slope * (np.arange(N, N + horizon))

        # Recursive out-of-sample forecast
        forecast_buffer = list(values)
        for step in range(horizon):
            t_curr = N + step
            row_curr = [1.0, float(t_curr), float(forecast_buffer[-1]), float(forecast_buffer[-2])]
            if has_seasonal:
                row_curr.append(float(forecast_buffer[-seasonal_period]))
            
            pred_val = float(np.dot(row_curr, w))
            forecast_buffer.append(pred_val)

        return np.array(forecast_buffer[N:])

    @classmethod
    def fit_and_predict_holt_winters(
        cls,
        values: np.ndarray,
        horizon: int,
        alpha: float = 0.3,
        beta: float = 0.1,
    ) -> np.ndarray:
        """
        Fits a double exponential smoothing (Holt's Linear Trend Model)
        and projects future values.
        """
        N = len(values)
        if N < 2:
            return np.repeat(values[-1], horizon)

        # Initialize level and trend
        level = values[0]
        trend = values[1] - values[0]

        levels = np.zeros(N)
        trends = np.zeros(N)
        levels[0] = level
        trends[0] = trend

        for t in range(1, N):
            y_t = values[t]
            last_level = levels[t - 1]
            last_trend = trends[t - 1]

            levels[t] = alpha * y_t + (1 - alpha) * (last_level + last_trend)
            trends[t] = beta * (levels[t] - last_level) + (1 - beta) * last_trend

        # Forecast out-of-sample
        final_level = levels[-1]
        final_trend = trends[-1]
        forecasts = np.zeros(horizon)
        for h in range(1, horizon + 1):
            forecasts[h - 1] = final_level + h * final_trend

        return forecasts

    @classmethod
    def fit_and_predict_simple_es(
        cls,
        values: np.ndarray,
        horizon: int,
        alpha: float = 0.3,
    ) -> np.ndarray:
        """
        Simple Exponential Smoothing (SES). Best for stationary data with no
        trend or seasonality. Each forecast is a weighted average of all past
        observations, decaying geometrically by alpha.
        """
        N = len(values)
        if N < 1:
            return np.zeros(horizon)
        level = values[0]
        for t in range(1, N):
            level = alpha * values[t] + (1 - alpha) * level
        return np.full(horizon, level)

    @classmethod
    def fit_and_predict_drift_model(
        cls,
        values: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        """
        Random Walk with Drift. Projects the average per-step change forward.
        Ideal for strongly trending financial data (no seasonality assumed).
        Equivalent to a naive ARIMA(0,1,0) with an estimated drift constant.
        """
        N = len(values)
        if N < 2:
            return np.repeat(values[-1], horizon)
        drift = (values[-1] - values[0]) / (N - 1)
        last_val = values[-1]
        return np.array([last_val + drift * (h + 1) for h in range(horizon)])

    @classmethod
    def fit_and_predict_theta(
        cls,
        values: np.ndarray,
        horizon: int,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Theta Model (simplified). Decomposes the series into:
          - Theta-0: linear regression trend line
          - Theta-2: SES-based local level
        Final forecast = average of both. Strong for trend + noise data.
        """
        N = len(values)
        if N < 3:
            return cls.fit_and_predict_simple_es(values, horizon, alpha)
        idx = np.arange(N, dtype=float)
        slope, intercept, _, _, _ = stats.linregress(idx, values)
        theta0_forecast = intercept + slope * np.arange(N, N + horizon, dtype=float)
        level = values[0]
        for t in range(1, N):
            level = alpha * values[t] + (1 - alpha) * level
        theta2_forecast = np.full(horizon, level)
        return (theta0_forecast + theta2_forecast) / 2.0

    @classmethod
    def calculate_metrics(
        cls, actual: np.ndarray, predicted: np.ndarray
    ) -> PredictionMetrics:
        """
        Calculate mathematical validation metrics: MAPE, MAE, and RMSE.
        Injects a small epsilon for MAPE denominator to avoid division by zero.
        """
        # Epsilon-safeguarded absolute percentage error
        ape = np.abs((actual - predicted) / np.maximum(np.abs(actual), 1e-8)) * 100
        mape_val = float(np.mean(ape))
        mae_val = float(np.mean(np.abs(actual - predicted)))
        rmse_val = float(np.sqrt(np.mean((actual - predicted) ** 2)))

        return PredictionMetrics(mape=mape_val, mae=mae_val, rmse=rmse_val)

    @classmethod
    def run_backtest(
        cls,
        values: np.ndarray,
        model_name: str,
        seasonal_period: int,
        test_size: int,
    ) -> Tuple[PredictionMetrics, np.ndarray]:
        """
        Splits history into train and holdout validation subsets.
        Fits the chosen model on train, forecasts the test set,
        and computes holdout performance metrics.
        """
        N = len(values)
        train_vals = values[:-test_size]
        test_vals = values[-test_size:]

        # Route to the correct model implementation
        test_preds = cls._run_model(train_vals, test_size, model_name, seasonal_period)
        metrics = cls.calculate_metrics(test_vals, test_preds)
        return metrics, test_preds

    @classmethod
    def _run_model(
        cls,
        values: np.ndarray,
        horizon: int,
        model_name: str,
        seasonal_period: int = 2,
    ) -> np.ndarray:
        """Dispatch to the correct forecasting implementation by model name."""
        if model_name == "SARIMA":
            return cls.fit_and_predict_ar(values, horizon, p=2, seasonal_period=seasonal_period)
        elif model_name == "Holt-Winters":
            return cls.fit_and_predict_holt_winters(values, horizon)
        elif model_name == "SimpleES":
            return cls.fit_and_predict_simple_es(values, horizon)
        elif model_name == "DriftModel":
            return cls.fit_and_predict_drift_model(values, horizon)
        elif model_name == "Theta":
            return cls.fit_and_predict_theta(values, horizon)
        else:  # ARIMA (default)
            return cls.fit_and_predict_ar(values, horizon, p=2)

    @classmethod
    def predict(cls, request: PredictionRequest) -> PredictionResponse:
        """
        Handles prediction process:
        1. Formats input data
        2. Detects seasonal period
        3. Extracts telemetry characteristics
        4. Simulates production model routing (SARIMA, Holt-Winters, ARIMA)
        5. Performs walk-forward validation to calculate real MAPE/RMSE/MAE
        6. Forecasts the requested horizon into the future.
        """
        history_points = request.history
        n_points = len(history_points)
        values = np.array([p.value for p in history_points])
        timestamps = [p.timestamp for p in history_points]

        # 1. Seasonality & Period determination
        detected_period = cls.detect_seasonal_period(values, request.seasonal_period)

        # 2. Extract characteristics & statistical telemetry
        indices = np.arange(n_points)
        telemetry = cls.extract_telemetry(values, indices, detected_period)

        # 3. Model Routing Rules (Adaptive Meta-Classifier)
        # The routing decision is now made dynamically by the k-NN Meta-Classifier Brain!
        from app.services.registry import registry
        model_selected = registry.route(telemetry)
        logger.debug(f"🧠 Meta-Classifier routed traffic to: {model_selected}")

        # 4. Perform Walk-forward Validation (Backtest)
        # Holdout subset size (K): minimum 3, maximum 5, or N // 3
        test_size = max(3, min(5, n_points // 3))
        metrics, _ = cls.run_backtest(values, model_selected, detected_period, test_size)

        # 5. Out-of-sample forecasting via unified dispatcher
        future_vals = cls._run_model(values, request.horizon, model_selected, detected_period)

        # 6. Generate future timestamps assuming a regular frequency
        # Estimate historical average timedelta to extrapolate timestamps
        if n_points > 1:
            timedeltas = [
                timestamps[i] - timestamps[i - 1] for i in range(1, n_points)
            ]
            avg_delta = sum(timedeltas, timedelta()) / len(timedeltas)
        else:
            avg_delta = timedelta(days=1)  # Default fallback

        # Construct out-of-sample predictions list
        predictions = []
        last_timestamp = timestamps[-1]
        for step in range(request.horizon):
            next_timestamp = last_timestamp + (step + 1) * avg_delta
            pred_point = TimeSeriesPoint(
                timestamp=next_timestamp, value=float(future_vals[step])
            )
            predictions.append(pred_point)

        return PredictionResponse(
            model_selected=model_selected,
            predictions=predictions,
            metrics=metrics,
            telemetry=telemetry,
        )
