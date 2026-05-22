import math
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class TimeSeriesPoint(BaseModel):
    timestamp: datetime = Field(
        ..., description="The date and time of the observation."
    )
    value: float = Field(
        ..., description="The observed value at this timestamp."
    )


class PredictionRequest(BaseModel):
    history: List[TimeSeriesPoint] = Field(
        ...,
        description="The historical time series data. Must contain at least 10 points in chronological order.",
    )
    horizon: int = Field(
        ...,
        ge=1,
        le=100,
        description="The number of points/steps to forecast into the future.",
    )
    seasonal_period: Optional[int] = Field(
        None,
        ge=2,
        le=365,
        description="The length of the seasonal cycle (e.g. 7 for daily data with weekly pattern, 24 for hourly data). If not provided, will be auto-detected or default to 1.",
    )

    @field_validator("history")
    @classmethod
    def validate_history(cls, v: List[TimeSeriesPoint]) -> List[TimeSeriesPoint]:
        if len(v) < 10:
            raise ValueError(
                "History must contain at least 10 data points for mathematical stability and accurate forecasting."
            )

        # Check chronological order (ascending) and validate finite values
        for i in range(len(v)):
            point = v[i]
            if not math.isfinite(point.value):
                raise ValueError(
                    f"Observation at index {i} (timestamp: {point.timestamp}) must be a finite number."
                )
            if i > 0 and point.timestamp <= v[i - 1].timestamp:
                raise ValueError(
                    f"Timestamps must be in strictly ascending chronological order. "
                    f"Point at index {i} ({point.timestamp}) is not after index {i-1} ({v[i-1].timestamp})."
                )
        return v


class PredictionMetrics(BaseModel):
    mape: float = Field(
        ...,
        description="Mean Absolute Percentage Error (percentage) computed via backtesting.",
    )
    mae: float = Field(
        ...,
        description="Mean Absolute Error computed via backtesting.",
    )
    rmse: float = Field(
        ...,
        description="Root Mean Squared Error computed via backtesting.",
    )


class TimeSeriesTelemetry(BaseModel):
    mean: float = Field(
        ..., description="The average value of the historical sequence."
    )
    variance: float = Field(
        ..., description="The variance of the historical sequence."
    )
    seasonality_score: float = Field(
        ...,
        description="A score between 0.0 and 1.0 showing seasonality strength at the detected/given period.",
    )
    trend_slope: float = Field(
        ...,
        description="The slope of the linear trend (change per step) over the history.",
    )


class PredictionResponse(BaseModel):
    model_selected: str = Field(
        ..., description="The name of the forecaster selected by the routing engine."
    )
    predictions: List[TimeSeriesPoint] = Field(
        ..., description="The generated out-of-sample forecast points."
    )
    metrics: PredictionMetrics = Field(
        ...,
        description="Robust holdout error metrics calculated via walk-forward validation.",
    )
    telemetry: TimeSeriesTelemetry = Field(
        ...,
        description="Extracted statistical telemetry of the input time series.",
    )
    history: Optional[List[TimeSeriesPoint]] = Field(
        None,
        description="Optional returned historical points (useful for stock lookups)."
    )
