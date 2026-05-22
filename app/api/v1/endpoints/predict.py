import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
import numpy as np
import httpx
from fastapi import APIRouter, HTTPException, Query, status, BackgroundTasks
from pydantic import BaseModel

from app.schemas.predict import PredictionRequest, PredictionResponse, TimeSeriesPoint
from app.services.prediction import PredictionService
from app.services.drift_monitor import drift_monitor
from app.services.tournament import TournamentRunner

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate time-series predictions",
    description=(
        "Ingests historical time-series observations, auto-selects the best "
        "forecasting algorithm (SARIMA, Holt-Winters, or ARIMA), calculates "
        "production-grade error metrics (MAPE, MAE, RMSE) using walk-forward "
        "historical validation, and returns the forecast observations."
    ),
)
def predict(request: PredictionRequest, background_tasks: BackgroundTasks) -> PredictionResponse:
    """
    Predict future values for a given time-series sequence.
    """
    logger.info(
        f"Received prediction request with {len(request.history)} points, "
        f"horizon={request.horizon}, seasonal_period={request.seasonal_period}"
    )
    try:
        response = PredictionService.predict(request)
        logger.info(
            f"Successfully generated prediction using model: {response.model_selected}. "
            f"MAPE: {response.metrics.mape:.4f}%"
        )
        
        # Schedule the background MLOps telemetry processing (Drift Monitoring)
        values = np.array([p.value for p in request.history])
        detected_period = PredictionService.detect_seasonal_period(values, request.seasonal_period)
        
        background_tasks.add_task(
            drift_monitor.process_telemetry,
            values,
            response.telemetry,
            response.metrics,
            detected_period
        )
        
        return response
    except ValueError as val_err:
        logger.warning(f"Validation error during forecasting: {str(val_err)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Forecasting validation failed: {str(val_err)}",
        )
    except Exception as exc:
        logger.error(
            f"Unexpected error during time-series forecasting: {str(exc)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while calculating the forecast.",
        )


@router.get(
    "/stock/{ticker}",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Get live stock data and forecast",
    description="Fetches live daily stock prices from Yahoo Finance for a given ticker and forecasts future prices.",
)
async def predict_stock(
    ticker: str,
    background_tasks: BackgroundTasks,
    horizon: int = 10,
) -> PredictionResponse:
    """
    Fetch live daily closing prices from Yahoo Finance for a stock ticker,
    and generate out-of-sample forecasts.
    """
    logger.info(f"Received live stock prediction request for ticker: {ticker}, horizon={horizon}")
    try:
        # 1. Fetch live stock data from Yahoo Finance chart endpoint (60 days)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker.upper()}?interval=1d&range=60d"
        headers = {"User-Agent": "Mozilla/5.0"}
        async with httpx.AsyncClient() as client:
            res = await client.get(url, headers=headers)
            if res.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to fetch stock data for ticker '{ticker}' from Yahoo Finance.",
                )
            data = res.json()
            
        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        close_prices = result["indicators"]["quote"][0]["close"]
        
        # Build history points (filtering out any null closing prices)
        history = []
        for t, p in zip(timestamps, close_prices):
            if p is not None:
                history.append(
                    TimeSeriesPoint(
                        timestamp=datetime.fromtimestamp(t),
                        value=float(p)
                    )
                )
                
        if len(history) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Not enough historical stock points returned (got {len(history)}, need >= 10).",
            )
            
        # Create PredictionRequest payload
        request = PredictionRequest(history=history, horizon=horizon, seasonal_period=None)
        
        # 2. Run prediction engine
        response = PredictionService.predict(request)
        
        # 3. Schedule drift monitoring asynchronously
        values = np.array([p.value for p in history])
        detected_period = PredictionService.detect_seasonal_period(values, request.seasonal_period)
        
        background_tasks.add_task(
            drift_monitor.process_telemetry,
            values,
            response.telemetry,
            response.metrics,
            detected_period
        )
        
        # We also override the return history with our live fetched stock history!
        # This makes it easy for the frontend to render the exact fetched points
        response.history = history
        return response
    except HTTPException as http_exc:
        raise http_exc
    except Exception as exc:
        logger.error(f"Error fetching stock prediction for {ticker}: {str(exc)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while fetching or forecasting stock {ticker}.",
        )


# ---------------------------------------------------------------------------
# Tournament Endpoint — runs all 6 models and returns winner + per-model scores
# ---------------------------------------------------------------------------

class TournamentRequest(BaseModel):
    history: List[TimeSeriesPoint]
    seasonal_period: Optional[int] = None

class TournamentResponse(BaseModel):
    previous_model: str
    winner: str
    winning_mae: float
    scores: Dict[str, Any]


@router.post(
    "/tournament",
    response_model=TournamentResponse,
    status_code=status.HTTP_200_OK,
    summary="Run full model tournament and retrain meta-classifier",
    description=(
        "Runs all 6 forecasting models (ARIMA, SARIMA, Holt-Winters, SimpleES, "
        "DriftModel, Theta) on the supplied history. Returns per-model MAPE/MAE/RMSE "
        "scores, the tournament winner, and updates the Meta-Classifier brain."
    ),
)
async def run_tournament(
    request: TournamentRequest,
) -> TournamentResponse:
    from app.services.registry import registry
    values = np.array([p.value for p in request.history])
    if len(values) < 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Need at least 10 history points to run a tournament.",
        )

    detected_period = PredictionService.detect_seasonal_period(values, request.seasonal_period)
    indices = np.arange(len(values))
    telemetry = PredictionService.extract_telemetry(values, indices, detected_period)

    previous_model = registry.route(telemetry)
    result = TournamentRunner.run_tournament(values, telemetry, detected_period)

    return TournamentResponse(
        previous_model=previous_model,
        winner=result["winner"],
        winning_mae=result["winning_mae"],
        scores=result["scores"],
    )


# ---------------------------------------------------------------------------
# Flexible Stream Endpoint — dataset-agnostic, works with any time-series
# Feed a ?step=N to replay historical data tick by tick
# ---------------------------------------------------------------------------

# In-memory buffer keyed by ticker (or custom dataset key)
_stream_buffers: Dict[str, List[Dict]] = {}


@router.get(
    "/stream/{dataset_key}",
    status_code=status.HTTP_200_OK,
    summary="Get one data point by step index (replay stream)",
    description=(
        "Returns a single historical observation by step index. "
        "If dataset_key is a known stock ticker (e.g. AAPL), it auto-fetches "
        "Yahoo Finance data as the buffer. Otherwise pass the data once via "
        "POST /stream/load to pre-load a custom dataset."
    ),
)
async def stream_get_step(
    dataset_key: str,
    step: int = Query(0, ge=0),
) -> Dict:
    key = dataset_key.upper()

    # Auto-fetch Yahoo Finance buffer if not cached or key looks like a ticker
    if key not in _stream_buffers:
        try:
            # 5-minute interval, last 5 trading days — gives ~400 intraday ticks
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{key}?interval=5m&range=5d"
            headers = {"User-Agent": "Mozilla/5.0"}
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.get(url, headers=headers)
            if res.status_code == 200:
                data = res.json()
                result = data["chart"]["result"][0]
                timestamps = result["timestamp"]
                closes = result["indicators"]["quote"][0]["close"]
                buffer = [
                    {"timestamp": datetime.fromtimestamp(t).isoformat(), "value": float(p)}
                    for t, p in zip(timestamps, closes) if p is not None
                ]
                _stream_buffers[key] = buffer
                logger.info(f"📦 Buffered {len(buffer)} points for stream key '{key}'")
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Dataset '{dataset_key}' not found and not a valid ticker."
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load data for '{dataset_key}': {str(e)}"
            )

    buffer = _stream_buffers[key]
    total = len(buffer)
    if step >= total:
        return {"done": True, "total": total}

    return {
        "done": False,
        "step": step,
        "total": total,
        "point": buffer[step],
    }


@router.post(
    "/stream/load",
    status_code=status.HTTP_200_OK,
    summary="Load a custom dataset into the stream buffer",
    description="Pre-load any custom time-series data into the streaming buffer under a given key.",
)
async def stream_load_custom(
    dataset_key: str,
    data: List[TimeSeriesPoint],
) -> Dict:
    if len(data) < 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Need at least 10 points to load a stream buffer.",
        )
    key = dataset_key.upper()
    _stream_buffers[key] = [
        {"timestamp": p.timestamp.isoformat(), "value": p.value} for p in data
    ]
    logger.info(f"📦 Custom dataset loaded into stream buffer '{key}' ({len(data)} points)")
    return {"loaded": True, "key": key, "total": len(data)}
