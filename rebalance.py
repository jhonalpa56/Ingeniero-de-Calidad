import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import time

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:  # pragma: no cover - optional dependency
    SARIMAX = None

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - optional dependency
    go = None

try:
    from prophet import Prophet
except ImportError:  # pragma: no cover - optional dependency
    Prophet = None

# ====== CONFIG ======
# Tickers Yahoo (ajusta si alguno no te cotiza: p.ej. VanEck Semis UCITS -> "SMH.L" o "SMGB.MI")
tickers = {
    "SWDA.L": {"name": "MSCI World Core", "bucket": "core"},
    "2B7K.DE": {"name": "MSCI World SRI", "bucket": "core"},
    "EQQQ.L": {"name": "Nasdaq-100", "bucket": "nasdaq"},
    "SMH.L": {"name": "Semiconductors UCITS", "bucket": "semis"},  # prueba "SMH.L" o "SMGB.MI"
    "ARKK": {"name": "ARK Innovation", "bucket": "ark"},
    "VBTC.DE": {"name": "Bitcoin ETN", "bucket": "btc"},
    "IGLN.L": {"name": "Physical Gold", "bucket": "gold"},
    "ICOM.L": {"name": "Broad Commodities", "bucket": "commodities"},
    "U3O8.DE": {"name": "Uranium Miners", "bucket": "uranium"},
    "INRG.L": {"name": "Global Clean Energy", "bucket": "energy"},
}

# Configuración editable por el usuario: añade ETFs, posiciones y efectivo aquí.
# - Para seguir un ETF agrega la clave en `tickers` y, si tienes posición,
#   define sus unidades y costo promedio (y opcionalmente el capital invertido).
# - Actualiza `AVAILABLE_CASH_USD` cada vez que cambie tu liquidez.
USER_POSITIONS = {
    "2B7K.DE": {"units": 157.5842, "avg_cost": 12.03, "currency": "EUR"},
    "ARKK": {"units": 8.0, "avg_cost": 84.67, "currency": "USD"},
    "EQQQ.L": {"units": 2.8483, "avg_cost": 449.8502, "currency": "GBP"},
    "ICOM.L": {"units": 60.7811, "avg_cost": 7.65, "currency": "GBP"},
    "IGLN.L": {"units": 7.1056, "avg_cost": 81.9855, "currency": "GBP"},
    "INRG.L": {"units": 33.8518, "avg_cost": 7.2758, "currency": "GBP"},
    "SMH.L": {"units": 18.0054, "avg_cost": 56.706, "currency": "GBP"},
    "SWDA.L": {"units": 23.475, "avg_cost": 94.1465, "currency": "GBP"},
    "U3O8.DE": {"units": 22.6566, "avg_cost": 12.954, "currency": "EUR"},
    "VBTC.DE": {"units": 19.5796, "avg_cost": 53.6524, "currency": "EUR"},
}
AVAILABLE_CASH_USD = 11500.25
RECOMMENDATION_THRESHOLDS = {
    "buy_more": 0.02,   # señal de compra si el retorno esperado >= 2%
    "trim": -0.02,      # señal de venta si el retorno esperado <= -2%
}
PROPHET_MIN_HISTORY = 180  # minimo de datos diarios para activar Prophet
DEFAULT_HORIZONS = [5, 21, 63]  # ~1 semana, 1 mes, 3 meses aprox


history_store: Dict[str, pd.DataFrame] = {}
ticker_meta: Dict[str, Dict[str, float]] = {}
CHART_OUTPUT_DIR = Path("charts")
FORECAST_HORIZON_DEFAULT = 21  # ~1 mes de ruedas (21 dias habiles)
BUCKET_FORECAST_HORIZON = {
    "core": 21,
    "nasdaq": 21,
    "semis": 21,
    "ark": 21,
    "btc": 21,
    "gold": 21,
    "commodities": 21,
    "uranium": 21,
    "energy": 21,
}
MODEL_PERIOD_DEFAULT = "5y"
BUCKET_MODEL_PERIOD = {
    "core": "5y",
    "nasdaq": "5y",
    "semis": "5y",
    "ark": "5y",
    "btc": "5y",
    "gold": "5y",
    "commodities": "5y",
    "uranium": "5y",
    "energy": "5y",
}
MACRO_MODEL_PERIOD = "5y"
CANDLE_LOOKBACK = 252
GLOBAL_MACRO_TICKERS = {
    "^GSPC": "S&P 500",
    "^VIX": "CBOE VIX",
    "^MOVE": "ICE BofA MOVE",
    "^TNX": "US 10Y Treasury Yield",
    "^IRX": "US 13W Treasury Yield",
    "UUP": "US Dollar Bullish ETF",
    "HYG": "High Yield Corp Bond",
    "LQD": "Investment Grade Bond",
}
BUCKET_MACRO_TICKERS = {
    "core": {
        "ACWI": "MSCI ACWI ETF",
        "EFA": "MSCI EAFE ETF",
        "EEM": "MSCI Emerging Markets ETF",
    },
    "nasdaq": {
        "^NDX": "Nasdaq 100 Index",
        "QQQ": "Invesco QQQ",
        "^SOX": "PHLX Semiconductor Index",
    },
    "semis": {
        "^SOX": "PHLX Semiconductor Index",
        "SOXX": "iShares Semiconductor ETF",
        "SMH": "VanEck Semiconductor ETF",
    },
    "ark": {
        "ARKK": "ARK Innovation ETF",
        "ARKW": "ARK Next Gen Internet ETF",
        "QQQ": "Invesco QQQ",
    },
    "btc": {
        "BTC-USD": "Bitcoin Spot",
        "ETH-USD": "Ethereum Spot",
        "^VIX": "CBOE VIX",
    },
    "gold": {
        "GC=F": "Gold Futures",
        "GLD": "SPDR Gold Trust",
        "SI=F": "Silver Futures",
    },
    "commodities": {
        "DBC": "Commodity Basket ETF",
        "USO": "WTI Oil ETF",
        "HG=F": "Copper Futures",
    },
    "uranium": {
        "URA": "Global X Uranium ETF",
        "URNM": "Sprott Uranium Miners",
        "CCJ": "Cameco Corp",
    },
    "energy": {
        "ICLN": "iShares Global Clean Energy",
        "TAN": "Invesco Solar ETF",
        "LIT": "Global Lithium ETF",
    },
}
PCA_COMPONENTS = 3
MAX_SARIMAX_ORDER = 2
BACKTEST_WINDOW = 60
BACKTEST_STEPS = 5
BACKTEST_ROLLS = 6

YF_SESSION = requests.Session()
YF_SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }
)
try:
    import yfinance.shared as yshared  # type: ignore

    yshared._DEFAULT_HEADERS["User-Agent"] = YF_SESSION.headers["User-Agent"]  # type: ignore[attr-defined]
except Exception:
    pass
_YF_SESSION_READY = False


def ensure_yf_session_ready() -> None:
    global _YF_SESSION_READY
    if _YF_SESSION_READY:
        return
    try:
        YF_SESSION.get("https://finance.yahoo.com", timeout=5)
    except Exception as exc:
        logger.warning("No se pudo preparar la sesion de Yahoo Finance: %s", exc)
    _YF_SESSION_READY = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rebalance")


def pct_str(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:,.2f}%"


def project_return(ret: float, base_period: int, horizon_days: int) -> float:
    if ret is None or pd.isna(ret):
        return np.nan
    if ret <= -0.95:
        return np.nan
    factor = horizon_days / max(base_period, 1)
    try:
        return np.expm1(np.log1p(ret) * factor)
    except Exception:
        return np.nan


def blended_momentum(metrics_row: pd.Series, horizon_days: int) -> float:
    components: list[float] = []
    weights: list[float] = []
    r1m = metrics_row.get("r1m")
    r1w = metrics_row.get("r1w")
    r1d = metrics_row.get("r1d")

    monthly = project_return(r1m, 21, horizon_days)
    if pd.notna(monthly):
        components.append(monthly)
        weights.append(0.65)

    weekly = project_return(r1w, 5, horizon_days)
    if pd.notna(weekly):
        components.append(weekly)
        weights.append(0.25)

    daily = project_return(r1d, 1, horizon_days)
    if pd.notna(daily):
        components.append(daily)
        weights.append(0.10)

    if not components:
        return np.nan
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    return float(np.dot(weights, np.array(components, dtype=float)))


def summarize_recommendation(
    ticker: str,
    metrics_row: pd.Series,
    forecast: pd.DataFrame,
    position: Optional[Dict[str, Any]],
    horizon_days: int,
) -> Dict[str, Any]:
    """Build a structured recommendation using price momentum and forecast."""
    name = tickers[ticker]["name"]
    bucket = tickers[ticker]["bucket"]
    currency = metrics_row.get("currency", "USD")
    price_usd = metrics_row.get("price_usd", np.nan)

    last_forecast = np.nan
    forecast_window = len(forecast) if forecast is not None else 0
    if forecast is not None and not forecast.empty and pd.notna(price_usd) and price_usd:
        last_forecast = float(forecast["yhat"].iloc[-1])
    expected_return = (
        (last_forecast / price_usd - 1.0) if pd.notna(last_forecast) and price_usd else np.nan
    )

    position_currency = position.get("currency") if position else None
    units = float(position.get("units", 0.0)) if position else 0.0
    invested = position.get("invested_usd") if position else None
    avg_cost = position.get("avg_cost_usd") if position else None

    adjustment_note = ""
    momentum_estimate = blended_momentum(metrics_row, horizon_days)
    if pd.notna(expected_return):
        if pd.notna(momentum_estimate):
            diff = abs(expected_return - momentum_estimate)
            threshold = 0.04
            if diff > threshold:
                weight_momentum = 0.55 + 1.5 * (diff - threshold)
                weight_momentum = max(0.55, min(0.95, weight_momentum))
                weight_forecast = max(0.0, 1.0 - weight_momentum)
                adjusted = weight_forecast * expected_return + weight_momentum * momentum_estimate
                adjustment_note = (
                    f"Forecast reconciliado con momentum (peso {weight_momentum:.0%}, diff {diff:.2%})."
                )
                expected_return = adjusted
        else:
            adjustment_note = "Sin momentum disponible; se mantiene forecast base."
    elif pd.notna(momentum_estimate):
        expected_return = momentum_estimate
        adjustment_note = "Forecast no disponible; se usa momentum proyectado."

    signal = "NO_DATA"
    rationale = "Sin forecast disponible."
    if pd.notna(expected_return):
        buy_thr = RECOMMENDATION_THRESHOLDS["buy_more"]
        sell_thr = RECOMMENDATION_THRESHOLDS["trim"]
        if units and units > 0:
            if expected_return >= buy_thr:
                signal = "BUY_MORE"
                rationale = "Retorno esperado atractivo; considera aumentar posicion."
            elif expected_return <= sell_thr:
                signal = "TRIM_OR_SELL"
                rationale = "Retorno esperado negativo; evaluar reducir exposicion."
            else:
                signal = "HOLD"
                rationale = "Retorno esperado neutro; mantener y monitorear."
        else:
            if expected_return >= buy_thr:
                signal = "ADD"
                rationale = "Retorno esperado positivo; candidato para nuevo capital."
            elif expected_return <= sell_thr:
                signal = "AVOID"
                rationale = "Retorno esperado debil; evitar nuevas compras."
            else:
                signal = "WATCH"
                rationale = "Retorno balanceado; seguir monitoreando."

    return {
        "ticker": ticker,
        "name": name,
        "bucket": bucket,
        "currency": currency,
        "price_usd": price_usd,
        "momentum_1d": metrics_row.get("r1d"),
        "momentum_1w": metrics_row.get("r1w"),
        "momentum_1m": metrics_row.get("r1m"),
        "dd30": metrics_row.get("dd30"),
        "expected_return": expected_return,
        "forecast_window": forecast_window,
        "signal": signal,
        "rationale": rationale,
        "units": units,
        "avg_cost_usd": avg_cost,
        "invested_usd": invested,
        "forecast_price": last_forecast,
        "note": adjustment_note,
        "position_currency": position_currency,
        "momentum_proj": momentum_estimate,
    }


def safe_download(
    tickers: Any,
    *,
    retries: int = 3,
    delay: float = 1.0,
    **kwargs: Any,
) -> pd.DataFrame:
    ensure_yf_session_ready()
    kwargs.setdefault("threads", False)
    single_ticker = isinstance(tickers, str) or (isinstance(tickers, (list, tuple, set)) and len(tickers) == 1)
    if isinstance(tickers, (list, tuple, set)) and len(tickers) == 1:
        tickers = list(tickers)[0]
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                tickers,
                progress=False,
                **kwargs,
            )
            if df is not None and not df.empty:
                return df
            if single_ticker:
                history_kwargs = {}
                for key in ("period", "interval", "start", "end", "auto_adjust"):
                    if key in kwargs:
                        history_kwargs[key] = kwargs[key]
                ticker_obj = yf.Ticker(str(tickers))
                df = ticker_obj.history(**history_kwargs)
                if df is not None and not df.empty:
                    return df
            if df is None:
                return df
        except Exception as exc:
            logger.error("Error descargando %s (intento %s/%s): %s", tickers, attempt, retries, exc)
        time.sleep(delay * attempt)
    logger.warning("Sin datos para %s tras %s intentos.", tickers, retries)
    return pd.DataFrame()


def chunked_list(values: list[str], size: int) -> list[list[str]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


fx_tickers = {"GBP": "GBPUSD=X", "EUR": "EURUSD=X", "USD": None}


def load_fx() -> dict[str, float]:
    fx_rates: dict[str, float] = {"USD": 1.0}
    for currency, ticker in fx_tickers.items():
        if not ticker:
            continue
        df = safe_download(
            ticker,
            period="5d",
            interval="1d",
            auto_adjust=False,
        ).dropna()
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(ticker, axis=1, level=1)
            except KeyError:
                df.columns = df.columns.get_level_values(0)
        if df.empty:
            continue
        price = df["Close"]
        if isinstance(price, pd.DataFrame):
            price = price.iloc[:, 0]
        fx_rates[currency] = float(price.iat[-1])
    return fx_rates


def infer_currency_and_scale(ticker: str, prices: pd.Series) -> tuple[str, float]:
    currency = "USD"
    if ticker.endswith(".L"):
        currency = "GBP"
    elif ticker.endswith(".DE"):
        currency = "EUR"

    scale = 1.0
    clean = prices.dropna()
    if currency == "GBP" and not clean.empty:
        median_price = float(clean.median())
        if median_price > 200:
            scale = 0.01
    return currency, scale


def normalize_position_data(
    ticker: str,
    position: Dict[str, Any],
    fx_rates: dict[str, float],
) -> Dict[str, Any]:
    meta = ticker_meta.get(ticker, {})
    units = float(position.get("units", 0.0))
    currency = position.get("currency") or meta.get("currency") or "USD"
    fx_rate = fx_rates.get(currency, 1.0)

    avg_cost_usd = position.get("avg_cost_usd")
    if avg_cost_usd is None:
        avg_cost_local = position.get("avg_cost")
        if avg_cost_local is not None:
            avg_cost_usd = float(avg_cost_local) * fx_rate
    else:
        avg_cost_usd = float(avg_cost_usd)

    invested_usd = position.get("invested_usd")
    if invested_usd is None and avg_cost_usd is not None:
        invested_usd = units * avg_cost_usd
    elif invested_usd is not None:
        invested_usd = float(invested_usd)

    return {
        "units": units,
        "avg_cost_usd": avg_cost_usd,
        "invested_usd": invested_usd,
        "currency": currency,
    }


def get_macro_universe() -> Dict[str, str]:
    universe: Dict[str, str] = dict(GLOBAL_MACRO_TICKERS)
    for symbol, info in tickers.items():
        bucket = info.get("bucket", "other")
        for macro_ticker, desc in BUCKET_MACRO_TICKERS.get(bucket, {}).items():
            universe.setdefault(macro_ticker, desc)
    return universe


def load_macro_features() -> pd.DataFrame:
    universe = get_macro_universe()
    if not universe:
        return pd.DataFrame()

    macro_list = list(universe.keys())
    frames: list[pd.DataFrame] = []
    for chunk in chunked_list(macro_list, 8):
        chunk_df = safe_download(
            chunk,
            period=MACRO_MODEL_PERIOD,
            interval="1d",
            auto_adjust=False,
        )
        if not chunk_df.empty:
            frames.append(chunk_df)
    if not frames:
        logger.warning("Descarga macro sin datos.")
        return pd.DataFrame()
    data = pd.concat(frames, axis=1)

    if isinstance(data.columns, pd.MultiIndex):
        try:
            close = data.xs("Close", axis=1, level=0)
        except KeyError:
            close = data.copy()
            close.columns = close.columns.get_level_values(-1)
    else:
        close = data

    close = close.replace(0, np.nan).ffill().bfill()
    close = close.dropna(how="all")
    if close.empty:
        logger.warning("Precios macro sin valores tras limpieza.")
        return pd.DataFrame()

    log_close = np.log(close).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    returns = log_close.diff().dropna(how="all")
    if returns.empty:
        return pd.DataFrame()

    returns = returns.fillna(0.0)
    return returns.reindex(columns=macro_list).fillna(0.0)


def build_price_features(prices: pd.Series) -> pd.DataFrame:
    clean = prices.dropna()
    if clean.empty:
        return pd.DataFrame()
    log_price = np.log(clean.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    log_price = log_price.dropna()
    returns = log_price.diff()
    df = pd.DataFrame(
        {
            "ret_1d": returns,
            "ret_5d": returns.rolling(5).sum(),
            "ret_10d": returns.rolling(10).sum(),
            "vol_5d": returns.rolling(5).std(),
            "vol_10d": returns.rolling(10).std(),
        }
    ).dropna()
    df = df.fillna(0.0)
    return df


def get_bucket_macro_columns(bucket: str) -> set[str]:
    cols = set(GLOBAL_MACRO_TICKERS.keys())
    cols.update(BUCKET_MACRO_TICKERS.get(bucket, {}).keys())
    return cols


def compute_pca_features(data: pd.DataFrame, components: int = PCA_COMPONENTS) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame(index=data.index if data is not None else None)
    demeaned = data - data.mean()
    if demeaned.shape[1] < 1:
        return pd.DataFrame(index=data.index)
    try:
        cov = np.cov(demeaned.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return pd.DataFrame(index=data.index)
    order = np.argsort(eigvals)[::-1][: min(components, demeaned.shape[1])]
    eigvecs = eigvecs[:, order]
    projected = demeaned.values @ eigvecs
    features = pd.DataFrame(index=data.index)
    for idx in range(projected.shape[1]):
        features[f"PCA_{idx+1}"] = projected[:, idx]
    return features


def linear_forecast(series: pd.Series, steps: int) -> pd.DataFrame:
    clean = series.dropna()
    if clean.empty or len(clean) < 2:
        return pd.DataFrame()
    x = np.arange(len(clean), dtype=float)
    y = clean.to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    future_index = pd.bdate_range(clean.index[-1] + pd.Timedelta(days=1), periods=steps)
    x_future = np.arange(len(clean), len(clean) + steps, dtype=float)
    values = slope * x_future + intercept
    return pd.DataFrame(
        {
            "yhat": values,
            "yhat_lower": values,
            "yhat_upper": values,
        },
        index=future_index,
    )


def metrics(ticker: str, fx_rates: dict[str, float]) -> pd.Series:
    """Fetches momentum and drawdown stats for the ticker."""
    global history_store, ticker_meta
    bucket = tickers.get(ticker, {}).get("bucket", "other")
    period = BUCKET_MODEL_PERIOD.get(bucket, MODEL_PERIOD_DEFAULT)
    df = safe_download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
    )
    if df.empty:
        logger.warning("Sin datos descargados para %s", ticker)
        return pd.Series(
            {
                "price_usd": np.nan,
                "r1d": np.nan,
                "r1w": np.nan,
                "r1m": np.nan,
                "dd30": np.nan,
                "currency": "USD",
            }
        )

    df = df.dropna()
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, axis=1, level=1)
        except KeyError:
            df.columns = df.columns.get_level_values(0)
    if df.empty or "Close" not in df:
        return pd.Series({
            "price_usd": np.nan,
            "r1d": np.nan,
            "r1w": np.nan,
            "r1m": np.nan,
            "dd30": np.nan,
            "currency": "USD",
        })

    px = df["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]

    currency, scale = infer_currency_and_scale(ticker, px)
    fx_rate = fx_rates.get(currency, 1.0)
    ticker_meta[ticker] = {"currency": currency, "scale": scale, "fx_rate": fx_rate}
    px_local = px * scale
    ohlc_columns = ["Open", "High", "Low", "Close"]
    ohlc_usd = pd.DataFrame()
    if set(ohlc_columns).issubset(df.columns):
        ohlc_local = df[ohlc_columns] * scale
        ohlc_usd = (ohlc_local * fx_rate).astype(float)
        history_store[ticker] = ohlc_usd
    else:
        history_store[ticker] = pd.DataFrame()

    last_local = px_local.iat[-1]
    prev_1d = px_local.iat[-2] if len(px_local) >= 2 else np.nan
    prev_1w = px_local.iat[-6] if len(px_local) >= 6 else np.nan
    prev_1m = px_local.iat[-22] if len(px_local) >= 22 else np.nan
    roll_max_30 = px_local.rolling(30).max().iat[-1] if len(px_local) >= 30 else np.nan

    r_1d = last_local / prev_1d - 1 if pd.notna(prev_1d) and prev_1d else np.nan
    r_1w = last_local / prev_1w - 1 if pd.notna(prev_1w) and prev_1w else np.nan
    r_1m = last_local / prev_1m - 1 if pd.notna(prev_1m) and prev_1m else np.nan
    dd30 = last_local / roll_max_30 - 1 if pd.notna(roll_max_30) and roll_max_30 else np.nan

    price_usd = float(last_local * fx_rate)

    return pd.Series({
        "price_usd": price_usd,
        "r1d": r_1d,
        "r1w": r_1w,
        "r1m": r_1m,
        "dd30": dd30,
        "currency": currency,
    })


def factor_model_forecast(
    price_series: pd.Series,
    macro_returns: pd.DataFrame,
    bucket: str,
    steps: int,
) -> pd.DataFrame:
    price = price_series.dropna()
    if price.empty or len(price) < 5:
        return pd.DataFrame()

    log_price = np.log(price.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    log_price = log_price.dropna()
    if log_price.empty:
        return pd.DataFrame()

    log_returns = log_price.diff().dropna()
    if log_returns.empty:
        return pd.DataFrame()

    window = min(len(log_returns), 252)
    ret = log_returns.iloc[-window:]

    features = pd.DataFrame(index=ret.index)
    price_feat = build_price_features(price)
    if not price_feat.empty:
        features = features.join(price_feat, how="outer")

    if macro_returns is not None and not macro_returns.empty:
        bucket_cols = get_bucket_macro_columns(bucket)
        macro_cols = [c for c in macro_returns.columns if c in bucket_cols]
        macro = macro_returns[macro_cols].copy() if macro_cols else pd.DataFrame(index=macro_returns.index)
        macro = macro.reindex(features.index if not features.empty else ret.index).ffill().bfill().fillna(0.0)
        if not macro.empty:
            if features.empty:
                features = macro
            else:
                features = features.join(macro, how="outer")
            pca_feats = compute_pca_features(macro)
            if not pca_feats.empty:
                features = features.join(pca_feats, how="outer")

    features = features.reindex(ret.index).ffill().bfill().fillna(0.0)

    if features.empty:
        drift = ret.mean()
        vol = ret.std(ddof=1)
        last_price = log_price.iloc[-1]
        future_times = pd.bdate_range(log_price.index[-1] + pd.Timedelta(days=1), periods=steps)
        growth = np.linspace(1, steps, steps)
        yhat = last_price + drift * growth
        stds = vol * np.sqrt(growth)
        df = pd.DataFrame(
            {
                "yhat": np.exp(yhat),
                "yhat_lower": np.exp(yhat - 1.28 * stds),
                "yhat_upper": np.exp(yhat + 1.28 * stds),
            },
            index=future_times,
        )
        return df

    X = features.to_numpy()
    y = ret.to_numpy()
    ones = np.ones((len(X), 1))
    X_design = np.hstack([ones, X])
    try:
        beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    except np.linalg.LinAlgError:
        return pd.DataFrame()

    fitted = X_design @ beta
    residuals = y - fitted
    sigma = residuals.std(ddof=X_design.shape[1]) if len(residuals) > X_design.shape[1] else residuals.std(ddof=1)
    if not np.isfinite(sigma) or sigma == 0:
        sigma = ret.std(ddof=1)

    last_macro = features.iloc[-1].to_numpy()
    intercept = beta[0]
    betas = beta[1:]
    future_returns = np.full(steps, intercept + last_macro @ betas, dtype=float)

    last_log_price = log_price.iloc[-1]
    cumulative = np.cumsum(future_returns)
    future_index = pd.bdate_range(log_price.index[-1] + pd.Timedelta(days=1), periods=steps)
    yhat_log = last_log_price + cumulative
    horizon = np.arange(1, steps + 1, dtype=float)
    stds = sigma * np.sqrt(np.maximum(horizon, 1.0))
    df = pd.DataFrame(
        {
            "yhat": np.exp(yhat_log),
            "yhat_lower": np.exp(yhat_log - 1.28 * stds),
            "yhat_upper": np.exp(yhat_log + 1.28 * stds),
        },
        index=future_index,
    )
    return df


def prophet_forecast(price_series: pd.Series, steps: int) -> pd.DataFrame:
    if Prophet is None:
        return pd.DataFrame()
    clean = price_series.dropna()
    if clean.empty or len(clean) < PROPHET_MIN_HISTORY:
        return pd.DataFrame()

    df = pd.DataFrame({"ds": clean.index, "y": clean.values}, copy=False)
    model = Prophet(
        interval_width=0.8,
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
    )
    try:
        model.fit(df)
    except Exception:
        return pd.DataFrame()

    future = model.make_future_dataframe(periods=steps, freq="B", include_history=False)
    forecast = model.predict(future)
    if forecast.empty:
        return pd.DataFrame()
    subset = forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]]
    subset.index = pd.DatetimeIndex(subset.index).tz_localize(None)
    return subset


def blend_forecasts(
    forecasts: Iterable[pd.DataFrame],
    weights: Optional[Iterable[float]] = None,
) -> pd.DataFrame:
    valid = [f for f in forecasts if f is not None and not f.empty]
    if not valid:
        return pd.DataFrame()
    if weights is None:
        weights = [1.0 / len(valid)] * len(valid)
    else:
        weights = list(weights)
        if len(weights) != len(valid):
            weights = [1.0 / len(valid)] * len(valid)
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    combined_index = valid[0].index
    for f in valid[1:]:
        combined_index = combined_index.union(f.index)
    blended = pd.DataFrame(index=combined_index, columns=["yhat", "yhat_lower", "yhat_upper"], dtype=float)
    for weight, forecast in zip(weights, valid):
        aligned = forecast.reindex(combined_index).ffill()
        blended = blended.add(aligned * weight, fill_value=0.0)
    return blended.dropna()


def build_asset_forecast(
    ticker: str,
    price_series: pd.Series,
    macro_returns: pd.DataFrame,
    bucket: str,
    steps: int,
) -> pd.DataFrame:
    if price_series is None or price_series.empty:
        return pd.DataFrame()

    macro_subset = pd.DataFrame()
    if macro_returns is not None and not macro_returns.empty:
        bucket_cols = get_bucket_macro_columns(bucket)
        macro_subset = macro_returns[macro_returns.columns.intersection(bucket_cols)].copy()

    prophet_fc = prophet_forecast(price_series, steps)
    sarimax_fc = forecast_with_sarimax(ticker, price_series, macro_subset, bucket, steps)
    factor_fc = factor_model_forecast(price_series, macro_subset, bucket, steps)
    linear_fc = linear_forecast(price_series, steps)

    components = [prophet_fc, sarimax_fc, factor_fc, linear_fc]
    valid = [c for c in components if c is not None and not c.empty]
    if not valid:
        return pd.DataFrame()

    weights = None
    if Prophet is not None and not prophet_fc.empty:
        if not sarimax_fc.empty and not factor_fc.empty and not linear_fc.empty:
            weights = [0.4, 0.25, 0.2, 0.15]
        elif not sarimax_fc.empty and not linear_fc.empty:
            weights = [0.55, 0.30, 0.15]
        elif not sarimax_fc.empty and not factor_fc.empty:
            weights = [0.5, 0.3, 0.2]
        elif not linear_fc.empty and not factor_fc.empty:
            weights = [0.55, 0.25, 0.20]
        elif not sarimax_fc.empty:
            weights = [0.65, 0.35]
        elif not factor_fc.empty:
            weights = [0.65, 0.35]
        elif not linear_fc.empty:
            weights = [0.7, 0.3]
    elif not sarimax_fc.empty and not factor_fc.empty and not linear_fc.empty:
        weights = [0.45, 0.30, 0.25]
    elif not sarimax_fc.empty and not factor_fc.empty:
        weights = [0.6, 0.4]
    elif not sarimax_fc.empty and not linear_fc.empty:
        weights = [0.65, 0.35]
    elif not factor_fc.empty and not linear_fc.empty:
        weights = [0.6, 0.4]

    if weights is None and len(valid) > 1:
        weights = [1.0 / len(valid)] * len(valid)

    return blend_forecasts(valid, weights=weights)


def backtest_ticker(
    ticker: str,
    price_series: pd.Series,
    macro_returns: pd.DataFrame,
    bucket: str,
    default_steps: int,
    window: int,
    rolls: int,
) -> Optional[Dict[str, float]]:
    price = price_series.dropna()
    steps = BUCKET_FORECAST_HORIZON.get(bucket, default_steps)
    if price.empty or len(price) < window + steps + 5:
        return None

    results = []
    for end in range(window, min(len(price) - steps, window + rolls)):
        train = price.iloc[:end]
        actual = price.iloc[end : end + steps]
        if actual.empty:
            continue

        train_macro = (
            macro_returns.loc[: train.index[-1]] if macro_returns is not None and not macro_returns.empty else None
        )
        if train_macro is not None:
            bucket_cols = get_bucket_macro_columns(bucket)
            train_macro = train_macro[train_macro.columns.intersection(bucket_cols)]

        sarimax_fc = forecast_with_sarimax(ticker, train, train_macro, bucket, steps)
        factor_fc = factor_model_forecast(train, train_macro, bucket, steps)
        forecast = blend_forecasts([sarimax_fc, factor_fc])
        if forecast.empty:
            continue
        prediction = forecast.iloc[0]["yhat"]
        actual_price = actual.iloc[0]
        prev_price = train.iloc[-1]
        mae = abs(prediction - actual_price)
        direction_hit = np.sign(prediction - prev_price) == np.sign(actual_price - prev_price)
        results.append(
            {
                "mae": mae,
                "direction_hit": float(direction_hit),
            }
        )

    if not results:
        return None

    mae = np.mean([r["mae"] for r in results])
    direction = np.mean([r["direction_hit"] for r in results])
    return {"mae": mae, "direction_hit": direction, "trials": len(results)}


def run_backtests(
    prices: Dict[str, pd.Series],
    macro_returns: pd.DataFrame,
    default_steps: int,
    window: int,
    rolls: int,
) -> None:
    logger.info("Iniciando backtest (ventana=%s, horizonte_base=%s, iteraciones=%s)", window, default_steps, rolls)
    summary = []
    for ticker, series in prices.items():
        bucket = tickers.get(ticker, {}).get("bucket", "other")
        stats = backtest_ticker(ticker, series, macro_returns, bucket, default_steps, window, rolls)
        if stats is None:
            logger.warning("Backtest sin resultados para %s", ticker)
            continue
        summary.append((ticker, stats))
        logger.info(
            "[%s] MAE=%.4f, acierto direccional=%.0f%% (%s pruebas)",
            ticker,
            stats["mae"],
            stats["direction_hit"] * 100,
            stats["trials"],
        )
    if summary:
        mean_mae = np.mean([s[1]["mae"] for s in summary])
        mean_dir = np.mean([s[1]["direction_hit"] for s in summary])
        logger.info("Backtest global -> MAE medio=%.4f, acierto direccional=%.0f%%", mean_mae, mean_dir * 100)


def forecast_with_sarimax(
    ticker: str,
    price_series: pd.Series,
    macro_returns: pd.DataFrame,
    bucket: str,
    steps: int,
) -> pd.DataFrame:
    if SARIMAX is None:
        return linear_forecast(price_series, steps)

    price = price_series.dropna()
    if price.empty or len(price) < 30:
        return linear_forecast(price_series, steps)

    log_price = np.log(price)
    log_price = log_price.replace([np.inf, -np.inf], np.nan).dropna()
    log_price = log_price.asfreq("B")
    log_price = log_price.ffill().dropna()
    if log_price.empty:
        return linear_forecast(price_series, steps)

    exog: pd.DataFrame | None = None
    if macro_returns is not None and not macro_returns.empty:
        bucket_cols = get_bucket_macro_columns(bucket)
        selected_cols = [c for c in macro_returns.columns if c in bucket_cols]
        if selected_cols:
            exog = macro_returns[selected_cols].reindex(log_price.index).ffill().bfill()
            if exog.isna().all().all():
                exog = None
            else:
                exog = exog.fillna(0.0)

    order_candidates = [(1, 1, 1), (1, 0, 1)]
    fitted = None
    for order in order_candidates:
        try:
            model = SARIMAX(
                log_price,
                order=order,
                exog=exog,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(method="powell", maxiter=500, disp=False)
            break
        except Exception:
            continue

    if fitted is None:
        return linear_forecast(price_series, steps)

    future_index = pd.bdate_range(log_price.index[-1] + pd.Timedelta(days=1), periods=steps)
    future_exog = None
    if exog is not None:
        last_row = exog.iloc[-1]
        future_exog = pd.DataFrame(
            np.tile(last_row.to_numpy(), (steps, 1)),
            index=future_index,
            columns=exog.columns,
        )

    forecast_res = fitted.get_forecast(steps=steps, exog=future_exog)
    mean = np.exp(forecast_res.predicted_mean)
    conf_int = forecast_res.conf_int(alpha=0.2)
    lower = np.exp(conf_int.iloc[:, 0])
    upper = np.exp(conf_int.iloc[:, 1])
    df = pd.DataFrame(
        {
            "yhat": mean,
            "yhat_lower": lower,
            "yhat_upper": upper,
        },
        index=future_index,
    )
    return df


def generate_candlestick_chart(
    ticker: str,
    chart_df: pd.DataFrame,
    macro_returns: pd.DataFrame,
    output_dir: Path,
    bucket: str,
    steps: int,
    *,
    return_fig: bool = False,
) -> Path | go.Figure | None:
    if go is None:
        return None
    if chart_df is None or chart_df.empty:
        return None
    required_cols = {"Open", "High", "Low", "Close"}
    if not required_cols.issubset(chart_df.columns):
        return None

    close_series = chart_df["Close"]
    display_df = chart_df.tail(CANDLE_LOOKBACK)
    forecast = build_asset_forecast(ticker, close_series, macro_returns, bucket, steps=steps)
    if forecast.empty:
        logger.warning("Sin forecast para %s", ticker)

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=display_df.index,
                open=display_df["Open"],
                high=display_df["High"],
                low=display_df["Low"],
                close=display_df["Close"],
                name="Precio USD",
            )
        ]
    )

    if not forecast.empty:
        fig.add_trace(
            go.Scatter(
                x=list(forecast.index) + list(forecast.index[::-1]),
                y=list(forecast["yhat_upper"]) + list(forecast["yhat_lower"][::-1]),
                fill="toself",
                fillcolor="rgba(255, 127, 14, 0.15)",
                line=dict(color="rgba(255,127,14,0)"),
                hoverinfo="skip",
                name="Intervalo 80%",
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast["yhat"],
                mode="lines+markers",
                name="Proyeccion SARIMAX",
                line=dict(color="#ff7f0e"),
            )
        )

    title = f"{ticker} - {tickers[ticker]['name']} (USD)"
    fig.update_layout(
        title=title,
        yaxis_title="Precio (USD)",
        xaxis_title="Fecha",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    if return_fig:
        return fig

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{ticker}_candlestick.html"
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def run_analysis(
    horizons: Optional[Iterable[int]] = None,
    *,
    cash: Optional[float] = None,
    positions_override: Optional[Dict[str, Dict[str, Any]]] = None,
    return_figures: bool = False,
) -> Dict[str, Any]:
    base_horizon = FORECAST_HORIZON_DEFAULT
    if horizons:
        horizon_set = {max(1, int(h)) for h in horizons}
        if horizon_set:
            base_horizon = max(horizon_set)
            sorted_horizons = sorted(horizon_set)
        else:
            sorted_horizons = [base_horizon]
    else:
        sorted_horizons = sorted(set(DEFAULT_HORIZONS + [base_horizon]))
    cash_available = cash if cash is not None else AVAILABLE_CASH_USD
    primary_horizon = sorted_horizons[-1]

    positions_source = positions_override if positions_override is not None else USER_POSITIONS

    fx_rates = load_fx()
    macro_returns = load_macro_features()
    history_store.clear()
    ticker_meta.clear()
    metrics_data = {ticker: metrics(ticker, fx_rates) for ticker in tickers}
    metrics_df = pd.DataFrame(metrics_data).T
    metrics_df["bucket"] = [tickers[t]["bucket"] for t in metrics_df.index]
    missing_mask = metrics_df["price_usd"].isna()
    missing_list = ", ".join(metrics_df.index[missing_mask]) if missing_mask.any() else ""

    normalized_positions = {
        ticker: normalize_position_data(ticker, pos, fx_rates)
        for ticker, pos in positions_source.items()
        if ticker in metrics_df.index
    }

    analysis_results: Dict[int, pd.DataFrame] = {}
    numeric_types = (int, float, np.floating)
    combined_position_view: list[Dict[str, Any]] = []

    for horizon in sorted_horizons:
        analysis_rows: list[Dict[str, Any]] = []
        for ticker in metrics_df.index:
            chart_df = history_store.get(ticker)
            close_series = pd.Series(dtype=float)
            if chart_df is not None and not chart_df.empty and "Close" in chart_df:
                close_series = chart_df["Close"].dropna()
            bucket = tickers[ticker]["bucket"]
            forecast = build_asset_forecast(
                ticker,
                close_series,
                macro_returns,
                bucket,
                steps=horizon,
            )
            analysis = summarize_recommendation(
                ticker,
                metrics_df.loc[ticker],
                forecast,
                normalized_positions.get(ticker),
                horizon,
            )
            analysis["forecast_horizon"] = horizon
            analysis_rows.append(analysis)
        analysis_df = pd.DataFrame(analysis_rows).set_index("ticker") if analysis_rows else pd.DataFrame()
        analysis_results[horizon] = analysis_df

        if not analysis_df.empty:
            portfolio_view = analysis_df[analysis_df["units"] > 0].copy()
            if not portfolio_view.empty:
                for ticker in normalized_positions.keys():
                    if ticker in portfolio_view.index:
                        combined_position_view.append(
                            {
                                "ticker": ticker,
                                "horizon": horizon,
                                "signal": portfolio_view.loc[ticker, "signal"],
                                "expected_return": portfolio_view.loc[ticker, "expected_return"],
                                "momentum_proj": portfolio_view.loc[ticker, "momentum_proj"],
                            }
                        )

    summary_tables: Optional[Dict[str, pd.DataFrame]] = None
    if combined_position_view:
        summary_df = pd.DataFrame(combined_position_view)
        signal_table = summary_df.pivot_table(index="ticker", columns="horizon", values="signal", aggfunc="first")
        expected_table = summary_df.pivot_table(index="ticker", columns="horizon", values="expected_return", aggfunc="first")
        momentum_table = summary_df.pivot_table(index="ticker", columns="horizon", values="momentum_proj", aggfunc="first")
        expected_table = expected_table.apply(lambda col: col.map(lambda x: pct_str(x) if pd.notna(x) else "n/a"))
        momentum_table = momentum_table.apply(lambda col: col.map(lambda x: pct_str(x) if pd.notna(x) else "n/a"))
        summary_tables = {
            "signals": signal_table,
            "expected": expected_table,
            "momentum": momentum_table,
        }

    chart_outputs: Dict[str, Any] | list[Path] | None
    if return_figures:
        chart_outputs = {}
        if go is not None:
            for ticker in metrics_df.index:
                chart_df = history_store.get(ticker)
                bucket = tickers.get(ticker, {}).get("bucket", "other")
                if chart_df is None or chart_df.empty:
                    continue
                fig = generate_candlestick_chart(
                    ticker,
                    chart_df,
                    macro_returns,
                    CHART_OUTPUT_DIR,
                    bucket=bucket,
                    steps=BUCKET_FORECAST_HORIZON.get(bucket, primary_horizon),
                    return_fig=True,
                )
                if fig is not None:
                    chart_outputs[ticker] = fig
    else:
        generated_paths: list[Path] = []
        if go is not None:
            for ticker in metrics_df.index:
                chart_df = history_store.get(ticker)
                bucket = tickers.get(ticker, {}).get("bucket", "other")
                if chart_df is None or chart_df.empty:
                    continue
                path = generate_candlestick_chart(
                    ticker,
                    chart_df,
                    macro_returns,
                    CHART_OUTPUT_DIR,
                    bucket=bucket,
                    steps=BUCKET_FORECAST_HORIZON.get(bucket, primary_horizon),
                )
                if path:
                    generated_paths.append(path)
        chart_outputs = generated_paths

    return {
        "metrics": metrics_df,
        "analysis": analysis_results,
        "summary": summary_tables,
        "cash_available": cash_available,
        "normalized_positions": normalized_positions,
        "horizons": sorted_horizons,
        "primary_horizon": primary_horizon,
        "chart_outputs": chart_outputs,
        "missing_tickers": missing_list,
        "macro_returns": macro_returns,
    }


def main(args: argparse.Namespace) -> None:
    result = run_analysis(
        horizons=args.horizons if args.horizons else [args.horizon],
        cash=args.cash,
        return_figures=False,
    )

    metrics_df = result["metrics"]
    analysis_results = result["analysis"]
    summary_tables = result["summary"]
    cash_available = result["cash_available"]
    horizons = result["horizons"]
    primary_horizon = result["primary_horizon"]
    chart_outputs = result["chart_outputs"]
    missing_list = result["missing_tickers"]

    numeric_types = (int, float, np.floating)

    if missing_list:
        print(f"ADVERTENCIA: sin precios recientes -> {missing_list} (se marcan en pausa)")

    print("\n=== MOMENTUM & DD30 ===")
    momentum_cols = ["price_usd", "currency", "r1d", "r1w", "r1m", "dd30", "bucket"]
    out = metrics_df[momentum_cols].copy()
    out["price_usd"] = out["price_usd"].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "n/a")
    for col in ["r1d", "r1w", "r1m", "dd30"]:
        out[col] = out[col].map(lambda x: pct_str(x) if isinstance(x, numeric_types) else x)
    print(out)

    for horizon in horizons:
        analysis_df = analysis_results.get(horizon, pd.DataFrame())
        if analysis_df.empty:
            print(f"\n=== Analisis segun forecast ({horizon} dias habiles) ===")
            print("Sin datos suficientes para este horizonte.")
            continue

        display_cols = [
            "name",
            "bucket",
            "price_usd",
            "expected_return",
            "momentum_proj",
            "signal",
            "momentum_1w",
            "momentum_1m",
            "note",
        ]
        temp = analysis_df[display_cols].copy()
        temp["price_usd"] = temp["price_usd"].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "n/a")
        temp["expected_return"] = temp["expected_return"].map(
            lambda x: pct_str(x) if isinstance(x, numeric_types) else "n/a"
        )
        temp["momentum_proj"] = temp["momentum_proj"].map(
            lambda x: pct_str(x) if isinstance(x, numeric_types) else "n/a"
        )
        temp["momentum_1w"] = temp["momentum_1w"].map(
            lambda x: pct_str(x) if isinstance(x, numeric_types) else "n/a"
        )
        temp["momentum_1m"] = temp["momentum_1m"].map(
            lambda x: pct_str(x) if isinstance(x, numeric_types) else "n/a"
        )
        temp["note"] = temp["note"].replace("", "-")
        print(f"\n=== Analisis segun forecast ({horizon} dias habiles) ===")
        print(temp)

        portfolio_view = analysis_df[analysis_df.get("units", 0) > 0].copy()
        if not portfolio_view.empty:
            portfolio_view = portfolio_view.sort_values("expected_return", ascending=False)
            portfolio_view["expected_return_pct"] = portfolio_view["expected_return"].map(
                lambda x: pct_str(x) if isinstance(x, numeric_types) else "n/a"
            )
            portfolio_view["momentum_proj_pct"] = portfolio_view["momentum_proj"].map(
                lambda x: pct_str(x) if isinstance(x, numeric_types) else "n/a"
            )
            portfolio_view["avg_cost_usd"] = portfolio_view["avg_cost_usd"].map(
                lambda x: f"{x:,.2f}" if isinstance(x, numeric_types) else "n/a"
            )
            portfolio_view["price_usd"] = portfolio_view["price_usd"].map(
                lambda x: f"{x:,.2f}" if isinstance(x, numeric_types) else "n/a"
            )
            portfolio_view["note"] = portfolio_view["note"].replace("", "-")
            print(f"\n=== Posiciones actuales: sugerencias ({horizon} dias) ===")
            print(
                portfolio_view[
                    [
                        "name",
                        "units",
                        "position_currency",
                        "avg_cost_usd",
                        "price_usd",
                        "expected_return_pct",
                        "momentum_proj_pct",
                        "signal",
                        "rationale",
                        "note",
                    ]
                ]
            )
        else:
            print(f"\nNo hay posiciones registradas en USER_POSITIONS para {horizon} dias.")

        opportunity_view = analysis_df[analysis_df.get("units", 0) == 0].copy()
        opportunity_view = opportunity_view.dropna(subset=["expected_return"])
        opportunity_view = opportunity_view.sort_values("expected_return", ascending=False)
        if not opportunity_view.empty and cash_available > 0:
            opportunity_view["expected_return_pct"] = opportunity_view["expected_return"].map(
                lambda x: pct_str(x) if isinstance(x, numeric_types) else "n/a"
            )
            opportunity_view["price_usd"] = opportunity_view["price_usd"].map(
                lambda x: f"{x:,.2f}" if isinstance(x, numeric_types) else "n/a"
            )
            opportunity_view["note"] = opportunity_view["note"].replace("", "-")
            print(f"\n=== Oportunidades con efectivo disponible (USD {cash_available:,.2f}) - {horizon} dias ===")
            print(
                opportunity_view[
                    [
                        "name",
                        "bucket",
                        "price_usd",
                        "expected_return_pct",
                        "signal",
                        "rationale",
                        "note",
                    ]
                ].head(10)
            )
        elif cash_available > 0:
            print(
                f"\nEfectivo disponible USD {cash_available:,.2f} pero sin forecasts accionables (horizonte {horizon})."
            )

    if summary_tables:
        print("\n=== Resumen de senales por horizonte ===")
        print(summary_tables["signals"])
        print("\nRetorno esperado (%) por horizonte:")
        print(summary_tables["expected"])
        print("\nMomentum proyectado (%) por horizonte:")
        print(summary_tables["momentum"])
    else:
        print("\nNo hay posiciones registradas para resumir por horizonte.")

    if go is None:
        print("\nNota: instala plotly (`pip install plotly`) para generar graficos de velas y proyecciones.")
    else:
        if isinstance(chart_outputs, dict):
            if chart_outputs:
                print("\nGraficos generados (uso interactivo): disponibles en memoria.")
            else:
                print("\nNo se generaron graficos (informacion de precios insuficiente).")
        else:
            if chart_outputs:
                print("\nArchivos de graficos generados (HTML interactivo):")
                for path in chart_outputs:
                    print(f" - {path}")
            else:
                print("\nNo se generaron graficos (informacion de precios insuficiente).")

    if args.backtest:
        price_map = {}
        for ticker, data in history_store.items():
            if data is not None and not data.empty and "Close" in data:
                price_map[ticker] = data["Close"]
        run_backtests(
            price_map,
            result["macro_returns"],
            default_steps=primary_horizon,
            window=args.backtest_window,
            rolls=args.backtest_rolls,
        )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seguimiento y proyecciones para ETFs")
    parser.add_argument(
        "--horizon",
        type=int,
        default=FORECAST_HORIZON_DEFAULT,
        help="Horizonte de proyeccion en dias habiles",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        help="Lista de horizontes en dias habiles para generar proyecciones simultaneas",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=None,
        help="Liquidez disponible en USD para sugerencias (override del valor en el codigo)",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Ejecuta backtest rapido de los modelos",
    )
    parser.add_argument(
        "--backtest-window",
        type=int,
        default=BACKTEST_WINDOW,
        help="Tamano de ventana de entrenamiento para backtest",
    )
    parser.add_argument(
        "--backtest-rolls",
        type=int,
        default=BACKTEST_ROLLS,
        help="Cantidad de iteraciones en backtest (desplazamientos)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de logging para la ejecucion",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    logger.setLevel(getattr(logging, cli_args.log_level))
    logger.info("Inicio de ejecucion a las %s", datetime.now().isoformat())
    main(cli_args)

