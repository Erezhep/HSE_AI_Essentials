"""
Прогноз числа происшествий на 3, 6 и 12 месяцев (F-06).

Одна задача: по месячному ряду предсказать будущие месяцы с интервалом.

Финальная модель — регрессия «тренд + сезонная волна»:
    y(t) = b0 + b1·t + a·sin(2πm/12) + c·cos(2πm/12) + шум
где t — номер месяца (тренд), m — календарный месяц (сезонность).
Сезонность описывается двумя параметрами вместо 11 (как у помесячных dummy или SARIMA),
поэтому модель не переобучается на коротком ряду (36 точек).
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


def future_index(series: pd.Series, horizon: int) -> pd.DatetimeIndex:
    start = series.index[-1] + pd.offsets.MonthBegin(1)
    return pd.date_range(start, periods=horizon, freq="MS")


# ------------------------------------------------------------------ baseline-модели
def seasonal_naive(train: pd.Series, horizon: int) -> np.ndarray:
    """«Будет как в тот же месяц прошлого года»."""
    return train.values[-12:][np.arange(horizon) % 12]


def mean_last_year(train: pd.Series, horizon: int) -> np.ndarray:
    """«Будет как в среднем за последний год»."""
    return np.repeat(train.values[-12:].mean(), horizon)


def holt_winters(train: pd.Series, horizon: int) -> np.ndarray:
    with warnings.catch_warnings():          # предупреждения сходимости на коротком ряду
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
    return model.forecast(horizon).values


def sarima(train: pd.Series, horizon: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(0, 1, 0, 12)).fit(disp=False)
    return model.forecast(horizon).values


# ------------------------------------------------------------------ финальная модель
class TrendSeasonModel:
    """Линейный тренд + K сезонных гармоник (OLS)."""

    def __init__(self, harmonics: int = 1, trend: bool = True):
        self.harmonics = harmonics
        self.trend = trend

    def _design(self, index: pd.DatetimeIndex) -> np.ndarray:
        t = (index.year - self.t0.year) * 12 + (index.month - self.t0.month)
        cols = [np.ones(len(index))]
        if self.trend:
            cols.append(np.asarray(t, float))
        for k in range(1, self.harmonics + 1):
            cols += [np.sin(2 * np.pi * k * index.month / 12), np.cos(2 * np.pi * k * index.month / 12)]
        return np.column_stack(cols)

    def fit(self, series: pd.Series) -> "TrendSeasonModel":
        self.t0 = series.index[0]
        self.last = series
        self.result = sm.OLS(series.values.astype(float), self._design(series.index)).fit()
        return self

    def predict(self, horizon: int, level: float = 0.9) -> pd.DataFrame:
        """Прогноз и интервал предсказания (по умолчанию 90%)."""
        idx = future_index(self.last, horizon)
        pred = self.result.get_prediction(self._design(idx)).summary_frame(alpha=1 - level)
        out = pd.DataFrame({"forecast": pred["mean"].values,
                            "lower": pred["obs_ci_lower"].values,
                            "upper": pred["obs_ci_upper"].values}, index=idx)
        return out.clip(lower=0)  # число происшествий не бывает отрицательным

    def predict_total(self, horizon: int, level: float = 0.9) -> tuple[float, float, float]:
        """Сумма за горизонт и её интервал (дисперсия суммы, а не сумма интервалов)."""
        from scipy import stats
        X = self._design(future_index(self.last, horizon)).sum(axis=0)
        total = float(X @ self.result.params)
        var = float(X @ self.result.cov_params() @ X) + horizon * self.result.scale
        q = stats.t.ppf(0.5 + level / 2, self.result.df_resid)
        half = q * np.sqrt(var)
        return total, max(total - half, 0.0), total + half

    def __call__(self, train: pd.Series, horizon: int) -> np.ndarray:
        """Единый интерфейс для backtesting: f(train, horizon) -> прогноз."""
        return TrendSeasonModel(self.harmonics, self.trend).fit(train).predict(horizon)["forecast"].values


# ------------------------------------------------------------------ оценка
def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def backtest(model_fn, series: pd.Series, first_origin: int = 24, horizon: int = 6) -> dict:
    """Скользящий backtesting: обучаемся на первых N месяцах, прогнозируем следующие
    `horizon`, сдвигаем N на месяц вперёд. Возвращает средние MAE и RMSE."""
    maes, rmses = [], []
    for origin in range(first_origin, len(series) - horizon + 1):
        train, actual = series.iloc[:origin], series.iloc[origin:origin + horizon]
        pred = model_fn(train, horizon)
        maes.append(mae(actual, pred))
        rmses.append(rmse(actual, pred))
    return {"MAE": float(np.mean(maes)), "RMSE": float(np.mean(rmses)), "n_origins": len(maes)}


# ------------------------------------------------------------------ типы происшествий
def type_shares(incidents: pd.DataFrame, last_months: int = 24) -> pd.Series:
    """Доли типов за последние месяцы: прогноз общего числа раскладывается по типам."""
    start = incidents["date"].max() - pd.DateOffset(months=last_months)
    return incidents.loc[incidents["date"] > start, "type"].value_counts(normalize=True)


def forecast_by_type(total: pd.DataFrame, shares: pd.Series) -> pd.DataFrame:
    """Число и типы происшествий по месяцам прогноза."""
    return pd.DataFrame({t: total["forecast"] * s for t, s in shares.items()}, index=total.index)


def horizon_summary(model: TrendSeasonModel, shares: pd.Series) -> pd.DataFrame:
    """Сводка для ТЗ: сколько происшествий ожидать за 3, 6 и 12 месяцев."""
    rows = []
    for h in (3, 6, 12):
        total, lower, upper = model.predict_total(h)
        row = {"horizon_months": h, "total": total, "lower": lower, "upper": upper}
        row.update({t: total * s for t, s in shares.items()})
        rows.append(row)
    return pd.DataFrame(rows).set_index("horizon_months").round(1)
