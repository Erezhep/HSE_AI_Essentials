"""
Risk scoring организаций и локаций (F-12, F-13, MVP «Топ-5 зон риска»).

Одна задача: оценить, сколько происшествий ожидать в организации в следующем месяце.

Модель — Poisson-регрессия. Почему Poisson: target — это число событий (0, 1, 2, …),
а не непрерывная величина. Модель предсказывает интенсивность λ:
    λ = exp(b0 + b1·x1 + … + bk·xk)
exp(b) показывает, во сколько раз меняется риск при росте признака на 1 стандартное
отклонение. Это удобно объяснять HSE-специалисту.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURES, TARGET


def make_risk_model(alpha: float = 1.0) -> Pipeline:
    return Pipeline([("scale", StandardScaler()),
                     ("poisson", PoissonRegressor(alpha=alpha, max_iter=3000))])


def fit_risk_model(org_month: pd.DataFrame, alpha: float = 1.0) -> Pipeline:
    known = org_month[org_month[TARGET].notna()]
    return make_risk_model(alpha).fit(known[FEATURES], known[TARGET])


def risk_effects(model: Pipeline) -> pd.Series:
    """Во сколько раз меняется ожидаемое число происшествий при +1 σ признака."""
    return pd.Series(np.exp(model.named_steps["poisson"].coef_), index=FEATURES).sort_values(ascending=False)


def score_next_month(model: Pipeline, org_month_with_next: pd.DataFrame,
                     organizations: pd.DataFrame, company_total: float | None = None) -> pd.DataFrame:
    """Риск каждой организации на следующий месяц.

    company_total — прогноз общего числа происшествий на этот месяц (src/forecast.py).
    Если задан, ожидаемые значения масштабируются так, чтобы их сумма совпала с ним:
    общий уровень (тренд, сезон) задаёт модель прогноза, а распределение между
    организациями — модель риска.
    """
    nxt = org_month_with_next[org_month_with_next["split"] == "future"].copy()
    nxt["expected_incidents"] = model.predict(nxt[FEATURES])
    if company_total is not None:
        nxt["expected_incidents"] *= company_total / nxt["expected_incidents"].sum()
    nxt["per_100_workers"] = nxt["expected_incidents"] / nxt["headcount"] * 100
    nxt = nxt.merge(organizations[["org_id", "name", "kind"]], on="org_id")
    # Индекс риска 0–100: ожидаемое число относительно самой рискованной организации
    nxt["risk_index"] = (100 * nxt["expected_incidents"] / nxt["expected_incidents"].max()).round()
    nxt["risk_level"] = pd.cut(nxt["risk_index"], [-1, 33, 66, 100], labels=["Низкий", "Средний", "Высокий"])
    cols = ["org_id", "name", "kind", "month", "expected_incidents",
            "per_100_workers", "risk_index", "risk_level", "bad_share_lag1", "overdue_lag1"]
    return nxt[cols].sort_values("expected_incidents", ascending=False).reset_index(drop=True)


def score_locations(org_risk: pd.DataFrame, incidents: pd.DataFrame, months: int = 12) -> pd.DataFrame:
    """Риск локаций: ожидаемые события организации делятся по её локациям
    пропорционально истории происшествий за последние `months` месяцев."""
    start = incidents["date"].max() - pd.DateOffset(months=months)
    recent = incidents[incidents["date"] > start]
    share = recent.groupby(["org_id", "location"]).size() / recent.groupby("org_id").size()
    share = share.rename("share").reset_index()
    loc = share.merge(org_risk[["org_id", "expected_incidents"]], on="org_id")
    loc["expected_incidents"] *= loc["share"]
    return (loc.groupby("location")["expected_incidents"].sum()
               .sort_values(ascending=False).rename("expected_incidents").reset_index())


def scenario(model: Pipeline, org_month_with_next: pd.DataFrame,
             violation_reduction: float, org_ids: list[str] | None = None) -> dict:
    """Scenario modeling: как изменится ожидаемое число происшествий, если
    снизить нарушения Коргау на `violation_reduction` (0.3 = на 30%) и устранить
    все просроченные нарушения. По умолчанию — во всех организациях."""
    nxt = org_month_with_next[org_month_with_next["split"] == "future"].copy()
    before = model.predict(nxt[FEATURES]).sum()
    target = nxt["org_id"].isin(org_ids) if org_ids else np.ones(len(nxt), bool)
    for col in ("bad_lag1", "bad_share_lag1"):
        nxt.loc[target, col] *= (1 - violation_reduction)
    nxt.loc[target, "overdue_lag1"] = 0
    after = model.predict(nxt[FEATURES]).sum()
    return {"expected_before": float(before), "expected_after": float(after),
            "relative_reduction": float(1 - after / before)}
