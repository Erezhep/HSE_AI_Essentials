"""Сохранённые модели загружаются, дают корректный вывод и не хуже зафиксированного качества."""
import json
from pathlib import Path

import joblib

from src.nlp import classify

MODELS = Path(__file__).resolve().parents[1] / "models"


def test_metadata_has_version_and_metrics():
    meta = json.loads((MODELS / "metadata.json").read_text(encoding="utf-8"))
    assert meta["model_version"]
    assert meta["models"]["text"]["metrics"]["test_macro_f1"] > 0.8
    assert meta["models"]["forecast"]["metrics"]["test_MAE"] < 10


def test_forecast_horizons_and_interval():
    model = joblib.load(MODELS / "forecast.joblib")
    for h in (3, 6, 12):
        fc = model.predict(h)
        assert len(fc) == h
        assert (fc["lower"] <= fc["forecast"]).all() and (fc["forecast"] <= fc["upper"]).all()
        assert (fc >= 0).all().all()


def test_text_model_classes_and_explanation():
    model = joblib.load(MODELS / "text.joblib")
    res = classify(model, "Работник поскользнулся, получил перелом руки и госпитализирован в больницу.")
    assert res["type"] in {"Опасная ситуация", "Микротравма", "Несчастный случай", "Авария"}
    assert set(res["scores"]) == set(model.classes_)
    assert res["evidence"]


def test_risk_model_is_positive_and_monotonic_in_violations():
    import pandas as pd
    from src.features import FEATURES
    model = joblib.load(MODELS / "risk.joblib")
    x = pd.DataFrame([[3, 3, 10, 0.5, 1, 1, 300]], columns=FEATURES, dtype=float)
    more_violations = x.assign(bad_share_lag1=0.9)
    assert model.predict(x)[0] > 0
    assert model.predict(more_violations)[0] > model.predict(x)[0]
