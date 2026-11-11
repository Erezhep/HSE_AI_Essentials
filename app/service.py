"""
Сервисный слой: собирает данные, модели и аналитику в ответы для API.

Одна задача: «склеить» модули src/ между собой. HTTP здесь нет (это app/api.py),
интерфейса тоже нет (это app/dashboard.py). Поэтому логику легко тестировать.
"""
import json
from pathlib import Path

import joblib
import pandas as pd

import sys

ROOT = Path.cwd().parent  # если текущая папка — app
sys.path.insert(0, str(ROOT))


from src.alerts import compute_alerts
from src.data import load_clean, load_reference
from src.economics import economic_effect
from src.features import build_org_month
from src.forecast import forecast_by_type, horizon_summary, type_shares
from src.nlp import classify, normalize
from src.recommendations import build_recommendations
from src.risk import scenario, score_locations, score_next_month

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"


class HSEService:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.incidents, self.korgau = load_clean()
        self.orgs = load_reference("organizations")
        self.costs = load_reference("incident_types")
        self.metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        self.forecast_model = joblib.load(model_dir / "forecast.joblib")
        self.risk_model = joblib.load(model_dir / "risk.joblib")
        self.text_model = joblib.load(model_dir / "text.joblib")
        self.shares = type_shares(self.incidents)
        self.org_month = build_org_month(self.incidents, self.korgau, self.orgs, include_next=True)

    # ---------------------------------------------------------------- статистика (F-02, F-05)
    def stats(self, date_from=None, date_to=None, org_id=None, type_=None, location=None) -> dict:
        df = self.incidents
        if date_from:
            df = df[df["date"] >= pd.Timestamp(date_from)]
        if date_to:
            df = df[df["date"] < pd.Timestamp(date_to) + pd.Timedelta(days=1)]
        if org_id:
            df = df[df["org_id"] == org_id]
        if type_:
            df = df[df["type"] == type_]
        if location:
            df = df[df["location"] == location]
        month = df["date"].dt.to_period("M").astype(str)
        root = df["cause"].str.extract(r"Коренная причина: ([^.]+)")[0]
        return {
            "total": int(len(df)),
            "by_type": df["type"].value_counts().to_dict(),
            "by_month": df.groupby([month, "type"]).size().unstack(fill_value=0).reset_index(names="month").to_dict("records"),
            "by_org": df["org_id"].value_counts().to_dict(),
            "by_location": df["location"].value_counts().to_dict(),
            "by_weekday": df["date"].dt.dayofweek.value_counts().sort_index().to_dict(),
            "by_hour": df["date"].dt.hour.value_counts().sort_index().to_dict(),
            "top_root_causes": root.value_counts().head(5).to_dict(),            # F-03
            "open_cases": int((df["status"] != "Закрытое").sum()),
        }

    # ---------------------------------------------------------------- прогноз (F-06)
    def forecast(self, horizon: int = 12) -> dict:
        fc = self.forecast_model.predict(horizon)
        by_type = forecast_by_type(fc, self.shares)
        months = [{"month": i.strftime("%Y-%m"), **{k: round(float(v), 1) for k, v in row.items()},
                   "by_type": {t: round(float(by_type.loc[i, t]), 1) for t in by_type.columns}}
                  for i, row in fc.iterrows()]
        summary = horizon_summary(self.forecast_model, self.shares).reset_index().to_dict("records")
        history = self.forecast_model.last
        return {"months": months, "summary": summary,
                "history": [{"month": i.strftime("%Y-%m"), "incidents": int(v)} for i, v in history.items()]}

    # ---------------------------------------------------------------- риск (F-12, MVP топ-5)
    def risk(self) -> pd.DataFrame:
        next_month_total = float(self.forecast_model.predict(1)["forecast"].iloc[0])
        return score_next_month(self.risk_model, self.org_month, self.orgs, next_month_total)

    def risk_top(self, n: int = 5) -> dict:
        orgs = self.risk()
        locs = score_locations(orgs, self.incidents)
        orgs = orgs.assign(month=orgs["month"].dt.strftime("%Y-%m"), risk_level=orgs["risk_level"].astype(str))
        return {"month": orgs["month"].iloc[0],
                "organizations": orgs.head(n).round(3).to_dict("records"),
                "locations": locs.head(n).round(2).to_dict("records"),
                "all_organizations": orgs.round(3).to_dict("records")}

    # ---------------------------------------------------------------- алерты (F-11, F-14)
    def alerts(self, as_of: str | None = None) -> list[dict]:
        top = self.risk()["org_id"].head(5).tolist()
        df = compute_alerts(self.korgau, as_of, risk_top_orgs=top)
        df["as_of"] = df["as_of"].dt.strftime("%Y-%m-%d")
        return df.to_dict("records")

    # ---------------------------------------------------------------- рекомендации (F-04)
    def recommendations(self) -> list[dict]:
        return build_recommendations(self.risk(), self.korgau, self.incidents, self.orgs,
                                     self.forecast_model.predict(12))

    # ---------------------------------------------------------------- экономика (F-07)
    def economics(self, violation_reduction: float = 0.3, program_cost_kzt: float = 30_000_000) -> dict:
        sc = scenario(self.risk_model, self.org_month, violation_reduction)
        total, _, _ = self.forecast_model.predict_total(12)
        effect = economic_effect(self.shares * total, self.costs, sc["relative_reduction"], program_cost_kzt)
        return {"scenario": {"violation_reduction": violation_reduction, **sc}, **effect}

    # ---------------------------------------------------------------- NLP (F-01)
    def classify(self, text: str) -> dict:
        result = classify(self.text_model, normalize(text))
        result["model_version"] = self.metadata["model_version"]
        return result
