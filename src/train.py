"""
Обучение и сохранение финальных моделей (Final project → «Подготовить финальную модель»).

Запуск из корня проекта:  python -m src.train

Что делает:
1. Загружает и очищает данные (src/data.py).
2. Для каждой задачи считает метрики на test (модель обучена на train + validation).
   Это финальная честная оценка.
3. Переобучает модель на ВСЕХ данных: в работе нужна модель, видевшая самые свежие месяцы.
4. Сохраняет модели в models/*.joblib и паспорт models/metadata.json:
   версия, дата, гиперпараметры, метрики, версии библиотек, контрольные суммы данных.
Гиперпараметры выбраны в notebooks/03_model_training.ipynb (reports/a3_chosen_models.json).
"""
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import statsmodels
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_poisson_deviance

from src.data import DATA_DIR, load_clean, load_reference, monthly_counts
from src.features import FEATURES, TARGET, build_org_month, split_period
from src.forecast import TrendSeasonModel, backtest, mae, rmse
from src.nlp import make_text_model, split_texts
from src.risk import fit_risk_model

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_VERSION = "1.0.0"
CHOSEN = {"forecast": {"harmonics": 1, "trend": True},
          "risk": {"alpha": 1.0},
          "text": {"C": 3.0, "ngram_max": 1, "min_df": 1}}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def train_forecast(incidents: pd.DataFrame) -> tuple[TrendSeasonModel, dict]:
    y = monthly_counts(incidents).astype(float)
    part = split_period(pd.Series(y.index, index=y.index))
    n_trval = int((part != "test").sum())
    cfg = CHOSEN["forecast"]
    pred = TrendSeasonModel(**cfg)(y.iloc[:n_trval], len(y) - n_trval)
    metrics = {"test_MAE": round(mae(y.iloc[n_trval:], pred), 3),
               "test_RMSE": round(rmse(y.iloc[n_trval:], pred), 3),
               "backtest_MAE": round(backtest(TrendSeasonModel(**cfg), y, 24, 6)["MAE"], 3)}
    return TrendSeasonModel(**cfg).fit(y), metrics


def train_risk(incidents, korgau, organizations):
    om = build_org_month(incidents, korgau, organizations)
    trval, test = om[om["split"] != "test"], om[om["split"] == "test"]
    model = fit_risk_model(trval, **CHOSEN["risk"])
    p = np.clip(model.predict(test[FEATURES]), 1e-3, None)
    metrics = {"test_MAE": round(mean_absolute_error(test[TARGET], p), 3),
               "test_poisson_deviance": round(mean_poisson_deviance(test[TARGET], p), 3)}
    return fit_risk_model(om, **CHOSEN["risk"]), metrics


def train_text(incidents):
    train, val, test = split_texts(incidents)
    trval = pd.concat([train, val])
    model = make_text_model(**CHOSEN["text"]).fit(trval["description"], trval["type"])
    p = model.predict(test["description"])
    metrics = {"test_accuracy": round(accuracy_score(test["type"], p), 4),
               "test_macro_f1": round(f1_score(test["type"], p, average="macro"), 4)}
    final = make_text_model(**CHOSEN["text"]).fit(incidents["description"], incidents["type"])
    return final, metrics


def main() -> dict:
    np.random.seed(42)
    incidents, korgau = load_clean()
    organizations = load_reference("organizations")
    MODEL_DIR.mkdir(exist_ok=True)

    forecast_model, m_fc = train_forecast(incidents)
    risk_model, m_risk = train_risk(incidents, korgau, organizations)
    text_model, m_text = train_text(incidents)

    joblib.dump(forecast_model, MODEL_DIR / "forecast.joblib")
    joblib.dump(risk_model, MODEL_DIR / "risk.joblib")
    joblib.dump(text_model, MODEL_DIR / "text.joblib")

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": {"period": [str(incidents["date"].min().date()), str(incidents["date"].max().date())],
                 "incidents_rows": len(incidents), "korgau_rows": len(korgau),
                 "sha256": {f.name: sha256(f) for f in sorted(DATA_DIR.glob("*.csv"))}},
        "models": {
            "forecast": {"file": "forecast.joblib", "algorithm": "OLS: линейный тренд + сезонная гармоника",
                         "params": CHOSEN["forecast"], "metrics": m_fc,
                         "input": "месячный ряд числа происшествий",
                         "output": "прогноз на 1–12 мес. + 90% интервал"},
            "risk": {"file": "risk.joblib", "algorithm": "StandardScaler + PoissonRegressor",
                     "params": CHOSEN["risk"], "features": FEATURES, "metrics": m_risk,
                     "input": "лаговые признаки организации за прошлый месяц",
                     "output": "ожидаемое число происшествий в следующем месяце"},
            "text": {"file": "text.joblib", "algorithm": "TF-IDF + LinearSVC(class_weight=balanced)",
                     "params": CHOSEN["text"], "metrics": m_text,
                     "input": "текст описания происшествия (рус.)",
                     "output": "тип: Опасная ситуация / Микротравма / Несчастный случай / Авария"},
        },
        "evaluation": "метрики test: модель обучена на train+validation; сохранённая модель переобучена на всех данных",
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__, "statsmodels": statsmodels.__version__},
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    meta = main()
    print(f"Модели сохранены в {MODEL_DIR} (версия {meta['model_version']})")
    for name, info in meta["models"].items():
        print(f"  {name:9s} {info['metrics']}")
