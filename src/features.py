"""
Признаки для моделей и временное разбиение.

Одна задача: из чистых журналов собрать таблицу «организация × месяц».
Все признаки строятся только из ПРОШЛЫХ месяцев (лаги), поэтому в начале месяца t
их уже можно посчитать. Это защита от утечки данных (data leakage).
"""
import pandas as pd

# Временное разбиение (для временных данных случайное перемешивание запрещено)
TRAIN_END = pd.Timestamp("2025-08-01")   # train: 2023-09 … 2025-08 (24 мес.)
VAL_END = pd.Timestamp("2026-02-01")     # val:   2025-09 … 2026-02 (6 мес.)
                                          # test:  2026-03 … 2026-08 (6 мес.)

FEATURES = ["inc_lag1", "inc_mean3", "bad_lag1", "bad_share_lag1",
            "overdue_lag1", "is_contractor", "headcount"]
TARGET = "incidents"


def split_period(month: pd.Series) -> pd.Series:
    """Метка 'train' / 'val' / 'test' по месяцу."""
    return pd.Series(
        pd.cut(month, [pd.Timestamp.min, TRAIN_END, VAL_END, pd.Timestamp.max],
               labels=["train", "val", "test"], right=True).astype(str),
        index=month.index)


def _month(dates: pd.Series) -> pd.Series:
    return dates.dt.to_period("M").dt.to_timestamp()


def build_org_month(incidents: pd.DataFrame, korgau: pd.DataFrame,
                    organizations: pd.DataFrame) -> pd.DataFrame:
    """Таблица «организация × месяц» с целевой переменной и лаговыми признаками."""
    months = pd.date_range(_month(incidents["date"]).min(), _month(incidents["date"]).max(), freq="MS")
    grid = pd.MultiIndex.from_product([organizations["org_id"], months], names=["org_id", "month"])

    inc = incidents.groupby(["org_id", _month(incidents["date"])]).size()
    kor = korgau.assign(month=_month(korgau["date"]),
                        overdue=korgau["resolved"].eq("Просрочено"))
    kor = kor.groupby(["org_id", "month"]).agg(
        korgau_total=("id", "size"), korgau_bad=("is_bad", "sum"), overdue=("overdue", "sum"))

    df = pd.DataFrame(index=grid)
    df["incidents"] = inc.reindex(grid, fill_value=0)
    df = df.join(kor).fillna(0).reset_index()
    df["bad_share"] = (df["korgau_bad"] / df["korgau_total"]).fillna(0)

    # Лаги: значения прошлых месяцев той же организации
    g = df.groupby("org_id")
    df["inc_lag1"] = g["incidents"].shift(1)
    df["inc_mean3"] = g["incidents"].transform(lambda s: s.shift(1).rolling(3).mean())
    df["bad_lag1"] = g["korgau_bad"].shift(1)
    df["bad_share_lag1"] = g["bad_share"].shift(1)
    df["overdue_lag1"] = g["overdue"].shift(1)

    orgs = organizations.set_index("org_id")
    df["is_contractor"] = df["org_id"].map(orgs["kind"].eq("Подрядчик")).astype(int)
    df["headcount"] = df["org_id"].map(orgs["headcount"])
    df["split"] = split_period(df["month"])
    return df.dropna(subset=FEATURES).reset_index(drop=True)
