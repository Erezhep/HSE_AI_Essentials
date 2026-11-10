"""
Загрузка и очистка данных.

Один модуль — одна задача: превратить «сырые» CSV из HSE-системы в чистые таблицы.
Этими же функциями пользуются notebooks, обучение моделей и API, поэтому очистка
везде одинаковая.
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

INCIDENT_TYPES = ["Опасная ситуация", "Микротравма", "Несчастный случай", "Авария"]
KORGAU_CATEGORIES = ["Работа на высоте", "СИЗ", "LOTO", "Газоопасные работы",
                     "Пожарная безопасность", "Электробезопасность",
                     "Грузоподъемные операции", "Транспорт",
                     "Порядок на рабочем месте", "Экология"]

# Разные написания одной категории -> каноническое название
TYPE_FIX = {"микротравма": "Микротравма", "микро-травма": "Микротравма",
            "опасная ситуация": "Опасная ситуация", "нс": "Несчастный случай",
            "несчастный случай": "Несчастный случай", "авария": "Авария"}
CATEGORY_FIX = {"сиз": "СИЗ", "loto": "LOTO", "высотные работы": "Работа на высоте",
                "транспорт": "Транспорт"}


def load_raw(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Читает два исходных файла без изменений."""
    incidents = pd.read_csv(data_dir / "incidents.csv")
    korgau = pd.read_csv(data_dir / "korgau_cards.csv")
    return incidents, korgau


def load_reference(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Справочники: 'organizations' или 'incident_types'."""
    return pd.read_csv(data_dir / f"{name}.csv")


def parse_dates(s: pd.Series) -> pd.Series:
    """В данных два формата: '2024-01-31 10:15:00' (основной) и '31.01.2024 10:15'."""
    iso = pd.to_datetime(s, format="ISO8601", errors="coerce")
    ru_time = pd.to_datetime(s, format="%d.%m.%Y %H:%M", errors="coerce")
    ru_date = pd.to_datetime(s, format="%d.%m.%Y", errors="coerce")
    return iso.fillna(ru_time).fillna(ru_date)


def _normalize(s: pd.Series, fix: dict) -> pd.Series:
    """Убирает пробелы и приводит синонимы к одному написанию."""
    s = s.astype(str).str.strip()
    return s.map(lambda v: fix.get(v.lower(), v))


def _drop_duplicates(df: pd.DataFrame, key_cols: list[str]) -> tuple[pd.DataFrame, int, int]:
    """Два вида дублей:
    1) полные копии строки;
    2) повторная регистрация того же события под новым id (совпадает всё, кроме id).
    """
    n0 = len(df)
    df = df.drop_duplicates()
    exact = n0 - len(df)
    n1 = len(df)
    df = df.drop_duplicates(subset=key_cols)
    return df, exact, n1 - len(df)


def clean_incidents(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Очистка журнала происшествий. Возвращает чистую таблицу и отчёт «до/после»."""
    df = raw.copy()
    report = {"rows_before": len(df)}

    report["bad_type_spelling"] = int((~df["type"].isin(INCIDENT_TYPES)).sum())
    df["type"] = _normalize(df["type"], TYPE_FIX)

    report["org_id_with_spaces"] = int((df["org_id"] != df["org_id"].str.strip()).sum())
    df["org_id"] = df["org_id"].str.strip()

    report["dates_in_ru_format"] = int(df["date"].str.contains(r"^\d{2}\.").sum())
    df["date"] = parse_dates(df["date"])
    report["unparsed_dates"] = int(df["date"].isna().sum())

    df, exact, repeated = _drop_duplicates(
        df, ["date", "type", "org_id", "location", "description"])
    report["exact_duplicates"], report["repeated_registrations"] = exact, repeated

    # Пустая причина — не ошибка: расследование ещё не завершено. Удалять такие строки
    # нельзя (потеряем свежие события), поэтому отмечаем флагом.
    df["cause_known"] = df["cause"].notna()
    report["missing_cause"] = int((~df["cause_known"]).sum())
    report["missing_cause_in_closed"] = int(((~df["cause_known"]) & (df["status"] == "Закрытое")).sum())

    df = df.sort_values("date").reset_index(drop=True)
    report["rows_after"] = len(df)
    return df, report


def clean_korgau(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Очистка Карт Коргау (та же логика, что для происшествий)."""
    df = raw.copy()
    report = {"rows_before": len(df)}

    report["bad_category_spelling"] = int((~df["category"].isin(KORGAU_CATEGORIES)).sum())
    df["category"] = _normalize(df["category"], CATEGORY_FIX)

    report["org_id_with_spaces"] = int((df["org_id"] != df["org_id"].str.strip()).sum())
    df["org_id"] = df["org_id"].str.strip()

    report["dates_in_ru_format"] = int(df["date"].str.contains(r"^\d{2}\.").sum())
    df["date"] = parse_dates(df["date"])
    report["unparsed_dates"] = int(df["date"].isna().sum())

    df, exact, repeated = _drop_duplicates(
        df, ["date", "obs_type", "org_id", "category", "description", "resolved"])
    report["exact_duplicates"], report["repeated_registrations"] = exact, repeated

    df["is_bad"] = df["obs_type"].eq("Плохая практика")
    df = df.sort_values("date").reset_index(drop=True)
    report["rows_after"] = len(df)
    return df, report


def load_clean(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Главная точка входа: загрузить и очистить оба журнала."""
    inc_raw, kor_raw = load_raw(data_dir)
    incidents, _ = clean_incidents(inc_raw)
    korgau, _ = clean_korgau(kor_raw)
    return incidents, korgau


def monthly_counts(incidents: pd.DataFrame, by: str | None = None) -> pd.DataFrame | pd.Series:
    """Число происшествий по месяцам (всего или в разрезе столбца `by`)."""
    month = incidents["date"].dt.to_period("M").dt.to_timestamp()
    if by is None:
        return incidents.groupby(month).size().rename("incidents").asfreq("MS", fill_value=0)
    table = incidents.groupby([month, incidents[by]]).size().unstack(fill_value=0)
    return table.asfreq("MS", fill_value=0)
