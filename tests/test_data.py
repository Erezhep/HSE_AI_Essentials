"""Очистка данных: исправляет ровно то, что должна, и ничего не теряет."""
import pandas as pd

from src.data import INCIDENT_TYPES, KORGAU_CATEGORIES, clean_incidents, clean_korgau, load_raw, parse_dates


def test_both_date_formats_are_parsed():
    s = pd.Series(["2024-01-31 10:15:00", "31.01.2024 10:15", "31.01.2024"])
    parsed = parse_dates(s)
    assert parsed.notna().all()
    assert (parsed.dt.date == pd.Timestamp("2024-01-31").date()).all()


def test_clean_incidents_normalizes_and_deduplicates():
    raw, _ = load_raw()
    clean, report = clean_incidents(raw)
    assert set(clean["type"]) == set(INCIDENT_TYPES)
    assert clean["date"].notna().all()
    assert not clean.duplicated(subset=["date", "type", "org_id", "location", "description"]).any()
    assert (clean["org_id"] == clean["org_id"].str.strip()).all()
    assert report["rows_after"] == report["rows_before"] - report["exact_duplicates"] - report["repeated_registrations"]


def test_missing_cause_is_kept_and_only_for_open_cases():
    raw, _ = load_raw()
    clean, report = clean_incidents(raw)
    assert report["missing_cause"] > 0                      # строки не удалены
    assert report["missing_cause_in_closed"] == 0           # пропуск только у незакрытых дел
    assert (~clean["cause_known"]).sum() == report["missing_cause"]


def test_clean_korgau_categories():
    _, raw = load_raw()
    clean, _ = clean_korgau(raw)
    assert set(clean["category"]) <= set(KORGAU_CATEGORIES)
    assert clean["is_bad"].dtype == bool
