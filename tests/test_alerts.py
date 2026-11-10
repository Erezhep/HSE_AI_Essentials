"""Правила алертов из ТЗ (раздел 3.2.2) на маленьких искусственных примерах."""
import pandas as pd

from src.alerts import compute_alerts

AS_OF = pd.Timestamp("2026-06-30")


def cards(org, dates, category="СИЗ"):
    return pd.DataFrame({"org_id": org, "date": pd.to_datetime(dates), "category": category, "is_bad": True})


def history(org, per_month=3):
    """Обычный уровень: per_month нарушений в каждом месяце предыдущего года."""
    days = [d for m in pd.date_range("2025-05-01", "2026-05-01", freq="MS")
            for d in pd.date_range(m, periods=per_month, freq="7D")]
    return cards(org, days, category="Транспорт")


def test_critical_when_count_exceeds_double_threshold():
    now = cards("ORG-X", pd.date_range("2026-06-05", periods=8, freq="3D"), category="Экология")
    alerts = compute_alerts(pd.concat([history("ORG-X"), now]), AS_OF)
    assert "Критический" in set(alerts["level"])


def test_high_when_same_category_more_than_3_times():
    now = cards("ORG-Y", pd.date_range("2026-06-10", periods=4, freq="2D"))
    alerts = compute_alerts(now, AS_OF)
    assert (alerts["level"] == "Высокий").any()


def test_exactly_3_same_category_is_not_high():
    now = cards("ORG-Y", pd.date_range("2026-06-10", periods=3, freq="2D"))
    alerts = compute_alerts(now, AS_OF)
    assert not (alerts["title"].str.startswith("Повторяющиеся")).any()


def test_medium_on_growth_vs_last_year_and_low_on_improvement():
    grow = pd.concat([cards("ORG-G", pd.date_range("2025-06-05", periods=5, freq="4D")),
                      cards("ORG-G", pd.date_range("2026-06-05", periods=7, freq="4D"), category="LOTO")])
    fall = pd.concat([cards("ORG-F", pd.date_range("2025-06-05", periods=8, freq="3D")),
                      cards("ORG-F", pd.date_range("2026-06-05", periods=2, freq="5D"))])
    levels = compute_alerts(pd.concat([grow, fall]), AS_OF).groupby("org_id")["level"].apply(set)
    assert "Средний" in levels["ORG-G"]
    assert "Низкий" in levels["ORG-F"]


def test_small_numbers_do_not_trigger_trend_alerts():
    tiny = pd.concat([cards("ORG-S", ["2025-06-10"]), cards("ORG-S", ["2026-06-10", "2026-06-12"])])
    assert compute_alerts(tiny, AS_OF).empty
