"""
Система алертов по Картам Коргау (F-11, F-14). Правила взяты из ТЗ, раздел 3.2.2.

Одна задача: на дату `as_of` проверить последние 30 дней нарушений каждой организации.

| Уровень      | Правило ТЗ                                                   | Как считаем |
|--------------|--------------------------------------------------------------|-------------|
| Критический  | число нарушений за период > пороговое значение × 2           | порог = среднее число нарушений за 30 дней в предыдущие 12 месяцев |
| Высокий      | нарушения одного типа > 3 раз за 30 дней в одной организации | группировка по категории |
| Средний      | тренд нарушений вырос > 15% к аналогичному периоду прошлого года | те же 30 дней год назад |
| Низкий       | информационное: показатели улучшились                       | снижение > 15% к прошлому году |

Pre-alert (F-14): организация входит в топ зон риска (модель риска) И у неё растут нарушения.
"""
import pandas as pd

WINDOW_DAYS = 30
LEVEL_ORDER = {"Критический": 0, "Высокий": 1, "Средний": 2, "Низкий": 3}
LEVEL_ICON = {"Критический": "🔴", "Высокий": "🟠", "Средний": "🟡", "Низкий": "🟢"}

# Минимальное число нарушений, чтобы сравнение «в процентах» имело смысл.
# Без этого рост с 1 до 2 нарушений (+100%) давал бы ложный алерт.
MIN_COUNT_FOR_TREND = 5


def _window(bad: pd.DataFrame, end: pd.Timestamp, days: int = WINDOW_DAYS) -> pd.DataFrame:
    return bad[(bad["date"] > end - pd.Timedelta(days=days)) & (bad["date"] <= end)]


def compute_alerts(korgau: pd.DataFrame, as_of: pd.Timestamp | str | None = None,
                   risk_top_orgs: list[str] | None = None) -> pd.DataFrame:
    bad = korgau[korgau["is_bad"]]
    as_of = pd.Timestamp(as_of) if as_of is not None else korgau["date"].max()

    current = _window(bad, as_of)
    last_year = _window(bad, as_of - pd.DateOffset(years=1))
    history = bad[(bad["date"] > as_of - pd.DateOffset(years=1) - pd.Timedelta(days=WINDOW_DAYS))
                  & (bad["date"] <= as_of - pd.Timedelta(days=WINDOW_DAYS))]

    now_n = current.groupby("org_id").size()
    ly_n = last_year.groupby("org_id").size()
    # Порог: среднее число нарушений за 30 дней по истории (12 месяцев до текущего окна)
    threshold = history.groupby("org_id").size() / 12

    alerts = []
    for org, n in now_n.items():
        thr = threshold.get(org, 0)
        if thr > 0 and n > 2 * thr and n >= MIN_COUNT_FOR_TREND:
            alerts.append((org, "Критический", "Резкий рост нарушений",
                           f"{n} нарушений за 30 дней при обычном уровне {thr:.1f} (порог ×2 = {2 * thr:.1f})"))

    for (org, cat), n in current.groupby(["org_id", "category"]).size().items():
        if n > 3:
            alerts.append((org, "Высокий", f"Повторяющиеся нарушения: {cat}",
                           f"{n} нарушений категории «{cat}» за 30 дней (правило: > 3)"))

    for org in set(now_n.index) | set(ly_n.index):
        n, prev = now_n.get(org, 0), ly_n.get(org, 0)
        if max(n, prev) < MIN_COUNT_FOR_TREND or prev == 0:
            continue
        change = n / prev - 1
        if change > 0.15:
            alerts.append((org, "Средний", "Рост к прошлому году",
                           f"{prev} → {n} нарушений ({change:+.0%} к тем же 30 дням год назад)"))
        elif change < -0.15:
            alerts.append((org, "Низкий", "Улучшение показателей",
                           f"{prev} → {n} нарушений ({change:+.0%} к тем же 30 дням год назад)"))

    for org in risk_top_orgs or []:
        thr = threshold.get(org, 0)
        if now_n.get(org, 0) > thr > 0:
            alerts.append((org, "Высокий", "Pre-alert: рост нарушений в зоне риска",
                           f"организация в топе модели риска; нарушений {now_n[org]} при обычном уровне {thr:.1f}"))

    df = pd.DataFrame(alerts, columns=["org_id", "level", "title", "details"])
    df["icon"] = df["level"].map(LEVEL_ICON)
    df["as_of"] = as_of.normalize()
    return df.sort_values(["level", "org_id"], key=lambda s: s.map(LEVEL_ORDER) if s.name == "level" else s
                          ).reset_index(drop=True)
