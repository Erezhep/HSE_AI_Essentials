"""
Рекомендации по мерам контроля (F-04, MVP «минимум 3 рекомендации»).

Одна задача: превратить результаты моделей и найденные паттерны в конкретные меры.

Как формируется рекомендация:
1. Модель риска называет организации, где в следующем месяце ожидается больше всего событий.
2. По Картам Коргау за 90 дней находим, какая категория нарушений у неё преобладает.
3. Категории сопоставляется типовая мера контроля из справочника ниже.
4. К мере прикладывается обоснование цифрами, чтобы её можно было проверить.
Плюс общие рекомендации из паттернов EDA: подрядчики, понедельники, сезон.
"""
import pandas as pd

CONTROL_MEASURES = {
    "Работа на высоте": "Внеплановая проверка лесов и страховочных систем; допуск к работам на высоте только после целевого инструктажа",
    "СИЗ": "Ежесменный контроль применения СИЗ мастером; пополнение запаса СИЗ на объекте",
    "LOTO": "Аудит процедуры LOTO: замки и бирки на всех пусковых устройствах, повторное обучение персонала",
    "Газоопасные работы": "Анализ воздушной среды перед каждым входом в ёмкость, калибровка газоанализаторов, обязательный наблюдающий",
    "Пожарная безопасность": "Проверка нарядов на огневые работы и первичных средств пожаротушения на всех площадках",
    "Электробезопасность": "Проверка электроинструмента и заземления; работы под напряжением — только по наряду-допуску",
    "Грузоподъемные операции": "Проверка строп и допусков стропальщиков; ограждение зоны подъёма и назначение сигнальщика",
    "Транспорт": "GPS-контроль скорости, обязательный предрейсовый осмотр, сигнальщик при движении задним ходом",
    "Порядок на рабочем месте": "Ежесменная уборка проходов; обработка скользких поверхностей песком",
    "Экология": "Поддоны под ёмкостями с ГСМ, проверка герметичности задвижек и трубопроводов",
}
PRIORITY_ORDER = {"Высокий": 0, "Средний": 1, "Низкий": 2}


def _org_recommendations(risk: pd.DataFrame, korgau: pd.DataFrame, top_n: int) -> list[dict]:
    recent = korgau[korgau["is_bad"] & (korgau["date"] > korgau["date"].max() - pd.Timedelta(days=90))]
    recs = []
    for _, row in risk.head(top_n).iterrows():
        cats = recent[recent["org_id"] == row["org_id"]]["category"].value_counts()
        if cats.empty:
            continue
        cat, n = cats.index[0], int(cats.iloc[0])
        recs.append({
            "priority": "Высокий" if row["risk_level"] == "Высокий" else "Средний",
            "scope": f'{row["org_id"]} — {row["name"]}',
            "measure": CONTROL_MEASURES[cat],
            "justification": (f'Ожидается {row["expected_incidents"]:.1f} происшествий в следующем месяце '
                              f'(индекс риска {row["risk_index"]:.0f}/100); преобладающее нарушение за 90 дней — '
                              f'«{cat}» ({n} случаев); доля плохих практик в прошлом месяце {row["bad_share_lag1"]:.0%}.'),
            "source": "модель риска + Карта Коргау",
        })
    return recs


def _contractor_recommendation(korgau: pd.DataFrame, organizations: pd.DataFrame) -> dict | None:
    bad = korgau[korgau["is_bad"]].merge(organizations[["org_id", "kind"]], on="org_id")
    overdue = bad.groupby("kind")["resolved"].apply(lambda s: (s == "Просрочено").mean())
    if overdue.get("Подрядчик", 0) <= overdue.get("Подразделение", 0):
        return None
    return {
        "priority": "Средний", "scope": "Все подрядные организации",
        "measure": "Закрепить в договорах подряда сроки устранения нарушений Коргау и санкции за просрочку; еженедельный разбор просроченных нарушений",
        "justification": (f'Подрядчики просрочивают устранение {overdue["Подрядчик"]:.0%} нарушений, '
                          f'подразделения — {overdue["Подразделение"]:.0%}. Просроченное нарушение остаётся действующей опасностью.'),
        "source": "паттерн Карт Коргау",
    }


def _monday_recommendation(incidents: pd.DataFrame) -> dict | None:
    share = (incidents["date"].dt.dayofweek == 0).mean()
    if share < 1.15 / 7:            # ниже равномерной доли с запасом 15% — не рекомендуем
        return None
    return {
        "priority": "Средний", "scope": "Все организации",
        "measure": "Целевой инструктаж и проверка готовности бригад в понедельник утром (первые 2 часа смены)",
        "justification": f"На понедельник приходится {share:.0%} происшествий при равномерной доле {1 / 7:.0%}.",
        "source": "паттерн журнала происшествий",
    }


def _season_recommendation(forecast: pd.DataFrame) -> dict | None:
    """Если ближайшие 3 месяца прогноза выше среднего по прогнозу на год — сезонная мера."""
    month = forecast.index[0].month
    if month in (11, 12, 1, 2):
        measure = "Зимний режим: обработка площадок и лестниц от гололёда, контроль скоростного режима на промысловых дорогах"
    elif month in (5, 6, 7, 8):
        measure = "Летний режим: график работ с учётом жары, усиленный контроль работ на высоте и огневых работ"
    else:
        return None
    return {
        "priority": "Средний", "scope": "Все организации", "measure": measure,
        "justification": (f'Прогноз на {forecast.index[0]:%m.%Y}: {forecast["forecast"].iloc[0]:.0f} происшествий '
                          f'(среднее за год прогноза — {forecast["forecast"].mean():.0f}); сезонный пик по данным EDA.'),
        "source": "модель прогноза + сезонность",
    }


def build_recommendations(risk: pd.DataFrame, korgau: pd.DataFrame, incidents: pd.DataFrame,
                          organizations: pd.DataFrame, forecast: pd.DataFrame, top_n: int = 3) -> list[dict]:
    recs = _org_recommendations(risk, korgau, top_n)
    for extra in (_contractor_recommendation(korgau, organizations),
                  _monday_recommendation(incidents), _season_recommendation(forecast)):
        if extra:
            recs.append(extra)
    return sorted(recs, key=lambda r: PRIORITY_ORDER[r["priority"]])
