"""
Экономический эффект (F-07, MVP «Карточки KPI с числом людей и суммой тенге»).

Одна задача: перевести прогноз и сценарий снижения риска в людей и тенге.

Методика (каждое допущение явно указано):
1. Прогноз происшествий на 12 месяцев по типам — из src/forecast.py.
2. Стоимость одного события по типу — справочник incident_types.csv
   (НС = 5 млн ₸ прямых затрат по ТЗ, табл. 8.2; косвенные = 200% от прямых по ТЗ).
3. Снижение числа происшествий — из scenario modeling модели риска (src/risk.py):
   на сколько упадёт ожидаемое число событий, если снизить нарушения Коргау.
   Допущение: относительное снижение одинаково для всех типов.
4. ROI = (экономия − стоимость программы) / стоимость программы.
"""
import pandas as pd


def economic_effect(forecast_by_type: pd.Series, costs: pd.DataFrame,
                    relative_reduction: float, program_cost_kzt: float) -> dict:
    """forecast_by_type — ожидаемое число событий за 12 мес. по типам."""
    c = costs.set_index("type")
    unit_cost = c["direct_cost_kzt"] * (1 + c["indirect_cost_ratio"])

    expected = forecast_by_type.reindex(c.index).fillna(0)
    prevented = expected * relative_reduction

    baseline_cost = float((expected * unit_cost).sum())
    savings = float((prevented * unit_cost).sum())
    people = float(prevented.get("Несчастный случай", 0) + prevented.get("Микротравма", 0))

    return {
        "expected_incidents_12m": expected.round(1).to_dict(),
        "prevented_incidents_12m": prevented.round(1).to_dict(),
        "prevented_injured_people": round(people, 1),        # НС + микротравмы
        "prevented_lost_days": round(float((prevented * c["avg_lost_days"]).sum())),
        "baseline_loss_kzt": round(baseline_cost),
        "savings_kzt": round(savings),
        "program_cost_kzt": round(program_cost_kzt),
        "roi": round((savings - program_cost_kzt) / program_cost_kzt, 2) if program_cost_kzt else None,
        "payback_months": round(program_cost_kzt / (savings / 12), 1) if savings > 0 else None,
        "assumptions": [
            "Стоимость событий — справочник incident_types.csv (НС 5 млн ₸ по ТЗ)",
            "Косвенные потери = 200% от прямых (ТЗ, табл. 8.2)",
            f"Снижение числа событий {relative_reduction:.0%} — результат сценарного моделирования модели риска",
            "Относительное снижение одинаково для всех типов происшествий",
        ],
    }
