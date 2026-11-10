"""
Единый стиль графиков для notebooks: тонкие линии, спокойная сетка, фиксированные цвета.

Цвет закреплён за сущностью (тип происшествия всегда одного цвета на всех графиках).
"""
import matplotlib.pyplot as plt

# Категориальная палитра (фиксированный порядок, проверена на различимость при дальтонизме)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TYPE_COLORS = {"Опасная ситуация": PALETTE[0], "Микротравма": PALETTE[1],
               "Несчастный случай": PALETTE[2], "Авария": PALETTE[3]}
KIND_COLORS = {"Подразделение": PALETTE[0], "Подрядчик": PALETTE[1]}
MAIN = PALETTE[0]
MUTED = "#9a9893"


def set_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (9, 4.2), "figure.dpi": 110,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#b5b3ad", "axes.labelcolor": "#52514e",
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e6e4df", "grid.linewidth": 0.6,
        "xtick.color": "#52514e", "ytick.color": "#52514e",
        "lines.linewidth": 2, "legend.frameon": False,
    })
