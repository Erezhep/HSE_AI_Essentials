"""
Дашборд HSE AI Analytics (уровень «представления» из архитектуры ТЗ, раздел 5).

Одна задача: показать пользователю результаты. Расчётов здесь нет: данные приходят
из app/client.py — через FastAPI (если он запущен или задан HSE_API_URL) или напрямую
из сервисного слоя (режим одного приложения для Streamlit Cloud).
Запуск:  streamlit run app/dashboard.py
"""
import altair as alt
import pandas as pd
import streamlit as st

import client

TYPE_COLORS = {"Опасная ситуация": "#2a78d6", "Микротравма": "#eb6834",
               "Несчастный случай": "#1baf7a", "Авария": "#eda100"}
TYPE_SCALE = alt.Scale(domain=list(TYPE_COLORS), range=list(TYPE_COLORS.values()))
MAIN = "#2a78d6"

st.set_page_config(page_title="HSE AI Analytics", page_icon="🛡️", layout="wide")


@st.cache_data(ttl=60, show_spinner=False)
def get(path: str, **params) -> dict | list:
    return client.get(path, **params)


def fmt_kzt(x: float) -> str:
    return f"{x / 1e6:,.1f} млн ₸".replace(",", " ")


# ------------------------------------------------------------------ шапка и проверка API
st.title("🛡️ HSE AI Analytics")
st.caption("AI-аналитика охраны труда: происшествия, Карты Коргау, прогноз, риски, алерты, экономический эффект")
mode, api_url = client.backend()
try:
    with st.spinner("Подключение… (облачный API после простоя просыпается до 1 минуты)"):
        health = get("/health")
except Exception:
    st.error(f"API недоступен по адресу {api_url}. Проверьте сервис и обновите страницу.")
    st.stop()
st.caption(f"Источник данных: {'API ' + api_url if mode == 'api' else 'встроенный режим (сервис в процессе дашборда)'}"
           f" · модель v{health['model_version']}")

tabs = st.tabs(["📊 Обзор", "📈 Прогноз", "🎯 Зоны риска", "🔔 Алерты", "💡 Рекомендации",
                "💰 Экономика", "🔎 Классификатор", "🛠 Мониторинг"])

# ------------------------------------------------------------------ 1. Обзор (F-05)
with tabs[0]:
    base = get("/stats")
    c1, c2, c3, c4 = st.columns(4)
    period = c1.date_input("Период", value=(pd.Timestamp("2023-09-01"), pd.Timestamp("2026-08-31")))
    org = c2.selectbox("Организация", ["Все"] + sorted(base["by_org"]))
    typ = c3.selectbox("Тип", ["Все"] + list(TYPE_COLORS))
    loc = c4.selectbox("Локация", ["Все"] + sorted(base["by_location"]))
    date_from, date_to = (period if isinstance(period, tuple) and len(period) == 2 else (None, None))
    s = get("/stats", date_from=str(date_from) if date_from else None, date_to=str(date_to) if date_to else None,
            org_id=org, type=typ, location=loc)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Происшествий", s["total"])
    k2.metric("Несчастных случаев", s["by_type"].get("Несчастный случай", 0))
    k3.metric("Аварий", s["by_type"].get("Авария", 0))
    k4.metric("Открытых дел", s["open_cases"])

    if s["total"]:
        monthly = pd.DataFrame(s["by_month"]).melt("month", var_name="Тип", value_name="Число")
        st.altair_chart(alt.Chart(monthly, title="Динамика по месяцам").mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("month:T", title=None), y=alt.Y("Число:Q", title="Происшествий"),
            color=alt.Color("Тип:N", scale=TYPE_SCALE), tooltip=["month:T", "Тип:N", "Число:Q"]),
            width="stretch")
        a, b, c = st.columns(3)
        for col, key, title in [(a, "by_org", "По организациям"), (b, "by_location", "По локациям"),
                                (c, "top_root_causes", "Топ корневых причин (F-03)")]:
            df = pd.Series(s[key], name="Число").rename_axis("Категория").reset_index()
            col.altair_chart(alt.Chart(df, title=title).mark_bar(color=MAIN, cornerRadiusEnd=4).encode(
                x=alt.X("Число:Q"), y=alt.Y("Категория:N", sort="-x", title=None),
                tooltip=["Категория", "Число"]), width="stretch")
    else:
        st.info("Нет происшествий для выбранных фильтров.")

# ------------------------------------------------------------------ 2. Прогноз (F-06)
with tabs[1]:
    horizon = st.radio("Горизонт прогноза", [3, 6, 12], index=2, horizontal=True, format_func=lambda h: f"{h} мес.")
    fc = get("/forecast", horizon=horizon)
    hist = pd.DataFrame(fc["history"]).assign(kind="факт")
    fut = pd.DataFrame(fc["months"])
    band = alt.Chart(fut).mark_area(opacity=.18, color=MAIN).encode(x="month:T", y="lower:Q", y2="upper:Q")
    line_h = alt.Chart(hist).mark_line(color="#0b0b0b", point=True).encode(
        x=alt.X("month:T", title=None), y=alt.Y("incidents:Q", title="Происшествий в месяц"),
        tooltip=["month:T", "incidents:Q"])
    line_f = alt.Chart(fut).mark_line(color=MAIN, point=True, strokeDash=[6, 3]).encode(
        x="month:T", y="forecast:Q", tooltip=["month:T", "forecast:Q", "lower:Q", "upper:Q"])
    st.altair_chart((band + line_h + line_f).properties(title="Факт и прогноз с 90% интервалом"),
                    width="stretch")
    summary = pd.DataFrame(fc["summary"]).set_index("horizon_months")
    summary.index = [f"{h} мес." for h in summary.index]
    st.subheader("Ожидаемое число происшествий по типам")
    st.dataframe(summary.rename(columns={"total": "Всего", "lower": "Нижняя граница", "upper": "Верхняя граница"}),
                 width="stretch")
    st.caption("Модель: линейный тренд + сезонная волна (OLS). Типы — по долям за последние 24 месяца.")

# ------------------------------------------------------------------ 3. Зоны риска (F-12)
with tabs[2]:
    r = get("/risk", n=5)
    st.subheader(f"Топ-5 зон риска на {r['month']}")
    orgs = pd.DataFrame(r["organizations"])
    st.dataframe(orgs[["org_id", "name", "kind", "expected_incidents", "risk_index", "risk_level",
                       "bad_share_lag1", "overdue_lag1"]].rename(columns={
        "org_id": "ID", "name": "Организация", "kind": "Тип", "expected_incidents": "Ожидается событий",
        "risk_index": "Индекс риска", "risk_level": "Уровень", "bad_share_lag1": "Доля нарушений (прошл. мес.)",
        "overdue_lag1": "Просрочено нарушений"}).style.format({"Ожидается событий": "{:.1f}",
                                                              "Доля нарушений (прошл. мес.)": "{:.0%}", "Индекс риска": "{:.0f}"}),
        width="stretch", hide_index=True)
    a, b = st.columns(2)
    allorg = pd.DataFrame(r["all_organizations"])
    a.altair_chart(alt.Chart(allorg, title="Индекс риска всех организаций").mark_bar(color=MAIN, cornerRadiusEnd=4).encode(
        x=alt.X("risk_index:Q", title="Индекс риска (0–100)"), y=alt.Y("org_id:N", sort="-x", title=None),
        tooltip=["name", "expected_incidents", "risk_index"]), width="stretch")
    locs = pd.DataFrame(r["locations"])
    b.altair_chart(alt.Chart(locs, title="Топ-5 локаций").mark_bar(color="#eb6834", cornerRadiusEnd=4).encode(
        x=alt.X("expected_incidents:Q", title="Ожидается событий"), y=alt.Y("location:N", sort="-x", title=None),
        tooltip=["location", "expected_incidents"]), width="stretch")
    st.caption("Модель: Poisson-регрессия по истории происшествий и нарушениям Коргау за прошлый месяц. "
               "Сумма по организациям согласована с общим прогнозом.")

# ------------------------------------------------------------------ 4. Алерты (F-11, F-14)
with tabs[3]:
    as_of = st.date_input("Дата проверки", value=pd.Timestamp("2026-08-31"), key="as_of")
    alerts = get("/alerts", as_of=str(as_of))
    if not alerts:
        st.success("Алертов нет.")
    counts = pd.Series([a["level"] for a in alerts]).value_counts()
    cols = st.columns(4)
    for col, (lvl, icon) in zip(cols, [("Критический", "🔴"), ("Высокий", "🟠"), ("Средний", "🟡"), ("Низкий", "🟢")]):
        col.metric(f"{icon} {lvl}", int(counts.get(lvl, 0)))
    for a in alerts:
        st.markdown(f"{a['icon']} **{a['level']}** · `{a['org_id']}` · {a['title']}  \n<small>{a['details']}</small>",
                    unsafe_allow_html=True)
    with st.expander("Правила алертов (ТЗ, раздел 3.2.2)"):
        st.markdown("- 🔴 **Критический** — нарушений за 30 дней больше обычного уровня × 2\n"
                    "- 🟠 **Высокий** — нарушения одной категории > 3 раз за 30 дней; pre-alert для организаций из топа риска\n"
                    "- 🟡 **Средний** — рост > 15% к тем же 30 дням прошлого года\n"
                    "- 🟢 **Низкий** — снижение > 15% (информационное)")

# ------------------------------------------------------------------ 5. Рекомендации (F-04)
with tabs[4]:
    for rec in get("/recommendations"):
        with st.container(border=True):
            icon = "🔴" if rec["priority"] == "Высокий" else "🟡"
            st.markdown(f"{icon} **Приоритет: {rec['priority']}** · {rec['scope']}")
            st.markdown(f"**Мера:** {rec['measure']}")
            st.caption(f"Обоснование: {rec['justification']} · Источник: {rec['source']}")

# ------------------------------------------------------------------ 6. Экономика (F-07)
with tabs[5]:
    c1, c2 = st.columns(2)
    red = c1.slider("Снижение нарушений Коргау (сценарий)", 0, 80, 30, step=5, format="%d%%") / 100
    cost = c2.number_input("Стоимость программы мер, млн ₸ в год", 1.0, 1000.0, 30.0, step=5.0) * 1e6
    e = get("/economics", violation_reduction=red, program_cost_kzt=cost)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Предотвращено пострадавших за год", f"{e['prevented_injured_people']:.0f} чел.")
    k2.metric("Экономия за год", fmt_kzt(e["savings_kzt"]))
    k3.metric("ROI", f"{e['roi']:.1f}×" if e["roi"] is not None else "—")
    k4.metric("Окупаемость", f"{e['payback_months']:.1f} мес." if e["payback_months"] else "—")
    st.markdown(f"Модель риска: при снижении нарушений на **{red:.0%}** ожидаемое число происшествий падает на "
                f"**{e['scenario']['relative_reduction']:.1%}**.")
    df = pd.DataFrame({"Ожидается за 12 мес.": e["expected_incidents_12m"],
                       "Предотвращено": e["prevented_incidents_12m"]})
    st.dataframe(df, width="stretch")
    with st.expander("Методика и допущения"):
        for a in e["assumptions"]:
            st.markdown(f"- {a}")
        st.markdown(f"- Ожидаемые потери без мер: {fmt_kzt(e['baseline_loss_kzt'])}")

# ------------------------------------------------------------------ 7. Классификатор (F-01)
with tabs[6]:
    text = st.text_area("Описание происшествия", height=120,
                        value="Оператор поскользнулся на обледенелой площадке, получил перелом лодыжки, доставлен в районную больницу.")
    if st.button("Определить тип", type="primary"):
        ok, res = client.classify(text)
        if not ok:
            st.error(res)
        else:
            st.markdown(f"### {res['type']}")
            if res["low_confidence"]:
                st.warning("Модель не уверена: два класса близки. Проверьте вручную.")
            st.markdown("**Слова, повлиявшие на решение:** " + ", ".join(f"`{w}`" for w in res["evidence"]))
            sc = pd.Series(res["scores"], name="Оценка").rename_axis("Тип").reset_index()
            st.altair_chart(alt.Chart(sc).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("Оценка:Q", title="Оценка SVM (не вероятность)"), y=alt.Y("Тип:N", sort="-x", title=None),
                color=alt.Color("Тип:N", scale=TYPE_SCALE, legend=None)), width="stretch")

# ------------------------------------------------------------------ 8. Мониторинг
with tabs[7]:
    get.clear()
    m = get("/monitoring")
    if m is None:
        st.info("Метрики запросов собирает FastAPI (app/monitoring.py). Во встроенном режиме API не используется — "
                "запустите `uvicorn app.api:app`, чтобы увидеть мониторинг.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Запросов", m["requests_total"])
        k2.metric("Ошибок", sum(m["errors"].values()))
        k3.metric("Задержка p50", f"{m['latency_ms']['p50']} мс")
        k4.metric("Задержка p95", f"{m['latency_ms']['p95']} мс")
        st.json(m, expanded=False)
    st.subheader("Паспорт модели")
    st.json(get("/model-info"), expanded=False)
