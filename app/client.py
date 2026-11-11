"""
Как дашборд получает данные. Одна задача: выбрать источник и вернуть JSON.

Два режима:
- «api»: дашборд обращается к FastAPI по HTTP (полная архитектура: API + UI);
- «embedded»: API не настроен и не запущен → дашборд вызывает тот же сервисный слой
  (app/service.py) напрямую, в своём процессе. Так весь сайт разворачивается одним
  приложением на Streamlit Community Cloud, без отдельного сервера.
Ответы в обоих режимах одинаковые: это один и тот же код app/service.py.
"""
import json
import os
import sys
from pathlib import Path

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # Streamlit добавляет в путь только папку app/
    sys.path.insert(0, str(ROOT))

LOCAL_API = "http://127.0.0.1:8000"


def _configured_url() -> str | None:
    """HSE_API_URL из Streamlit Secrets или переменной окружения."""
    try:
        if "HSE_API_URL" in st.secrets:
            return st.secrets["HSE_API_URL"]
    except Exception:                  # локально secrets.toml нет — это нормально
        pass
    return os.environ.get("HSE_API_URL")


@st.cache_resource(ttl=300, show_spinner=False)
def backend() -> tuple[str, str | None]:
    """('api', url) или ('embedded', None)."""
    url = _configured_url()
    if url:
        return "api", url.rstrip("/")
    try:
        requests.get(f"{LOCAL_API}/health", timeout=1.5).raise_for_status()
        return "api", LOCAL_API
    except requests.RequestException:
        return "embedded", None


@st.cache_resource(show_spinner="Загрузка моделей…")
def _service():
    """Сервис в процессе дашборда. Если моделей нет или они сохранены другой версией
    scikit-learn, модели переобучаются (≈ 10 секунд)."""
    import sklearn
    from src.train import MODEL_DIR, main as train
    meta = MODEL_DIR / "metadata.json"
    if not meta.exists() or json.loads(meta.read_text(encoding="utf-8"))["versions"]["scikit-learn"] != sklearn.__version__:
        train()
    from app.service import HSEService
    return HSEService()


def _jsonable(obj):
    """Тот же формат, что вернул бы API (numpy-числа → обычные числа)."""
    return json.loads(json.dumps(obj, ensure_ascii=False,
                                 default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def _embedded(path: str, p: dict):
    s = _service()
    routes = {
        "/health": lambda: {"status": "ok", "model_version": s.metadata["model_version"]},
        "/model-info": lambda: s.metadata,
        "/stats": lambda: s.stats(p.get("date_from"), p.get("date_to"), p.get("org_id"),
                                  p.get("type"), p.get("location")),
        "/forecast": lambda: s.forecast(int(p.get("horizon", 12))),
        "/risk": lambda: s.risk_top(int(p.get("n", 5))),
        "/alerts": lambda: s.alerts(p.get("as_of")),
        "/recommendations": lambda: s.recommendations(),
        "/economics": lambda: s.economics(float(p.get("violation_reduction", 0.3)),
                                          float(p.get("program_cost_kzt", 30_000_000))),
        "/monitoring": lambda: None,
    }
    return _jsonable(routes[path]())


def get(path: str, **params):
    params = {k: v for k, v in params.items() if v not in (None, "", "Все")}
    mode, url = backend()
    if mode == "embedded":
        return _embedded(path, params)
    r = requests.get(f"{url}{path}", params=params, timeout=120)   # 120 с: облачный API может «просыпаться»
    r.raise_for_status()
    return r.json()


def classify(text: str) -> tuple[bool, dict | str]:
    """(успех, результат или текст ошибки) — одинаковая проверка ввода в обоих режимах."""
    mode, url = backend()
    if mode == "api":
        r = requests.post(f"{url}/classify", json={"text": text}, timeout=60)
        if r.status_code != 200:
            detail = r.json().get("detail")
            return False, detail if isinstance(detail, str) else "Текст должен содержать от 10 до 5000 символов"
        return True, r.json()
    if not 10 <= len(text) <= 5000 or sum(ch.isalpha() for ch in text) < 5:
        return False, "Текст должен содержать от 10 до 5000 символов и хотя бы 5 букв"
    return True, _jsonable(_service().classify(text))
