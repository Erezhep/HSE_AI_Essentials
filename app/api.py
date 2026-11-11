"""
REST API AI-модуля HSE (уровень «ML Core → API» из архитектуры ТЗ, раздел 5).

Одна задача: принять HTTP-запрос, вызвать сервис, вернуть JSON.
Запуск:  uvicorn app.api:app --reload      Документация: http://127.0.0.1:8000/docs
"""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from monitoring import Monitor
from service import HSEService

state: dict = {"service": None, "error": None}
monitor = Monitor()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Модели загружаются один раз при старте, а не на каждый запрос."""
    try:
        state["service"] = HSEService()
    except Exception as exc:          # нет моделей или данных → сервис жив, но /health = 503
        state["error"] = f"{type(exc).__name__}: {exc}"
    yield


app = FastAPI(title="HSE AI Analytics API", version="1.0.0", lifespan=lifespan,
              description="Прогноз происшествий, зоны риска, алерты Коргау, рекомендации, экономический эффект, NLP")


@app.middleware("http")
async def track(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        monitor.record_request(request.url.path, 500, (time.perf_counter() - start) * 1000)
        return JSONResponse({"detail": "Internal error"}, status_code=500)
    monitor.record_request(request.url.path, response.status_code, (time.perf_counter() - start) * 1000)
    return response


def service() -> HSEService:
    if state["service"] is None:
        raise HTTPException(503, f"Модели не загружены: {state['error']}. Выполните: python -m src.train")
    return state["service"]


# ------------------------------------------------------------------ служебные
@app.get("/health", tags=["service"])
def health():
    if state["service"] is None:
        return JSONResponse({"status": "unavailable", "error": state["error"]}, status_code=503)
    return {"status": "ok", "model_version": state["service"].metadata["model_version"]}


@app.get("/model-info", tags=["service"])
def model_info():
    return service().metadata


@app.get("/monitoring", tags=["service"])
def monitoring():
    return monitor.snapshot()


# ------------------------------------------------------------------ аналитика
@app.get("/stats", tags=["analytics"], summary="Статистика происшествий с фильтрами (F-02, F-05)")
def stats(date_from: str | None = None, date_to: str | None = None, org_id: str | None = None,
          type: str | None = None, location: str | None = None):
    return service().stats(date_from, date_to, org_id, type, location)


@app.get("/forecast", tags=["analytics"], summary="Прогноз на 3/6/12 месяцев (F-06)")
def forecast(horizon: int = Query(12, ge=1, le=24)):
    return service().forecast(horizon)


@app.get("/risk", tags=["analytics"], summary="Топ зон риска на следующий месяц (F-12)")
def risk(n: int = Query(5, ge=1, le=20)):
    return service().risk_top(n)


@app.get("/alerts", tags=["analytics"], summary="Алерты по Карте Коргау (F-11, F-14)")
def alerts(as_of: str | None = Query(None, description="Дата проверки YYYY-MM-DD; по умолчанию последняя")):
    return service().alerts(as_of)


@app.get("/recommendations", tags=["analytics"], summary="Рекомендации по мерам контроля (F-04)")
def recommendations():
    return service().recommendations()


@app.get("/economics", tags=["analytics"], summary="Эффект в людях и тенге (F-07)")
def economics(violation_reduction: float = Query(0.3, ge=0, le=1),
              program_cost_kzt: float = Query(30_000_000, gt=0)):
    return service().economics(violation_reduction, program_cost_kzt)


# ------------------------------------------------------------------ NLP
class IncidentText(BaseModel):
    text: str = Field(..., min_length=10, max_length=5000,
                      examples=["Слесарь поскользнулся на обледенелой площадке, получил перелом лодыжки, госпитализирован."])


@app.post("/classify", tags=["nlp"], summary="Тип происшествия по тексту (F-01)")
def classify(body: IncidentText):
    if sum(ch.isalpha() for ch in body.text) < 5:
        raise HTTPException(422, "В тексте слишком мало букв для анализа")
    result = service().classify(body.text)
    monitor.record_prediction(result)
    return result
