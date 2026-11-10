"""API: все эндпоинты отвечают, некорректный ввод отклоняется, мониторинг считает запросы."""
import pytest
from fastapi.testclient import TestClient

from app.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


@pytest.mark.parametrize("h", [3, 6, 12])
def test_forecast_horizon(client, h):
    body = client.get("/forecast", params={"horizon": h}).json()
    assert len(body["months"]) == h
    assert {s["horizon_months"] for s in body["summary"]} == {3, 6, 12}


def test_forecast_rejects_bad_horizon(client):
    assert client.get("/forecast", params={"horizon": 0}).status_code == 422


def test_risk_top5(client):
    body = client.get("/risk", params={"n": 5}).json()
    assert len(body["organizations"]) == 5
    values = [o["expected_incidents"] for o in body["organizations"]]
    assert values == sorted(values, reverse=True)


def test_stats_filters(client):
    all_ = client.get("/stats").json()["total"]
    one = client.get("/stats", params={"org_id": "ORG-08"}).json()["total"]
    assert 0 < one < all_


def test_alerts_and_recommendations(client):
    assert isinstance(client.get("/alerts").json(), list)
    recs = client.get("/recommendations").json()
    assert len(recs) >= 3                                  # MVP ТЗ: минимум 3 рекомендации
    assert all(r["measure"] and r["justification"] for r in recs)


def test_economics(client):
    e = client.get("/economics", params={"violation_reduction": 0.3}).json()
    assert e["savings_kzt"] > 0 and e["prevented_injured_people"] > 0
    assert client.get("/economics", params={"violation_reduction": 1.5}).status_code == 422


def test_classify_valid_and_invalid(client):
    ok = client.post("/classify", json={"text": "Пострадавших нет, работы приостановлены после срабатывания газоанализатора."})
    assert ok.status_code == 200 and ok.json()["type"] == "Опасная ситуация"
    assert client.post("/classify", json={"text": "коротко"}).status_code == 422
    assert client.post("/classify", json={"text": "1234567890 !!!"}).status_code == 422
    assert client.post("/classify", json={}).status_code == 422


def test_monitoring_counts_requests(client):
    m = client.get("/monitoring").json()
    assert m["requests_total"] > 0 and m["latency_ms"]["p50"] is not None
    assert m["predictions"]["total"] >= 1
