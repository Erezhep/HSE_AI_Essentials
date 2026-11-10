"""
Мониторинг сервиса.

Одна задача: считать, что происходит с API, пока он работает.
- технические метрики: число запросов, ошибки, задержка (p50 / p95);
- ML-метрики: распределение предсказанных классов и доля «неуверенных» ответов.
  Если доля, например, «Аварий» резко выросла, это сигнал проверить модель
  (дрейф данных) или реальную обстановку на объектах.
Каждый запрос пишется одной JSON-строкой в logs/api.log.
"""
import json
import logging
import time
from collections import Counter, deque
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


class Monitor:
    def __init__(self, window: int = 1000):
        self.started = time.time()
        self.requests = Counter()
        self.errors = Counter()
        self.latencies_ms = deque(maxlen=window)      # последние N запросов
        self.predicted_classes = Counter()
        self.low_confidence = 0

        LOG_DIR.mkdir(exist_ok=True)
        self.log = logging.getLogger("hse_api")
        if not self.log.handlers:
            handler = logging.FileHandler(LOG_DIR / "api.log", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.log.addHandler(handler)
            self.log.setLevel(logging.INFO)

    def record_request(self, path: str, status: int, latency_ms: float) -> None:
        self.requests[path] += 1
        if status >= 400:
            self.errors[f"{path} {status}"] += 1
        self.latencies_ms.append(latency_ms)
        self.log.info(json.dumps({"ts": round(time.time(), 3), "path": path, "status": status,
                                  "latency_ms": round(latency_ms, 1)}))

    def record_prediction(self, result: dict) -> None:
        self.predicted_classes[result["type"]] += 1
        self.low_confidence += int(result["low_confidence"])

    def snapshot(self) -> dict:
        lat = sorted(self.latencies_ms)
        pct = lambda q: round(lat[min(int(q * len(lat)), len(lat) - 1)], 1) if lat else None
        n_pred = sum(self.predicted_classes.values())
        return {
            "uptime_sec": round(time.time() - self.started),
            "requests_total": sum(self.requests.values()),
            "requests_by_endpoint": dict(self.requests),
            "errors": dict(self.errors),
            "latency_ms": {"p50": pct(0.5), "p95": pct(0.95), "window": len(lat)},
            "predictions": {"total": n_pred, "by_class": dict(self.predicted_classes),
                            "low_confidence_share": round(self.low_confidence / n_pred, 3) if n_pred else None},
        }
