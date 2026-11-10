"""
NLP-классификация происшествий по тексту описания (F-01).

Одна задача: по тексту «молнии» определить тип: Опасная ситуация / Микротравма /
Несчастный случай / Авария.

Финальная модель: TF-IDF (отдельные слова) + линейный SVM с балансировкой классов.
Гиперпараметры (C=3, униграммы, min_df=1) подобраны GridSearchCV в notebooks/03_model_training.ipynb.
- TF-IDF превращает текст в вектор: слово получает большой вес, если часто встречается
  в этом описании и редко — в остальных.
- Линейный SVM проводит между классами разделяющие гиперплоскости.
- class_weight='balanced' штрафует ошибки на редких классах сильнее (аварий ≈ 1%).
"""
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

RANDOM_STATE = 42


def normalize(text: str) -> str:
    """Одинаковая нормализация при обучении и в API: Unicode, регистр, пробелы."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return re.sub(r"\s+", " ", text).strip()


def split_texts(incidents: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70 / 15 / 15. Одно и то же разбиение во всех notebooks."""
    train, rest = train_test_split(incidents, test_size=0.30, stratify=incidents["type"],
                                   random_state=RANDOM_STATE)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["type"],
                                 random_state=RANDOM_STATE)
    return train, val, test


def make_text_model(C: float = 3.0, ngram_max: int = 1, min_df: int = 1) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, ngram_max),
                                  min_df=min_df, sublinear_tf=True)),
        ("svm", LinearSVC(C=C, class_weight="balanced", random_state=RANDOM_STATE)),
    ])


def classify(model: Pipeline, text: str, top_words: int = 5) -> dict:
    """Класс, оценки по классам и слова, которые сильнее всего повлияли на решение."""
    scores = model.decision_function([text])[0]
    classes = model.classes_
    best = int(np.argmax(scores))
    order = np.sort(scores)[::-1]

    # Вклад слова = его TF-IDF вес в тексте × коэффициент SVM для выбранного класса
    tfidf, svm = model.named_steps["tfidf"], model.named_steps["svm"]
    vec = tfidf.transform([text])
    contrib = vec.multiply(svm.coef_[best]).toarray()[0]
    idx = np.argsort(contrib)[::-1][:top_words]
    vocab = tfidf.get_feature_names_out()
    evidence = [vocab[i] for i in idx if contrib[i] > 0]

    return {
        "type": classes[best],
        "scores": {c: round(float(s), 3) for c, s in zip(classes, scores)},
        "margin": round(float(order[0] - order[1]), 3),  # разрыв между двумя лучшими классами
        "low_confidence": bool(order[0] - order[1] < 0.3),
        "evidence": evidence,
    }
