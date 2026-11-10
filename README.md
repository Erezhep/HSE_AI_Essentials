# 🛡️ HSE AI Analytics

**AI Essentials: Theory and Applications** (магистратура, Applied AI) пәні бойынша курстық жоба.
Негізі — «Внедрение ИИ в сфере Охраны Труда» хакатонының ТЗ-сы, «AI-Аналитика HSE-системы» модулі ([`docs/TZ_hackathon_HSE.docx`](docs/TZ_hackathon_HSE.docx)).

Жүйе еңбекті қорғаудың екі журналымен жұмыс істейді: **Происшествия** және **Карта Коргау**. Нәтижесінде ол:
1. өткен оқиғаларды талдайды;
2. алдағы 3/6/12 айды болжайды;
3. келесі айдағы ең қауіпті 5 аймақты көрсетеді;
4. ТЗ ережелері бойынша алерт береді;
5. шараларды ұсынады;
6. әсерін **адам** және **теңге** түрінде есептейді.

> 📖 Жобаны кезең-кезеңмен түсіну үшін: **[docs/GUIDE_kz.md](docs/GUIDE_kz.md)**. Онда әр шешімнің себебі және қорғауға арналған 15 сұрақ-жауап бар.

---

## Негізгі нәтижелер

| Тапсырма | Финалдық модель | Сапасы (test) |
|---|---|---|
| Инциденттер санын болжау (F-06) | Сызықтық тренд + маусымдық гармоника | MAE 4.8 оқиға/ай (seasonal naive: 9.2) |
| Ұйымның тәуекелі (F-12, F-13) | Poisson регрессиясы | deviance 1.96 (наив: 2.20); Коргау белгілерімен қате −4.8% |
| Мәтіннен түрін анықтау (F-01) | TF-IDF + LinearSVC (balanced) | accuracy 0.95, macro-F1 0.92 (baseline: 0.58) |
| Алерттер (F-11, F-14) | ТЗ 3.2.2 ережелері | precision 0.48, жалпы деңгей 0.36 (lift 1.34) |

**Басты ғылыми нәтиже:** Коргау бұзушылықтары инциденттердің алдында жүреді. Байланыс 1 айлық лагта ең күшті (r ≈ 0.28), сондықтан ерте ескерту жүйесін құруға негіз бар.

---

## Курс тапсырмалары қай жерде

| Тапсырма | Файлдар |
|---|---|
| **A1** Problem & Dataset | [`docs/A1_proposal.md`](docs/A1_proposal.md) |
| **A2** Data Preparation & EDA | [`notebooks/01_data_preparation_eda.ipynb`](notebooks/01_data_preparation_eda.ipynb), [`docs/A2_report.md`](docs/A2_report.md) |
| **Midterm** Baseline | [`notebooks/02_baseline_midterm.ipynb`](notebooks/02_baseline_midterm.ipynb), [`docs/midterm_presentation.md`](docs/midterm_presentation.md) |
| **A3** Training & Fine-Tuning | [`notebooks/03_model_training.ipynb`](notebooks/03_model_training.ipynb), [`notebooks/04_transformer_finetune_lora.ipynb`](notebooks/04_transformer_finetune_lora.ipynb) (Colab GPU) |
| **Final** прототип | `app/`, `models/`, `tests/`, [`docs/technical_documentation.md`](docs/technical_documentation.md), [`docs/final_presentation.md`](docs/final_presentation.md) |

---

## Құрылымы (әр файл бір ғана міндетке жауап береді)

```
HSE_AI_Essentials/
├── data/                      ТЗ бойынша синтетикалық деректер (4 CSV)
├── data_generation/           деректер генераторы (seed=42, қайталанатын)
├── src/                       ML ядросы
│   ├── data.py                жүктеу және тазалау
│   ├── features.py            «ұйым × ай» кестесі, лагтар, уақыт бойынша бөлу
│   ├── forecast.py            болжау + backtesting
│   ├── risk.py                тәуекел, топ-5, сценарийлер
│   ├── nlp.py                 мәтін классификаторы
│   ├── alerts.py              ТЗ алерттері
│   ├── recommendations.py     ұсыныстар
│   ├── economics.py           адам/теңге/ROI
│   ├── plotting.py            графиктердің ортақ стилі
│   └── train.py               финалдық модельдерді үйрету және сақтау
├── app/
│   ├── service.py             модульдерді біріктіреді
│   ├── api.py                 FastAPI
│   ├── monitoring.py          мониторинг
│   └── dashboard.py           Streamlit дашборды
├── models/                    сақталған модельдер + metadata.json
├── notebooks/                 01 EDA · 02 baseline · 03 оқыту · 04 трансформер/LoRA
├── reports/                   эксперимент нәтижелері (JSON)
├── tests/                     pytest (24 тест)
└── docs/                      ТЗ, тапсырма есептері, техникалық құжаттама, гайд
```

---

## Іске қосу

Python 3.11+ қажет. Windows PowerShell командалары:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements-dev.txt

# (міндетті емес) деректерді қайта жасау және модельдерді қайта үйрету
python data_generation\generate_data.py
python -m src.train

# 1-терминал: API      → http://127.0.0.1:8000/docs
uvicorn app.api:app --reload

# 2-терминал: дашборд  → http://localhost:8501
streamlit run app/dashboard.py

# тесттер
pytest
```

Notebooks: `jupyter notebook notebooks/` → **Restart & Run All**. Notebook 04 Google Colab-та GPU-мен іске қосылады, нұсқаулық оның ішінде берілген.

> Егер `models/*.joblib` басқа scikit-learn нұсқасында жүктелмесе, `python -m src.train` командасын бір рет орындаңыз.

## Интернетте deploy жасау

- **API → Render.com:** New → Blueprint → репозиторийді таңдау (`render.yaml` дайын тұр). Тексеру үшін `https://<сіздің-адрес>/health` ашыңыз.
- **Дашборд → Streamlit Community Cloud:** New app → `app/dashboard.py`. Secrets бөлімінде: `HSE_API_URL = "https://<API-адресі>"`.

## API

`/health` · `/model-info` · `/monitoring` · `/stats` · `/forecast?horizon=12` · `/risk?n=5` · `/alerts?as_of=YYYY-MM-DD` · `/recommendations` · `/economics?violation_reduction=0.3` · `POST /classify`
Толық сипаттама: [`docs/technical_documentation.md`](docs/technical_documentation.md) және `/docs` (Swagger).

## Шектеулер

Деректер синтетикалық, ал заңдылықтарды генератор салған. Жоба нақты кәсіпорынның жағдайын емес, әдістердің жұмыс істейтінін көрсетеді. Уақыт қатары қысқа (36 ай). ТЗ-дағы F-08 (фото талдау) орындалмады, себебі деректерде фото жоқ. Толығырақ: техникалық құжаттаманың 10-бөлімі.
