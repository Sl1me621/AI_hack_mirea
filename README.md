## Мониторинг активности и безопасности на депо

Полный прототип: FastAPI + PostgreSQL + SQLAlchemy на backend, статичный HTML/CSS/JS на frontend. Данные хранятся в таблицах `events` и `window_analytics`; фронт работает с агрегированными окнами по 10 секунд.

### Стек и зависимости
- Python 3.11
- FastAPI, SQLAlchemy, Pydantic, psycopg2-binary
- Uvicorn для запуска сервера
- OpenCV + EasyOCR (для OCR номера поезда в скрипте)
- Frontend: чистые HTML/CSS/JS (без сборки)

Установка зависимостей:
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Конфигурация БД
- Переменная окружения: `DATABASE_URL`
- Дефолт: `postgresql+psycopg2://postgres:postgres@localhost:5432/depot_analytics`
- Модуль подключения: `backend/db.py` (engine, SessionLocal, Base)

### Инициализация и тестовые данные
1) Создать таблицы и справочники зон/одежды:
```bash
python scripts/init_db.py
```
2) Залить пример окон аналитики (video_id="session_1"):
```bash
python scripts/sample_client.py
```

### Запуск backend
```bash
python -m uvicorn backend.main:app --reload
```
Сервер: `http://127.0.0.1:8000`

### Запуск frontend
```bash
cd frontend
python -m http.server 5500
```
Открыть `http://127.0.0.1:5500`, нажать «Обновить данные».

### Основные сущности (ORM)
- `WindowAnalytics` (таблица window_analytics): video_id, окно [start_sec, end_sec], workers_in_safe/medium/danger, dangerous_situations_count, violations_count, cleaning_workers_count, train_present/arrived/departed, train_number
- `Event`, `Worker`, `Zone`, `ClothingType` для сырых событий (используется частично)

### API (основное)
- `GET /health` — статус
- `POST /api/windows/batch` — принимает `{ "windows": [ WindowAnalyticsCreate... ] }`, сохраняет окна
- `GET /api/windows` — список окон (параметры `video_id`, `limit`, `offset`)
- `GET /api/metrics/current_state` — метрики по последнему окну (опционально `video_id`)
- `GET /api/metrics/dangerous_situations` — окна, где `dangerous_situations_count > 0` (опционально `video_id`)
- `POST /api/events` — создать событие (EventCreate)
- `GET /api/events` — список событий с фильтрами (start/end, event_type, zone_name, worker_id, activity)
- `GET /api/events/cleaning` — события с activity="cleaning"
- `GET /api/zones` — справочник зон
- `GET /api/clothing_types` — справочник типов одежды

### Форматы Pydantic
- `WindowAnalyticsCreate/Read`: поля окна (см. `backend/models.py`)
- `WindowAnalyticsBatchCreate`: `{ "windows": [...] }`
- `CurrentState`: workers_in_safe/medium/danger, dangerous_situations_count, cleaning_workers_count
- `EventCreate/Read`: timestamp, event_type, worker_id, zone_name, clothing_type_name, train_number, activity, is_violation

### Frontend (frontend/)
- `index.html` — карточки метрик, таблица окон, блок опасных окон, блок уборки
- `script.js` — fetch: `/api/metrics/current_state`, `/api/windows`, `/api/metrics/dangerous_situations`; фильтрует уборку из windows; форматирует окна
- `style.css` — тёмная тема с акцентами по зонам (safe/medium/danger)

### OCR-скрипт (scripts/train_ocr_easyocr.py)
Назначение: распознать номер поезда из изображения и записать в `window_analytics.train_number` для выбранного `video_id`.
- Аргументы: `--image-path screen.png`, `--video-id session_1`
- Логика: читает ROI, предобрабатывает, запускает EasyOCR, нормализует строку, обновляет все записи с пустым train_number для указанного video_id.
- Запуск:
```bash
python scripts/train_ocr_easyocr.py --image-path screen.png --video-id session_1
```
Выводит найденный номер и количество обновлённых строк.

### Структура репозитория
- `backend/` — FastAPI-приложение (main.py, api/windows.py, crud.py, db.py, models.py, orm_models.py)
- `frontend/` — статика дашборда (index.html, style.css, script.js)
- `scripts/` — утилиты (init_db.py, sample_client.py, train_ocr_easyocr.py)
- `requirements.txt` — зависимости backend + OCR

### Мини чек-лист запуска
1. venv + `pip install -r requirements.txt`
2. Настроить `DATABASE_URL` (если нужно)
3. `python scripts/init_db.py`
4. `python scripts/sample_client.py` (опционально, чтобы наполнить окнами)
5. `python -m uvicorn backend.main:app --reload`
6. В другом терминале: `cd frontend && python -m http.server 5500`
7. Открыть `http://127.0.0.1:5500`, нажать «Обновить данные`
8. (Опционально) `python scripts/train_ocr_easyocr.py --image-path screen.png --video-id session_1` чтобы проставить номер поезда в окнах
