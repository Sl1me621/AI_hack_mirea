# Пошаговый запуск проекта

## 1. Подготовка окружения
- Требуется Python 3.11.
- Создать и активировать venv:
```bash
python -m venv .venv
.venv\Scripts\activate
```
- Установить зависимости:
```bash
pip install -r requirements.txt
```

## 2. База данных и API
- База по умолчанию: `analytics.db` в корне репозитория (SQLite). Переопределить можно через `DATABASE_URL`.
- Создать таблицы и запустить API:
```bash
python -m uvicorn backend.main:app --reload
```
- При старте создаются таблицы, а фоновая задача раз в ~5–60 сек сбрасывает буфер окон в БД. При остановке также происходит flush.
- (Опционально) засев справочников зон/одежды:
```bash
python scripts/init_db.py
```

## 3. CV-пайплайн
- Основной скрипт: `runfile/log.py`.
- Перед запуском укажи пути к видео/весам в `runfile/log.py` (`VIDEO_PATH`, `DETECT_MODEL_PATH`, `POSE_MODEL_PATH`).
- Запуск:
```bash
python runfile/log.py
```
- Скрипт каждые 10 секунд формирует окно, сохраняет JSONL рядом с видео и сразу отправляет каждое окно в API `POST /api/windows/batch`. Флаг `SHOW_PREVIEW=1` можно выставить в окружении, если нужен показ окна OpenCV.

## 4. Быстрая проверка API
- Отправить тестовую пачку окон:
```bash
python scripts/sample_client.py
```
- Проверить соединение с SQLite:
```bash
python backend/test.py
```
- Примеры ручных запросов (после запуска uvicorn):
```
GET http://127.0.0.1:8000/api/windows
GET http://127.0.0.1:8000/api/metrics/current_state
```

## 5. Фронтенд/дашборд
- Статика в `frontend/`. Запуск локального сервера:
```bash
cd frontend
python -m http.server 5500
```
- Открыть в браузере `http://127.0.0.1:5500`. Данные берутся из API (SQLite).

## 6. Краткий сценарий end-to-end
1) Активировать venv, установить зависимости.  
2) Запустить API: `python -m uvicorn backend.main:app --reload`.  
3) (Опционально) `python scripts/init_db.py`.  
4) Запустить CV: `python runfile/log.py` (или `scripts/sample_client.py` для быстрой проверки).  
5) Подождать минуту, чтобы фоновые flush записали окна в БД (или просто перезапустить API для моментального flush).  
6) Открыть фронт `http://127.0.0.1:5500` и обновить данные.  
7) веса для моделей лежать на google disk(https://drive.google.com/file/d/1o7w33_CoOzq3pADGRIt8d06bMVk8W_Vy/view?usp=drive_link)
## 7. Замечания
- Все пути и команды рассчитаны на Windows PowerShell; под Unix замените активацию venv и слэши путей.
- Если нужно использовать другую БД или порт, задайте `DATABASE_URL` перед запуском uvicorn.
- Если API недоступен, `runfile/log.py` всё равно сохранит JSONL, но не сможет отправить окна.
