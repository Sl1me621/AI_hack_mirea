import sqlite3
import json
import os

# ==== ПУТИ К ФАЙЛАМ ====
BASE_DIR = r"C:\Users\PC\OneDrive\Документы\hackgleb\AI_hack_mirea"
LOG_PATH = rf"{BASE_DIR}\logs\windows_simple.jsonl"
DB_PATH = rf"{BASE_DIR}\windows.db"


# ==== СОЗДАНИЕ ТАБЛИЦЫ ЕСЛИ ЕЕ НЕТ ====
def init_db(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS window_analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            video_id TEXT,
            window_index INTEGER,
            start_time_sec REAL,
            end_time_sec REAL,

            workers_in_safe INTEGER,
            workers_in_medium INTEGER,
            workers_in_danger INTEGER,

            cleaning_workers_count INTEGER,

            dangerous_situations_count INTEGER,
            violations_count INTEGER,

            train_present INTEGER,
            train_arrived_in_window INTEGER,
            train_departed_in_window INTEGER,

            train_number TEXT
        );
    """)
    conn.commit()


# ==== ИМПОРТ ИЗ JSONL В БД ====
def import_logs():
    if not os.path.exists(LOG_PATH):
        print("Файл логов не найден:", LOG_PATH)
        return

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    cur = conn.cursor()

    print("Импорт логов из:", LOG_PATH)

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            cur.execute("""
                INSERT INTO window_analytics (
                    video_id,
                    window_index,
                    start_time_sec,
                    end_time_sec,

                    workers_in_safe,
                    workers_in_medium,
                    workers_in_danger,

                    cleaning_workers_count,

                    dangerous_situations_count,
                    violations_count,

                    train_present,
                    train_arrived_in_window,
                    train_departed_in_window,
                    train_number
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data.get("video_id"),
                data.get("window_index"),
                data.get("start_time_sec"),
                data.get("end_time_sec"),

                data.get("workers_in_safe"),
                data.get("workers_in_medium"),
                data.get("workers_in_danger"),

                data.get("cleaning_workers_count"),

                data.get("dangerous_situations_count"),
                data.get("violations_count"),

                int(bool(data.get("train_present"))),
                int(bool(data.get("train_arrived_in_window"))),
                int(bool(data.get("train_departed_in_window"))),

                data.get("train_number"),
            ))

    conn.commit()
    conn.close()

    print("Импорт завершён. БД создана/обновлена:", DB_PATH)


if __name__ == "__main__":
    import_logs()
