import os
from argparse import ArgumentParser
from functools import lru_cache
from pathlib import Path
import sys

import cv2
import easyocr
import numpy as np
from sqlalchemy import or_

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.db import SessionLocal
from backend.orm_models import WindowAnalytics

ROI_TRAIN = (0.58, 0.45, 0.98, 0.90)


@lru_cache(maxsize=1)
def get_reader():
    print("Инициализация EasyOCR Reader (рус+англ, GPU=True)...")
    return easyocr.Reader(["en", "ru"], gpu=True)


def crop_relative(img: np.ndarray, roi: tuple) -> np.ndarray:
    h, w = img.shape[:2]
    x1_rel, y1_rel, x2_rel, y2_rel = roi
    x1 = int(x1_rel * w)
    y1 = int(y1_rel * h)
    x2 = int(x2_rel * w)
    y2 = int(y2_rel * h)
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Некорректный ROI: {roi} -> ({x1},{y1})-({x2},{y2})")
    return img[y1:y2, x1:x2]


def preprocess_for_train_ocr(roi_bgr: np.ndarray, scale: float = 2.5) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    gray_big = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    gray_blur = cv2.GaussianBlur(gray_big, (3, 3), 0)
    return gray_blur


TRAIN_CHAR_MAP = {
    "З": "Э",
    "3": "Э",
    "О": "0",
    "O": "0",
    "о": "0",
}


def normalize_train_text(raw: str) -> str:
    chars = []
    for ch in raw:
        if ch.isalnum() or ch.isspace():
            chars.append(ch)
        else:
            chars.append(" ")
    cleaned = "".join(chars)
    cleaned = " ".join(cleaned.split())
    mapped_chars = []
    for ch in cleaned:
        if ch in TRAIN_CHAR_MAP:
            mapped_chars.append(TRAIN_CHAR_MAP[ch])
        else:
            mapped_chars.append(ch)
    normalized = "".join(mapped_chars)
    tokens = normalized.split()
    if len(tokens) >= 2:
        normalized = " ".join(tokens[:2])
    return normalized


def read_train_number_from_screen(screen_path: str = "screen.png") -> str | None:
    if not os.path.exists(screen_path):
        raise FileNotFoundError(f"Файл с кадром не найден: {screen_path}")
    img = cv2.imread(screen_path)
    if img is None:
        raise RuntimeError(f"OpenCV не смог прочитать {screen_path}")
    print("Размер кадра:", img.shape)
    roi = crop_relative(img, ROI_TRAIN)
    cv2.imwrite("debug_train_roi.jpg", roi)
    pre = preprocess_for_train_ocr(roi, scale=2.5)
    cv2.imwrite("debug_train_pre.jpg", pre)
    reader = get_reader()
    results = reader.readtext(pre, detail=0)
    print("\n=== OCR номер поезда (screen + ROI_TRAIN) ===")
    print("Сырые строки OCR:", results)
    if not results:
        print("EasyOCR не распознал текст в ROI — пустой список.")
        return None
    joined_raw = " ".join(results)
    print("joined_raw =", repr(joined_raw))
    normalized = normalize_train_text(joined_raw)
    print("train_number_normalized =", repr(normalized))
    if not normalized:
        print("После нормализации номер пустой.")
        return None
    return normalized


def write_train_number_to_db(video_id: str, train_number: str) -> int:
    session = SessionLocal()
    try:
        rows = (
            session.query(WindowAnalytics)
            .filter(WindowAnalytics.video_id == video_id)
            .filter(or_(WindowAnalytics.train_number.is_(None), WindowAnalytics.train_number == ""))
            .all()
        )
        for row in rows:
            row.train_number = train_number
        session.commit()
        return len(rows)
    finally:
        session.close()


def main():
    parser = ArgumentParser()
    parser.add_argument("--image-path", default="screen.png")
    parser.add_argument("--video-id", default="session_1")
    args = parser.parse_args()
    train_number = read_train_number_from_screen(args.image_path)
    updated = 0
    if train_number:
        updated = write_train_number_to_db(args.video_id, train_number)
    print("\n=== ИТОГ ===")
    print("train_number =", repr(train_number))
    print("updated rows:", updated)
    print("Сгенерированные отладочные файлы:")
    print("  debug_train_roi.jpg  — вырезанный фрагмент с поездом/номером")
    print("  debug_train_pre.jpg  — тот же фрагмент после предобработки (gray+апскейл+blur)")


if __name__ == "__main__":
    main()
