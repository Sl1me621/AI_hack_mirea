import cv2  # OpenCV — работа с видео/кадрами, чтение ролика, отрисовка боксов/текста и т.п.
import math  # math — математика: sqrt/hypot, тригонометрия, ceil и т.д.
import os  # os — работа с путями, возьмём имя видеофайла для поля video_id
import json  # json — сериализация словарей Python в JSON-строки (для логов)
import numpy as np  # numpy — быстрые массивы/линейная алгебра, используем для координат и кейпоинтов
from ultralytics import YOLO  # ultralytics.YOLO — модель детекции/трекинга/pose (YOLOv8/11)

# сколько секунд покрывает одно лог-окно
WINDOW_SIZE_SEC = 10.0  # 10 секунд на одно окно

# до какого времени анализируем видео (требуется только первые 60 секунд)
MAX_ANALYZE_SEC = 60.0  # берём только первые 60 секунд ролика
# ==== НАСТРОЙКИ ====
DETECT_MODEL_PATH = "weights/yolo11_x/weights/best.pt"  # твоя обученная детекция (классы людей+поезд)
POSE_MODEL_PATH   = "yolo11s-pose.pt"                      # предобученная pose-модель (без дообучения)
VIDEO_PATH        = "C:/Users/User/Desktop/rem.mov"

MAX_MISSED_SECONDS   = 4.0     # разрыв для трека (секунд)
MAX_REASSOC_DIST     = 100.0   # макс. расстояние (px) для сопоставления по координатам
SCENE_DIFF_THRESHOLD = 40.0    # чувствительность перескока сцены

# поезд движется, если низ бокса смещается быстрее этого порога
TRAIN_MOVE_SPEED_THR = 15.0    # пикс/сек по y-низу bbox

# ==== ЗОНЫ ====
ZONES = {
    "green_safe": [
        (0, 719),
        (513, 718),
        (539, 546),
        (553, 441),
        (567, 341),
        (583, 251),
        (591, 203),
        (584, 196),
        (0, 196),
    ],
    "yellow_warning": [
        (511, 719), (533, 588), (548, 480), (560, 381),
        (575, 289), (585, 225), (593, 182), (594, 155),
        (616, 150), (621, 175), (627, 211), (639, 265),
        (652, 322), (665, 385), (691, 503), (747, 718),
    ],
    "red_danger": [
        (751, 719), (1279, 717), (1279, 316), (1142, 262),
        (1011, 216), (900, 183), (809, 155), (740, 133),
        (707, 126), (679, 131), (644, 131), (639, 149),
        (618, 152), (640, 262), (688, 477), (745, 718),
    ],
}

ZONE_COLORS = {
    "green_safe":     (0, 255, 0),
    "yellow_warning": (0, 255, 255),
    "red_danger":     (0, 0, 255),
    "none":           (200, 200, 200),
}

# ==== РОЛИ ПО КЛАССАМ YOLO ====
# проверь, что ID классов совпадают с твоей моделью!
# 0: noname
# 1: person_blue
# 2: person_gray
# 3: person_orange
# 4: person_red
# 5: person_white
# 6: train
def class_to_role(cls_id: int) -> str:
    mapping = {
        0: "noname",
        1: "person_blue",
        2: "person_gray",
        3: "person_orange",
        4: "person_red",
        5: "person_white",
        6: "train",
    }
    return mapping.get(cls_id, "unknown")


ROLE_RULES = {
    "noname": {
        "allowed_zones": [],
    },
    "person_blue": {
        "allowed_zones": ["green_safe", "yellow_warning", "red_danger"],
    },
    "person_gray": {
        "allowed_zones": ["green_safe", "yellow_warning", "red_danger"],
    },
    "person_orange": {
        # уборщик — теперь и в красную зону можно, но потом пометим как "опасность, непонятные действия"
        "allowed_zones": ["green_safe", "yellow_warning", "red_danger"],
    },
    "person_red": {
        "allowed_zones": ["green_safe", "yellow_warning"],
    },
    "person_white": {
        "allowed_zones": ["green_safe", "yellow_warning", "red_danger"],
    },
    "train": {
        "allowed_zones": ["green_safe", "yellow_warning", "red_danger"],
    },
    "unknown": {
        "allowed_zones": [],
    },
}


def draw_zones(frame):
    overlay = frame.copy()
    alpha = 0.30
    for name, poly in ZONES.items():
        pts = np.array(poly, dtype=np.int32)
        color = ZONE_COLORS.get(name, (255, 255, 255))
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(overlay, [pts], True, color, 3)
        x0, y0 = pts[0]
        cv2.putText(
            overlay, name, (int(x0), int(y0) - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
        )
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def point_in_polygon(x, y, polygon):
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        intersect = ((y1 > y) != (y2 > y)) and \
                    (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1)
        if intersect:
            inside = not inside
    return inside


def get_zone_for_point(x, y):
    for name, poly in ZONES.items():
        if point_in_polygon(x, y, poly):
            return name
    return "none"


# ====== ПОЗА (uprilght / bent) через YOLO Pose ======

def infer_posture_from_kpts(kpts_xy: np.ndarray) -> str:
    """
    kpts_xy: [K, 2] массив (x, y) точек.
    Используем mid-shoulder и mid-hip, считаем угол корпуса к вертикали.
    """
    try:
        # индексы под COCO: 5,6 - плечи; 11,12 - бёдра
        ls = kpts_xy[5]
        rs = kpts_xy[6]
        lh = kpts_xy[11]
        rh = kpts_xy[12]

        if (ls == 0).all() or (rs == 0).all() or (lh == 0).all() or (rh == 0).all():
            return "unknown"

        shoulder = (ls + rs) / 2.0
        hip = (lh + rh) / 2.0

        vec = shoulder - hip
        dx, dy = float(vec[0]), float(vec[1])

        # вертикаль направлена вверх (0, -1)
        angle = abs(np.degrees(np.arctan2(dx, -dy)))  # 0 = строго вверх, 90 = горизонталь

        if angle < 25:
            return "upright"
        elif angle > 45:
            return "bent"
        else:
            return "unknown"
    except Exception:
        return "unknown"


def extract_poses(frame, pose_model):
    """
    Возвращает список поз:
    [{"bbox": (x1,y1,x2,y2), "pose": "upright"/"bent"/"unknown"}]
    """
    pose_results = pose_model.predict(
        frame,
        imgsz=640,
        conf=0.25,
        verbose=False,
        device=0,
    )
    poses = []
    r_pose = pose_results[0]

    if r_pose.keypoints is None or r_pose.boxes is None:
        return poses

    kpts_xy = r_pose.keypoints.xy.cpu().numpy()  # [N, K, 2]

    for i, box in enumerate(r_pose.boxes):
        x1p, y1p, x2p, y2p = box.xyxy[0].tolist()
        k = kpts_xy[i]  # [K, 2]
        posture = infer_posture_from_kpts(k)
        poses.append({
            "bbox": (x1p, y1p, x2p, y2p),
            "pose": posture,
        })
    return poses


def match_pose_to_box(x1, y1, x2, y2, poses, max_dist=80.0):
    """
    Ищем ближайшую позу по центру бокса.
    """
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    best_pose = None
    best_d = 1e9

    for p in poses:
        px1, py1, px2, py2 = p["bbox"]
        pcx = (px1 + px2) / 2.0
        pcy = (py1 + py2) / 2.0
        d = math.hypot(cx - pcx, cy - pcy)
        if d < best_d and d < max_dist:
            best_d = d
            best_pose = p

    return best_pose


def main():
    # ===== подготовка FPS =====
    cap = cv2.VideoCapture(VIDEO_PATH)  # открываем видео, чтобы прочитать FPS
    fps = cap.get(cv2.CAP_PROP_FPS)  # частота кадров
    cap.release()
    if fps <= 0:
        fps = 30.0  # запасной вариант, если в метаданных FPS нет

    max_missed_frames = int(MAX_MISSED_SECONDS * fps)  # максимум пропущенных кадров для трека
    print(f"Video FPS = {fps:.2f}, max_missed_frames = {max_missed_frames}")

    # ===== подготовка окон логов (0–60 сек) =====
    # количество окон по 10 секунд, которое перекроет первые 60 секунд
    num_windows = int(math.ceil(MAX_ANALYZE_SEC / WINDOW_SIZE_SEC))

    # внутри будем хранить множества id треков, чтобы считать людей по уникальным SID
    log_windows = []
    for i in range(num_windows):
        log_windows.append({
            "window_index": i,
            "start_time_sec": i * WINDOW_SIZE_SEC,
            "end_time_sec": (i + 1) * WINDOW_SIZE_SEC,
            "workers_in_safe_ids": set(),      # уникальные SID работников в green_safe
            "workers_in_medium_ids": set(),    # уникальные SID работников в yellow_warning
            "workers_in_danger_ids": set(),    # уникальные SID работников в red_danger
            "cleaning_workers_ids": set(),     # уникальные SID уборщиков (person_orange)
            "dangerous_situations_count": 0,   # число опасных ситуаций (нарушение или warning)
            "violations_count": 0,             # число нарушений (строгие нарушения правил)
            "train_present": False,            # был ли поезд в данном окне
        })

    # ===== загрузка моделей YOLO =====
    detect_model = YOLO(DETECT_MODEL_PATH)  # модель детекции/трекинга людей и поезда
    pose_model = YOLO(POSE_MODEL_PATH)     # модель позы для классификации «upright/bent»

    # поток трекинга от Ultralytics (генератор по кадрам)
    results = detect_model.track(
        source=VIDEO_PATH,
        imgsz=640,
        conf=0.3,
        iou=0.5,
        tracker="bytetrack.yaml",
        persist=True,
        device=0,
        stream=True,
        verbose=False,
    )

    # ===== собственный ID-трекинг поверх raw id от трекера =====
    raw_to_stable = {}   # сопоставление "сырой" id (из трекера) -> стабильный sid
    stable_tracks = {}   # sid -> состояние трека
    next_stable_id = 0   # следующий свободный sid

    # события по кадрам (штучные события, как раньше)
    events = []

    # состояние поездов по сегментам
    train_state = {}  # ключ (segment_id, sid) -> словарь состояния поезда

    # сегменты (по сценам/перескокам)
    segment_id = 0
    prev_gray = None
    frame_idx = -1

    stop_processing = False  # флаг, чтобы остановиться после 60 секунд

    # ===== основной цикл по кадрам =====
    for r in results:
        if stop_processing:
            break

        frame_idx += 1
        current_time = frame_idx / fps if fps > 0 else 0.0  # время текущего кадра в секундах

        # анализируем только первые 60 секунд, дальше выходим из цикла
        if current_time >= MAX_ANALYZE_SEC:
            stop_processing = True
            break

        # берём кадр
        if r.orig_img is not None:
            frame = r.orig_img.copy()
        else:
            frame = r.plot()

        # рисуем зоны поверх кадра
        draw_zones(frame)

        # ===== обнаружение смены сцены (scene cut) =====
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            mean_diff = diff.mean()
            if mean_diff > SCENE_DIFF_THRESHOLD:
                segment_id += 1
                raw_to_stable.clear()
                stable_tracks.clear()
                train_state.clear()
                next_stable_id = 0
                print(
                    f"\n=== SCENE CUT at frame {frame_idx}, "
                    f"mean_diff={mean_diff:.1f}, new segment_id={segment_id} ===\n"
                )
        prev_gray = gray

        # ===== позы на кадре =====
        poses = extract_poses(frame, pose_model)

        used_sids_this_frame = set()
        train_moving_this_frame = False  # флаг, что на этом кадре поезд движется

        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                raw_id = int(box.id[0]) if box.id is not None else None

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                # === привязка к стабильному sid ===
                sid = None

                # 1) пробуем взять уже сопоставленный sid по raw_id
                if raw_id is not None and raw_id in raw_to_stable:
                    cand_sid = raw_to_stable[raw_id]
                    track = stable_tracks.get(cand_sid)
                    if track is not None and frame_idx - track["last_frame"] <= max_missed_frames:
                        sid = cand_sid

                # 2) если не нашли, пробуем сопоставить по расстоянию до прошлых центров
                if sid is None:
                    best_sid = None
                    best_dist = 1e9
                    for cand_sid, track in stable_tracks.items():
                        if frame_idx - track["last_frame"] > max_missed_frames:
                            continue
                        if cand_sid in used_sids_this_frame:
                            continue
                        dist = math.hypot(cx - track["cx"], cy - track["cy"])
                        if dist < best_dist and dist < MAX_REASSOC_DIST:
                            best_dist = dist
                            best_sid = cand_sid
                    if best_sid is not None:
                        sid = best_sid

                # 3) если всё ещё не нашли — создаём новый sid
                if sid is None:
                    sid = next_stable_id
                    next_stable_id += 1
                    stable_tracks[sid] = {
                        "cx": cx,
                        "cy": cy,
                        "last_frame": frame_idx,
                        "cls": cls_id,
                        "history": [],
                    }
                else:
                    # обновляем существующий трек
                    stable_tracks[sid]["cx"] = cx
                    stable_tracks[sid]["cy"] = cy
                    stable_tracks[sid]["last_frame"] = frame_idx
                    stable_tracks[sid]["cls"] = cls_id

                used_sids_this_frame.add(sid)
                if raw_id is not None:
                    raw_to_stable[raw_id] = sid

                # ===== зона + роль =====
                zone = get_zone_for_point(cx, cy)
                role = class_to_role(cls_id)
                allowed_zones = ROLE_RULES.get(role, ROLE_RULES["unknown"])["allowed_zones"]

                # ===== поза =====
                pose_info = match_pose_to_box(x1, y1, x2, y2, poses)
                posture = pose_info["pose"] if pose_info is not None else "unknown"

                # сохраняем историю (при желании можно использовать потом)
                stable_tracks[sid]["history"].append({
                    "segment": segment_id,
                    "frame": frame_idx,
                    "time_sec": current_time,
                    "cx": cx,
                    "cy": cy,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cls": cls_id,
                    "conf": conf,
                    "zone": zone,
                    "role": role,
                    "posture": posture,
                })

                # ===== движение поезда по низу бокса =====
                if role == "train":
                    key_train = (segment_id, sid)
                    bottom_y = y2
                    ts = train_state.get(key_train)
                    if ts is None:
                        ts = {
                            "last_frame": frame_idx,
                            "last_bottom_y": bottom_y,
                            "moving": False,
                        }
                        train_state[key_train] = ts
                    else:
                        dt_frames_t = frame_idx - ts["last_frame"]
                        dt_t = dt_frames_t / fps if fps > 0 else 0.0
                        dy = bottom_y - ts["last_bottom_y"]
                        speed_t = abs(dy) / dt_t if dt_t > 0 else 0.0
                        ts["moving"] = speed_t > TRAIN_MOVE_SPEED_THR
                        ts["last_frame"] = frame_idx
                        ts["last_bottom_y"] = bottom_y
                    if ts["moving"]:
                        train_moving_this_frame = True

                # ===== проверки нарушений зон и роли =====
                violation = False
                violation_reason = None
                warnings = []

                # noname — всегда нарушение
                if role == "noname":
                    violation = True
                    violation_reason = "FORBIDDEN_ROLE_NONAME"
                    events.append({
                        "type": violation_reason,
                        "segment": segment_id,
                        "sid": sid,
                        "role": role,
                        "zone": zone,
                        "time_sec": current_time,
                    })
                else:
                    # базовое правило: зона не из allowed_zones -> нарушение
                    if zone != "none" and zone not in allowed_zones:
                        violation = True
                        violation_reason = "ZONE_FORBIDDEN_FOR_ROLE"
                        events.append({
                            "type": violation_reason,
                            "segment": segment_id,
                            "sid": sid,
                            "role": role,
                            "zone": zone,
                            "time_sec": current_time,
                        })

                # поезд движется -> в красной зоне никто кроме поезда
                if zone == "red_danger" and role != "train" and train_moving_this_frame:
                    violation = True
                    violation_reason = "TRAIN_MOVING_RED_FORBIDDEN"
                    events.append({
                        "type": violation_reason,
                        "segment": segment_id,
                        "sid": sid,
                        "role": role,
                        "zone": zone,
                        "time_sec": current_time,
                    })

                # оранжевые в красной: разрешено, но «опасность, непонятные действия»
                if zone == "red_danger" and role == "person_orange" and not violation:
                    warnings.append("ORANGE_IN_RED_UNCLEAR")
                    events.append({
                        "type": "ORANGE_IN_RED_UNCLEAR",
                        "segment": segment_id,
                        "sid": sid,
                        "role": role,
                        "zone": zone,
                        "time_sec": current_time,
                    })

                # позные правила:
                # уборщик (orange) и белый — ожидаем нагибаться
                if role in ("person_orange", "person_white") and posture != "bent":
                    warnings.append("EXPECTED_BENT")
                    events.append({
                        "type": "EXPECTED_BENT_NOT_SEEN",
                        "segment": segment_id,
                        "sid": sid,
                        "role": role,
                        "zone": zone,
                        "posture": posture,
                        "time_sec": current_time,
                    })

                # красный должен ходить, не нагибаться
                if role == "person_red" and posture == "bent":
                    warnings.append("RED_BENT_UNEXPECTED")
                    events.append({
                        "type": "RED_BENT_UNEXPECTED",
                        "segment": segment_id,
                        "sid": sid,
                        "role": role,
                        "zone": zone,
                        "posture": posture,
                        "time_sec": current_time,
                    })

                # ===== агрегация по временным окнам для логов (первые 60 секунд) =====
                if fps > 0:
                    win_idx = int(current_time // WINDOW_SIZE_SEC)
                else:
                    win_idx = 0

                if 0 <= win_idx < num_windows:
                    w = log_windows[win_idx]

                    # распределяем работников по зонам (по уникальным SID)
                    if role != "train":
                        if zone == "green_safe":
                            w["workers_in_safe_ids"].add(sid)
                        elif zone == "yellow_warning":
                            w["workers_in_medium_ids"].add(sid)
                        elif zone == "red_danger":
                            w["workers_in_danger_ids"].add(sid)

                        # уборщиков считаем отдельно (оранжевые жилеты)
                        if role == "person_orange":
                            w["cleaning_workers_ids"].add(sid)

                    # факт присутствия поезда в этом окне
                    if role == "train":
                        w["train_present"] = True

                    # опасные ситуации: любое нарушение или warning
                    if violation or warnings:
                        w["dangerous_situations_count"] += 1
                        if violation:
                            w["violations_count"] += 1

                # ===== визуализация =====
                # все боксы зелёные, нарушение -> красный
                if violation:
                    box_color = (0, 0, 255)
                else:
                    box_color = (0, 255, 0)

                # рисуем прямоугольник объекта
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    box_color,
                    2,
                )

                danger_text = " DANGER" if violation else ""
                warn_text = ""
                if warnings and not violation:
                    warn_text = " WARN"

                label = f"id={sid} r={role} z={zone} pose={posture}{danger_text}{warn_text}"
                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    box_color,
                    1,
                    cv2.LINE_AA,
                )

                print(
                    f"t={current_time:.2f}s seg={segment_id} fr={frame_idx} sid={sid} "
                    f"role={role} zone={zone} posture={posture} "
                    f"viol={violation} reason={violation_reason} warnings={warnings} conf={conf:.2f}"
                )

        # показываем кадр (можешь убрать, если только логи нужны)
        cv2.imshow("YOLO + Zones + Roles + Pose", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

    # ===== вывод событий (как раньше) =====
    print("\n=== EVENTS (for DB / dashboard) ===")
    for e in events:
        print(e)

    # ===== итоговые логи по окнам (0–60 сек) в формате JSONL =====
    # video_id берём из имени файла (без расширения), можно заменить на своё значение
    video_id = os.path.splitext(os.path.basename(VIDEO_PATH))[0]

    print("\n=== WINDOW LOGS (first 60 seconds) ===")
    prev_train_present = False
    for w in log_windows:
        record = {
            "video_id": video_id,
            "window_index": w["window_index"],
            "start_time_sec": w["start_time_sec"],
            "end_time_sec": w["end_time_sec"],
            "workers_in_safe": len(w["workers_in_safe_ids"]),
            "workers_in_medium": len(w["workers_in_medium_ids"]),
            "workers_in_danger": len(w["workers_in_danger_ids"]),
            "cleaning_workers_count": len(w["cleaning_workers_ids"]),
            "dangerous_situations_count": w["dangerous_situations_count"],
            "violations_count": w["violations_count"],
            "train_present": w["train_present"],
            "train_arrived_in_window": False,
            "train_departed_in_window": False,
            "train_number": None,  # номер поезда сейчас не извлекаем, оставляем null
        }

        # отмечаем в каком окне поезд появился/исчез
        if w["train_present"] and not prev_train_present:
            record["train_arrived_in_window"] = True
        if prev_train_present and not w["train_present"]:
            record["train_departed_in_window"] = True
        prev_train_present = w["train_present"]

        # печатаем строку JSON — можно перенаправить в .jsonl файл
        print(json.dumps(record, ensure_ascii=False))

    # ===== сохранение логов в файл JSONL =====
    output_path = os.path.join(os.path.dirname(VIDEO_PATH), f"{video_id}_logs.jsonl")

    with open(output_path, "w", encoding="utf-8") as f:
        for w in log_windows:
            record = {
                "video_id": video_id,
                "window_index": w["window_index"],
                "start_time_sec": w["start_time_sec"],
                "end_time_sec": w["end_time_sec"],
                "workers_in_safe": len(w["workers_in_safe_ids"]),
                "workers_in_medium": len(w["workers_in_medium_ids"]),
                "workers_in_danger": len(w["workers_in_danger_ids"]),
                "cleaning_workers_count": len(w["cleaning_workers_ids"]),
                "dangerous_situations_count": w["dangerous_situations_count"],
                "violations_count": w["violations_count"],
                "train_present": w["train_present"],
                "train_arrived_in_window": False,
                "train_departed_in_window": False,
                "train_number": None,
            }

            # помечаем прибытие/уход поезда
            if w["window_index"] > 0:
                prev = log_windows[w["window_index"] - 1]
                if w["train_present"] and not prev["train_present"]:
                    record["train_arrived_in_window"] = True
                if prev["train_present"] and not w["train_present"]:
                    record["train_departed_in_window"] = True

            # запись в файл
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nЛоги сохранены в файл: {output_path}")


if __name__ == "__main__":
    main()
