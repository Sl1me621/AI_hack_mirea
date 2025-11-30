# run_track_log_final.py
import cv2
import math
import numpy as np
from ultralytics import YOLO
import os
import json
from collections import deque

# === БАЗОВАЯ ПАПКА ПРОЕКТА ===
BASE_DIR = r"C:\Users\PC\OneDrive\Документы\hackgleb\AI_hack_mirea"

# ==== ПУТИ ====
DETECT_MODEL_PATH = rf"{BASE_DIR}\weights\runs\yolo11_x\weights\best.pt"
POSE_MODEL_PATH   = rf"{BASE_DIR}\weights\yolo11x-pose.pt"
VIDEO_PATH        = rf"{BASE_DIR}\data\ремонты.mov"

LOG_PATH          = rf"{BASE_DIR}\logs\windows_simple.jsonl"
WINDOW_SIZE_SEC   = 10.0
START_TIME_SEC    = 0   # откуда стартуем просмотр

# КУДА СОХРАНЯЕМ АННОТИРОВАННОЕ ВИДЕО
VIDEO_ID = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_VIDEO_PATH = rf"{BASE_DIR}\output\{VIDEO_ID}_annotated.mp4"

# интервалы, когда поезд точно едет (для прототипа / съёмки видео)
TRAIN_MOVING_INTERVALS = [
    (0.0,   112.0),
    (696.0, 750.0),
]

def is_train_moving_time(t_sec: float) -> bool:
    for s, e in TRAIN_MOVING_INTERVALS:
        if s <= t_sec <= e:
            return True
    return False

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

SAFE_ZONES   = {"green_safe"}
MEDIUM_ZONES = {"yellow_warning"}
DANGER_ZONES = {"red_danger"}

# ==== РОЛИ ПО КЛАССАМ YOLO ====
# 0: noname
# 1: person_blue
# 2: person_gray
# 3: person_orange  (уборщик)
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
    "noname": {"allowed_zones": []},
    "person_blue": {"allowed_zones": ["green_safe", "yellow_warning", "red_danger"]},
    "person_gray": {"allowed_zones": ["green_safe", "yellow_warning", "red_danger"]},
    "person_orange": {"allowed_zones": ["green_safe", "yellow_warning", "red_danger"]},
    "person_red": {"allowed_zones": ["green_safe", "yellow_warning"]},
    "person_white": {"allowed_zones": ["green_safe", "yellow_warning", "red_danger"]},
    "train": {"allowed_zones": ["green_safe", "yellow_warning", "red_danger"]},
    "unknown": {"allowed_zones": []},
}

# ====== ВСПОМОГАТЕЛЬНОЕ: зоны ======

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


def draw_zones(frame):
    overlay = frame.copy()
    alpha = 0.25
    colors = {
        "green_safe":     (150, 220, 180),
        "yellow_warning": (120, 220, 255),
        "red_danger":     (140, 140, 255),
    }
    for name, poly in ZONES.items():
        pts = np.array(poly, dtype=np.int32)
        color = colors.get(name, (255, 255, 255))
        cv2.polylines(overlay, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.fillPoly(overlay, [pts], color)
        x0, y0 = pts[0]
        label = {
            "green_safe": "SAFE",
            "yellow_warning": "WARNING",
            "red_danger": "DANGER",
        }.get(name, name)
        cv2.putText(frame, label, (int(x0) + 10, int(y0) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (int(x0) + 10, int(y0) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

# ====== ПОЗА: улучшенный присед ======

def infer_posture_from_kpts(kpts_xy: np.ndarray) -> str:
    """
    Используем плечи, бёдра, колени:
    - считаем угол корпуса (плечи-таз) к вертикали;
    - смотрим соотношение длины бедра к длине торса, чтобы выловить присед.
    """
    try:
        ls = kpts_xy[5]
        rs = kpts_xy[6]
        lh = kpts_xy[11]
        rh = kpts_xy[12]
        lk = kpts_xy[13]
        rk = kpts_xy[14]

        pts = [ls, rs, lh, rh, lk, rk]
        if any(np.all(p == 0) for p in pts):
            return "unknown"

        shoulder = (ls + rs) / 2.0
        hip = (lh + rh) / 2.0
        knee = (lk + rk) / 2.0

        dx, dy = float(shoulder[0] - hip[0]), float(shoulder[1] - hip[1])
        if dx == 0 and dy == 0:
            return "unknown"

        angle = abs(np.degrees(np.arctan2(dx, -dy)))  # к вертикали

        torso_len = abs(shoulder[1] - hip[1])
        upper_leg_len = abs(knee[1] - hip[1])

        if torso_len < 5:
            return "unknown"

        ratio = upper_leg_len / torso_len  # насколько таз приблизился к коленям

        if ratio < 0.6:
            return "squat"

        if angle < 25:
            return "upright"
        elif angle > 45:
            return "bent"
        else:
            return "upright"
    except Exception:
        return "unknown"


def extract_poses(frame, pose_model):
    pose_results = pose_model.predict(
        frame,
        imgsz=640,
        conf=0.25,
        verbose=False,
        device=0,
        half=True,
    )
    poses = []
    r_pose = pose_results[0]
    if r_pose.keypoints is None or r_pose.boxes is None:
        return poses
    kpts_xy = r_pose.keypoints.xy.cpu().numpy()
    for i, box in enumerate(r_pose.boxes):
        x1p, y1p, x2p, y2p = box.xyxy[0].tolist()
        posture = infer_posture_from_kpts(kpts_xy[i])
        poses.append({"bbox": (x1p, y1p, x2p, y2p), "posture": posture})
    return poses


def match_pose_to_box(x1, y1, x2, y2, poses, max_dist=80.0):
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    best = None
    best_d = 1e9
    for p in poses:
        px1, py1, px2, py2 = p["bbox"]
        pcx = (px1 + px2) / 2.0
        pcy = (py1 + py2) / 2.0
        d = math.hypot(cx - pcx, cy - pcy)
        if d < best_d and d < max_dist:
            best_d = d
            best = p
    return best

# ====== движение людей: walk / stand / squat c траекторией ======

def update_motion_state(track, fps: float,
                        move_on: float = 0.12,
                        move_off: float = 0.06):
    """
    Смортим траекторию за последние ~0.8 сек.
    Если человек идёт по прямой — путь большой, он = walk.
    """
    hist = track["history"]
    n = len(hist)
    if n < 2:
        track["speed_norm"] = 0.0
        track["is_moving"] = False
        return

    window_sec = 0.8
    last_t = hist[-1]["time_sec"]

    pts = []
    for h in reversed(hist):
        if last_t - h["time_sec"] <= window_sec:
            pts.append(h)
        else:
            break

    if len(pts) < 2:
        track["speed_norm"] = 0.0
        track["is_moving"] = False
        return

    pts = list(reversed(pts))
    dt = pts[-1]["time_sec"] - pts[0]["time_sec"]
    if dt < 0.3:
        track["speed_norm"] = 0.0
        track["is_moving"] = False
        return

    path_len = 0.0
    heights = []
    for i in range(1, len(pts)):
        dx = pts[i]["cx"] - pts[i - 1]["cx"]
        dy = pts[i]["cy"] - pts[i - 1]["cy"]
        path_len += math.hypot(dx, dy)
        heights.append(pts[i]["y2"] - pts[i]["y1"])

    if path_len < 2.0:
        speed_norm = 0.0
    else:
        speed_raw = path_len / dt
        h_box = float(np.median(heights)) if heights else (pts[-1]["y2"] - pts[-1]["y1"])
        if h_box <= 1:
            speed_norm = 0.0
        else:
            speed_norm = speed_raw / h_box

    track["speed_norm"] = speed_norm
    was = track.get("is_moving", False)

    if not was:
        track["is_moving"] = speed_norm > move_on
    else:
        track["is_moving"] = speed_norm > move_off


def classify_action(track) -> str:
    hist = track["history"]
    if not hist:
        return "stand"
    last = hist[-1]
    posture = last.get("posture", "unknown")
    moving = track.get("is_moving", False)
    if moving:
        return "walk"
    if posture == "squat":
        return "squat"
    return "stand"

# ====== движение поезда (очень медленное) ======

class TrainTracker:
    """
    Трекер поезда под очень медленное движение.
    - берём нижний центр bbox;
    - считаем сдвиг по диагонали за 5 сек;
    - очень маленькие пороги.
    """
    def __init__(self, fps, img_w, img_h):
        self.fps = fps
        self.img_w = img_w
        self.img_h = img_h
        self.diag = math.hypot(img_w, img_h)
        self.state = {}  # track_id -> {"hist": deque[(t,(x,y))], "moving": bool}

        self.history_sec = 5.0
        self.min_dt = 1.0

        self.on_thr = 0.0015   # ~0.15% диагонали/сек
        self.off_thr = 0.0007  # ~0.07%

    def update(self, track_id, t_sec, bottom_cx, bottom_cy):
        s = self.state.get(track_id)
        if s is None:
            s = {"hist": deque(), "moving": False}
            self.state[track_id] = s
        hist = s["hist"]
        hist.append((t_sec, (bottom_cx, bottom_cy)))

        while hist and (t_sec - hist[0][0]) > self.history_sec:
            hist.popleft()

        moving = s["moving"]

        if len(hist) >= 2 and self.diag > 0:
            t0, p0 = hist[0]
            t1, p1 = hist[-1]
            dt = t1 - t0
            if dt >= self.min_dt:
                dx = p1[0] - p0[0]
                dy = p1[1] - p0[1]
                dist_pix = math.hypot(dx, dy)
                speed_pix_per_sec = dist_pix / dt
                speed_norm = speed_pix_per_sec / self.diag

                # print(f"TRAIN speed_norm={speed_norm:.6f}")  # для дебага

                if not moving:
                    moving = speed_norm > self.on_thr
                else:
                    moving = speed_norm > self.off_thr

        s["moving"] = moving
        return moving

# ====== окна для логов ======

def init_window_stats(window_idx: int):
    start_t = window_idx * WINDOW_SIZE_SEC
    end_t = start_t + WINDOW_SIZE_SEC
    return {
        "window_index": window_idx,
        "start_time_sec": start_t,
        "end_time_sec": end_t,

        "workers_safe_ids": set(),
        "workers_medium_ids": set(),
        "workers_danger_ids": set(),

        "cleaning_workers_ids": set(),

        "dangerous_situations_count": 0,
        "violations_count": 0,

        "train_present": False,
    }


def finalize_and_log_window(ws, prev_train_present: bool):
    if ws is None:
        return prev_train_present

    train_present = ws["train_present"]
    train_arrived = train_present and (not prev_train_present)
    train_departed = (not train_present) and prev_train_present

    payload = {
        "video_id": VIDEO_ID,
        "window_index": ws["window_index"],
        "start_time_sec": float(ws["start_time_sec"]),

        "end_time_sec": float(ws["end_time_sec"]),

        "workers_in_safe":   len(ws["workers_safe_ids"]),
        "workers_in_medium": len(ws["workers_medium_ids"]),
        "workers_in_danger": len(ws["workers_danger_ids"]),

        "cleaning_workers_count": len(ws["cleaning_workers_ids"]),

        "dangerous_situations_count": int(ws["dangerous_situations_count"]),
        "violations_count":           int(ws["violations_count"]),

        "train_present": bool(train_present),
        "train_arrived_in_window":  bool(train_arrived),
        "train_departed_in_window": bool(train_departed),
        "train_number": None,
    }

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print("WINDOW_LOG:", payload)
    return train_present

# ====== красивый bbox ======

def draw_box(frame, x1, y1, x2, y2, color, label):
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    box_y1 = max(y1 - th - 8, 0)
    box_y2 = y1
    box_x2 = x1 + tw + 10
    cv2.rectangle(frame, (x1, box_y1), (box_x2, box_y2), color, -1)
    cv2.putText(frame, label, (x1 + 5, box_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)

# ====== MAIN ======

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: cannot open video:", VIDEO_PATH)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(START_TIME_SEC * fps)
    start_frame = min(max(start_frame, 0), max(total_frames - 1, 0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print(f"Video FPS = {fps:.2f}, total_frames = {total_frames}, start_frame = {start_frame}")
    print("video_id =", VIDEO_ID)

    detect_model = YOLO(DETECT_MODEL_PATH)
    pose_model   = YOLO(POSE_MODEL_PATH)

    detect_model.to("cuda:0")
    pose_model.to("cuda:0")

    ret, sample = cap.read()
    if not ret:
        print("No frames in video")
        return
    h_img, w_img = sample.shape[:2]

    # --- ИНИЦИАЛИЗАЦИЯ ВИДЕОРАЙТЕРА ДЛЯ СОХРАНЕНИЯ АННОТИРОВАННОГО ВИДЕО ---
    os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (w_img, h_img))

    # возвращаемся к стартовому фрейму
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    train_tracker = TrainTracker(fps, w_img, h_img)

    frame_idx = start_frame - 1

    stable_tracks = {}  # track_id -> {"history": [...]}
    current_window_idx = None
    current_window_stats = None
    prev_train_present = False

    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            current_time = frame_idx / fps

            # окна
            window_idx = int(current_time // WINDOW_SIZE_SEC)
            if current_window_idx is None:
                current_window_idx = window_idx
                current_window_stats = init_window_stats(window_idx)
            elif window_idx > current_window_idx:
                prev_train_present = finalize_and_log_window(current_window_stats, prev_train_present)
                current_window_idx = window_idx
                current_window_stats = init_window_stats(window_idx)
            ws = current_window_stats

            results = detect_model.track(
                source=frame,
                imgsz=640,
                conf=0.3,
                iou=0.5,
                tracker="bytetrack.yaml",
                persist=True,
                device=0,
                half=True,
                stream=False,
                verbose=False,
            )
            frame_vis = frame.copy()
            draw_zones(frame_vis)

            poses = extract_poses(frame_vis, pose_model)

            if results:
                r = results[0]
                if r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cls_id = int(box.cls[0])
                        role = class_to_role(cls_id)

                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        zone = get_zone_for_point(cx, cy)

                        track_id = int(box.id[0]) if box.id is not None else None
                        if track_id is None:
                            track_id = hash((frame_idx, x1, y1, x2, y2))

                        # поезд
                        if role == "train":
                            bottom_cx = (x1 + x2) / 2.0
                            bottom_cy = y2
                            moving_bbox = train_tracker.update(track_id, current_time, bottom_cx, bottom_cy)
                            moving = moving_bbox or is_train_moving_time(current_time)
                            ws["train_present"] = True

                            train_color = (60, 60, 255) if moving else (80, 220, 120)
                            draw_box(frame_vis, x1, y1, x2, y2,
                                     train_color,
                                     "TRAIN MOVING" if moving else "TRAIN STOPPED")
                            continue  # поезд в логи по людям не идёт

                        # люди и noname
                        if not (role.startswith("person_") or role == "noname"):
                            continue

                        # трек по людям (для action)
                        tr = stable_tracks.get(track_id)
                        if tr is None:
                            tr = {"history": []}
                            stable_tracks[track_id] = tr

                        pose_info = match_pose_to_box(x1, y1, x2, y2, poses)
                        posture = pose_info["posture"] if pose_info is not None else "unknown"

                        tr["history"].append({
                            "frame": frame_idx,
                            "time_sec": current_time,
                            "cx": cx,
                            "cy": cy,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "posture": posture,
                        })
                        max_hist = int(fps * 2)
                        if len(tr["history"]) > max_hist:
                            tr["history"] = tr["history"][-max_hist:]

                        update_motion_state(tr, fps)
                        action = classify_action(tr)

                        # --- окно: распределение людей по зонам ---
                        if zone in SAFE_ZONES:
                            ws["workers_safe_ids"].add(track_id)
                        elif zone in MEDIUM_ZONES:
                            ws["workers_medium_ids"].add(track_id)
                        elif zone in DANGER_ZONES:
                            ws["workers_danger_ids"].add(track_id)

                        # уборщики, реально работающие
                        if role == "person_orange" and action in ("walk", "squat"):
                            ws["cleaning_workers_ids"].add(track_id)

                        # --- опасные ситуации / нарушения ---
                        danger = False
                        violation = False
                        danger_type = None  # 'possible' / 'critical'

                        allowed_zones = ROLE_RULES.get(role, ROLE_RULES["unknown"])["allowed_zones"]

                        # нарушение: зона запрещена для роли
                        if zone != "none" and zone not in allowed_zones:
                            violation = True
                            danger = True
                            danger_type = "critical"

                        # правило: оранж. хит — тем, кому red_danger разрешена, и они не белые
                        if (zone in DANGER_ZONES and
                                zone in allowed_zones and
                                role != "person_white"):
                            danger = True
                            if not violation:
                                danger_type = "possible"

                        if danger:
                            ws["dangerous_situations_count"] += 1
                        if violation:
                            ws["violations_count"] += 1

                        # --- ЦВЕТА ДЛЯ ЛЮДЕЙ ВКЛЮЧАЯ ОСОБЫЕ ПРАВИЛА ДЛЯ КРАСНОЙ ЗОНЫ ---
                        if zone in DANGER_ZONES:
                            # в красной зоне:
                            # - person_white — зелёный
                            # - уборщик (person_orange), person_red и noname — красные
                            # - остальные (blue, gray и т.п.) — оранжевые
                            if role == "person_white":
                                color = (80, 220, 120)    # зелёный
                            elif role in ("person_orange", "person_red", "noname"):
                                color = (0, 0, 255)        # красный
                            else:
                                color = (0, 165, 255)      # оранжевый
                        else:
                            # остальные зоны — как раньше: violation/possible/ok
                            if violation:
                                color = (0, 0, 255)          # красный — жёсткое нарушение
                            elif danger_type == "possible":
                                color = (0, 165, 255)        # оранжевый — разрешён, но опасная зона
                            else:
                                color = (80, 220, 120)       # зелёный

                        label = f"{role.replace('person_', '')} | {zone} | {action}"
                        if danger_type == "possible":
                            label += " | POSSIBLE"
                        if violation:
                            label += " | VIOL"

                        draw_box(frame_vis, x1, y1, x2, y2, color, label)

            # показываем кадр
            cv2.imshow("Tracking + Logging", frame_vis)

            # и одновременно пишем в видеофайл
            if out_writer is not None:
                out_writer.write(frame_vis)

        # управление клавишами
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord('u'):
            paused = not paused
            print("PAUSE" if paused else "RESUME")

    # финализируем последнее окно
    finalize_and_log_window(current_window_stats, prev_train_present)

    cap.release()
    if out_writer is not None:
        out_writer.release()
    cv2.destroyAllWindows()
    print("DONE. Logs ->", LOG_PATH)
    print("Annotated video saved to:", OUTPUT_VIDEO_PATH)


if __name__ == "__main__":
    main()
