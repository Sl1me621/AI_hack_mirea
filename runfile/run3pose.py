# track_pose_zones.py
import cv2
import math
import numpy as np
from ultralytics import YOLO

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
    # FPS → нужен для скорости поезда
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        fps = 30.0

    max_missed_frames = int(MAX_MISSED_SECONDS * fps)
    print(f"Video FPS = {fps:.2f}, max_missed_frames = {max_missed_frames}")

    detect_model = YOLO(DETECT_MODEL_PATH)
    pose_model = YOLO(POSE_MODEL_PATH)

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

    raw_to_stable = {}
    stable_tracks = {}
    next_stable_id = 0

    events = []
    train_state = {}  # (segment_id, sid) -> dict

    segment_id = 0
    prev_gray = None
    frame_idx = -1

    for r in results:
        frame_idx += 1

        if r.orig_img is not None:
            frame = r.orig_img.copy()
        else:
            frame = r.plot()

        draw_zones(frame)

        # ===== scene cut =====
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
        train_moving_this_frame = False
        current_time = frame_idx / fps

        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                raw_id = int(box.id[0]) if box.id is not None else None

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                # === трекинг id ===
                sid = None
                if raw_id is not None and raw_id in raw_to_stable:
                    cand_sid = raw_to_stable[raw_id]
                    track = stable_tracks.get(cand_sid)
                    if track is not None and frame_idx - track["last_frame"] <= max_missed_frames:
                        sid = cand_sid

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
                    stable_tracks[sid]["cx"] = cx
                    stable_tracks[sid]["cy"] = cy
                    stable_tracks[sid]["last_frame"] = frame_idx
                    stable_tracks[sid]["cls"] = cls_id

                used_sids_this_frame.add(sid)
                if raw_id is not None:
                    raw_to_stable[raw_id] = sid

                # зона + роль
                zone = get_zone_for_point(cx, cy)
                role = class_to_role(cls_id)
                allowed_zones = ROLE_RULES.get(role, ROLE_RULES["unknown"])["allowed_zones"]

                # поза
                pose_info = match_pose_to_box(x1, y1, x2, y2, poses)
                posture = pose_info["pose"] if pose_info is not None else "unknown"

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

                # оранжевые в красной: разрешено, но "опасность, непонятные действия"
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

                # ===== визуализация =====
                # все боксы зелёные, нарушение -> красный
                if violation:
                    box_color = (0, 0, 255)
                else:
                    box_color = (0, 255, 0)

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

        cv2.imshow("YOLO + Zones + Roles + Pose", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

    print("\n=== EVENTS (for DB / dashboard) ===")
    for e in events:
        print(e)


if __name__ == "__main__":
    main()
