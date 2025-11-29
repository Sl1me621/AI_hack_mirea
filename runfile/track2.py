import cv2
import math
import numpy as np
from ultralytics import YOLO

# ==== НАСТРОЙКИ ====
MODEL_PATH = "weights/yolo11_x/weights/best.pt"   # TODO: путь к твоей модели
VIDEO_PATH = "C:/Users/User/Desktop/remi2.mp4"         # TODO: путь к твоему видео

MAX_MISSED_SECONDS = 4.0      # разрыв для трека (секунд)
MAX_REASSOC_DIST = 100.0      # макс. расстояние (px) для сопоставления по координатам
SCENE_DIFF_THRESHOLD = 40.0   # чувствительность перескока сцены

# Пороги для анализа движений
IDLE_SPEED_THR = 10.0         # пикс/сек: меньше -> считаем, что стоит (допускаем покачивания)
IDLE_TIME_GREEN = 20.0        # секунд простоя в зелёной зоне -> событие
IDLE_TIME_YELLOW = 10.0       # секунд простоя в жёлтой зоне -> событие

# Порог для определения, что поезд ДВИЖЕТСЯ
TRAIN_MOVE_SPEED_THR = 15.0   # пикс/сек по НИЗУ хитбокса


# ==== ЗОНЫ ====
# green_safe расширена до левого края (x=0) — примерно, без повторной разметки
ZONES = {
    "green_safe": [
        (0, 719),   # левый нижний угол
        (513, 718),
        (539, 546),
        (553, 441),
        (567, 341),
        (583, 251),
        (591, 203),
        (584, 196),
        (0, 196),   # левый верхний для зелёной
    ],

    "yellow_warning": [   # зелёная дорожка
        (511, 719), (533, 588), (548, 480), (560, 381),
        (575, 289), (585, 225), (593, 182), (594, 155),
        (616, 150), (621, 175), (627, 211), (639, 265),
        (652, 322), (665, 385), (691, 503), (747, 718),
    ],

    "red_danger": [       # правая опасная зона (рельсы)
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


# правила: в каких зонах можно быть
ROLE_RULES = {
    "noname": {
        "allowed_zones": [],  # нельзя нигде
    },
    "person_blue": {
        "allowed_zones": ["green_safe", "yellow_warning", "red_danger"],
    },
    "person_gray": {
        "allowed_zones": ["green_safe", "yellow_warning", "red_danger"],
    },
    "person_orange": {
        "allowed_zones": ["green_safe", "yellow_warning"],
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
        "allowed_zones": [],  # на всякий случай
    },
}


def draw_zones(frame):
    """Рисует полупрозрачные зоны поверх кадра."""
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
    """Точка внутри многоугольника (ray casting)."""
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


def main():
    # FPS → кадры в секунды
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        fps = 30.0
    max_missed_frames = int(MAX_MISSED_SECONDS * fps)
    print(f"Video FPS = {fps:.2f}, max_missed_frames = {max_missed_frames}")

    model = YOLO(MODEL_PATH)

    results = model.track(
        source=VIDEO_PATH,
        imgsz=640,
        conf=0.22,
        iou=0.5,
        tracker="bytetrack.yaml",
        persist=True,
        device=0,
        stream=True,
        verbose=False,
    )

    # ByteTrack raw id -> наш стабильный id
    raw_to_stable = {}
    stable_tracks = {}      # sid -> info
    next_stable_id = 0

    # состояние по человеку для действий
    person_state = {}       # (segment_id, sid) -> dict
    events = []             # все события

    # состояние по поезду (отслеживаем по низу хитбокса)
    train_state = {}        # (segment_id, sid) -> dict

    segment_id = 0
    prev_gray = None

    for frame_idx, r in enumerate(results):
        if r.orig_img is not None:
            frame = r.orig_img.copy()
        else:
            frame = r.plot()

        draw_zones(frame)

        # ==== поиск перескока сцены ====
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            mean_diff = diff.mean()
            if mean_diff > SCENE_DIFF_THRESHOLD:
                segment_id += 1
                raw_to_stable.clear()
                stable_tracks.clear()
                person_state.clear()
                train_state.clear()
                next_stable_id = 0
                print(
                    f"\n=== SCENE CUT at frame {frame_idx}, "
                    f"mean_diff={mean_diff:.1f}, new segment_id={segment_id} ===\n"
                )
        prev_gray = gray

        used_sids_this_frame = set()
        train_moving_this_frame = False  # флаг: поезд сейчас движется

        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                raw_id = int(box.id[0]) if box.id is not None else None

                # центр тела — по нему анализируем зону
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                # === трекинг id ===
                sid = None
                if raw_id is not None and raw_id in raw_to_stable:
                    candidate_sid = raw_to_stable[raw_id]
                    track = stable_tracks.get(candidate_sid)
                    if track is not None and frame_idx - track["last_frame"] <= max_missed_frames:
                        sid = candidate_sid

                if sid is None:
                    best_sid = None
                    best_dist = 1e9
                    for candidate_sid, track in stable_tracks.items():
                        if frame_idx - track["last_frame"] > max_missed_frames:
                            continue
                        if candidate_sid in used_sids_this_frame:
                            continue
                        dist = math.hypot(cx - track["cx"], cy - track["cy"])
                        if dist < best_dist and dist < MAX_REASSOC_DIST:
                            best_dist = dist
                            best_sid = candidate_sid
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

                # ===== зона + роль =====
                zone = get_zone_for_point(cx, cy)
                role = class_to_role(cls_id)
                allowed_zones = ROLE_RULES.get(role, ROLE_RULES["unknown"])["allowed_zones"]

                # пишем историю (если пригодится)
                stable_tracks[sid]["history"].append({
                    "segment": segment_id,
                    "frame": frame_idx,
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
                })

                # ===== логика действий (idle / moving) =====
                key = (segment_id, sid)
                st = person_state.get(key)
                if st is None:
                    st = {
                        "last_frame": frame_idx,
                        "last_cx": cx,
                        "last_cy": cy,
                        "last_zone": zone,
                        "zone_enter_frame": frame_idx,
                        "idle_start_frame": frame_idx,
                        "last_action": "unknown",
                    }
                    person_state[key] = st
                else:
                    dt_frames = frame_idx - st["last_frame"]
                    dt = dt_frames / fps if fps > 0 else 0.0
                    dist = math.hypot(cx - st["last_cx"], cy - st["last_cy"])
                    speed = dist / dt if dt > 0 else 0.0

                    # смена зоны
                    if zone != st["last_zone"]:
                        st["zone_enter_frame"] = frame_idx
                        st["last_zone"] = zone
                        st["idle_start_frame"] = frame_idx

                    # idle / moving
                    if speed < IDLE_SPEED_THR:
                        if st["idle_start_frame"] is None:
                            st["idle_start_frame"] = frame_idx
                        idle_time = (frame_idx - st["idle_start_frame"]) / fps

                        if zone == "green_safe" and idle_time > IDLE_TIME_GREEN:
                            events.append({
                                "type": "IDLE_GREEN_TOO_LONG",
                                "segment": segment_id,
                                "sid": sid,
                                "role": role,
                                "zone": zone,
                                "duration_sec": idle_time,
                                "frame": frame_idx,
                            })
                            st["last_action"] = "idle_green_long"
                        elif zone == "yellow_warning" and idle_time > IDLE_TIME_YELLOW:
                            events.append({
                                "type": "IDLE_YELLOW_TOO_LONG",
                                "segment": segment_id,
                                "sid": sid,
                                "role": role,
                                "zone": zone,
                                "duration_sec": idle_time,
                                "frame": frame_idx,
                            })
                            st["last_action"] = "idle_yellow_long"
                        else:
                            if st["last_action"] not in ("idle_green_long", "idle_yellow_long"):
                                st["last_action"] = "idle"
                    else:
                        st["idle_start_frame"] = None
                        st["last_action"] = "moving"

                    st["last_frame"] = frame_idx
                    st["last_cx"] = cx
                    st["last_cy"] = cy

                # ===== отслеживание поезда по НИЗУ хитбокса =====
                if role == "train":
                    key_train = (segment_id, sid)
                    ts = train_state.get(key_train)
                    bottom_y = y2
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

                # ===== проверка нарушения правил роли/зоны =====
                violation = False
                violation_reason = None

                if role == "noname":
                    violation = True
                    violation_reason = "FORBIDDEN_ROLE_NONAME"
                    events.append({
                        "type": violation_reason,
                        "segment": segment_id,
                        "sid": sid,
                        "role": role,
                        "zone": zone,
                        "frame": frame_idx,
                    })
                else:
                    if zone != "none" and zone not in allowed_zones:
                        violation = True
                        violation_reason = "ZONE_FORBIDDEN_FOR_ROLE"
                        events.append({
                            "type": violation_reason,
                            "segment": segment_id,
                            "sid": sid,
                            "role": role,
                            "zone": zone,
                            "frame": frame_idx,
                        })

                # дополнительное правило: поезд движется -> в красной зоне нельзя никому, кроме поезда
                if zone == "red_danger" and role != "train" and train_moving_this_frame:
                    violation = True
                    violation_reason = "TRAIN_MOVING_RED_FORBIDDEN"
                    events.append({
                        "type": violation_reason,
                        "segment": segment_id,
                        "sid": sid,
                        "role": role,
                        "zone": zone,
                        "frame": frame_idx,
                    })

                # ===== РИСОВАНИЕ БОКСА =====
                # все хитбоксы зелёные по умолчанию, нарушение = красный
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

                action = person_state[(segment_id, sid)]["last_action"]
                danger_text = ""
                if violation and zone == "red_danger":
                    danger_text = " DANGER!"

                label = f"id={sid} r={role} z={zone}{danger_text}"
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

                # лог в консоль
                print(f"seg={segment_id} fr={frame_idx} sid={sid} cls={cls_id} "
                    f"role={role} zone={zone} viol={violation} "
                    f"reason={violation_reason} action={action} conf={conf:.2f}"
                )

        cv2.imshow("YOLO + Zones + Roles + Train", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

    print("\n=== EVENTS (for DB / dashboard) ===")
    for e in events:
        print(e)


if __name__ == "__main__":
    main()




