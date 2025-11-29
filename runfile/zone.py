import cv2

IMAGE_PATH = "frame.jpg"  # кадр из видео

current_polygon = []
all_polygons = []
POINTS_PER_POLYGON = 16


def mouse_callback(event, x, y, flags, param):
    global current_polygon
    img = param.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append((x, y))
        print(f"P{len(current_polygon)}: {(x, y)}")

        for p in current_polygon:
            cv2.circle(img, p, 4, (0, 0, 255), -1)

        cv2.imshow("image", img)


def main():
    global current_polygon, all_polygons

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Не удалось открыть", IMAGE_PATH)
        return

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_callback, img)

    print(f"Кликай {POINTS_PER_POLYGON} точек для каждой зоны.")
    print("Когда накликал нужное кол-во точек → нажми 'n' (next zone).")
    print("Когда закончил все зоны → нажми 's' (save) для вывода координат.")
    print("ESC — выход без сохранения.")

    while True:
        cv2.imshow("image", img)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('n'):
            if len(current_polygon) != POINTS_PER_POLYGON:
                print(f"❌ Нужно ровно {POINTS_PER_POLYGON} точек, сейчас {len(current_polygon)}")
            else:
                all_polygons.append(current_polygon)
                print(f"✔️ Зона сохранена. Всего полигонов: {len(all_polygons)}")
                current_polygon = []

        if key == ord('s'):
            if len(current_polygon) == POINTS_PER_POLYGON:
                all_polygons.append(current_polygon)

            print("\n=== Ваши полигоны ===")
            for i, poly in enumerate(all_polygons):
                print(f"Z{i} = {poly}")
            break

        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()