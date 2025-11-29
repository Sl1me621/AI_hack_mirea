from ultralytics import YOLO  

def main():
    model = YOLO("yolo11x.pt")

    results = model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        project="runs",
        name="yolo11_custom3",
        cache=True,
        seed=42
    )

    metrics = model.val(
        data="data.yaml",
        device=0
    )
    print("Validation metrics:", metrics)


if __name__ == "__main__":
    main()