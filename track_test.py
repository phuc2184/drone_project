from ultralytics import YOLO

# Nguon hinh: 0 = webcam, hoac thay bang duong dan video, vi du "video.mp4"
SOURCE = 0

model = YOLO("yolov8n.pt")  # ban nho nhat, hop voi GTX 1650

results = model.track(
    source=SOURCE,
    classes=[0],            # chi lay class "person"
    tracker="bytetrack.yaml",
    persist=True,
    stream=True,
    show=True,              # hien cua so co khung va ID
    verbose=False,
)

for r in results:
    if r.boxes.id is not None:
        print("Cac ID dang thay:", r.boxes.id.int().tolist())
