
from ultralytics import YOLO

# Load pretrained YOLO segmentation model
model = YOLO("yolo26n-seg.pt")

# Input image
results = model("https://ultralytics.com/images/bus.jpg")

# Get the first result
result = results[0]

# Display segmentation results
result.show()

# Print detected objects
print("\nDetected objects:")

if result.boxes is not None:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        print(
            "Object:", model.names[class_id],
            "| Confidence:", round(confidence, 3)
        )

# Check segmentation masks
if result.masks is not None:
    print("\nSegmentation masks:", len(result.masks))
else:
    print("\nNo segmentation masks detected.")
