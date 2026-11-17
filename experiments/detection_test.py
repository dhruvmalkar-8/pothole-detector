
from ultralytics import YOLO

# Load a pretrained YOLO detection model
model = YOLO("yolo26n.pt")

# Give it an image
results = model("https://ultralytics.com/images/bus.jpg")

# Display detected objects
results[0].show()

# Print detection information
for box in results[0].boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])

    print("Object:", model.names[class_id])
    print("Confidence:", round(confidence, 3))

# Print all available categories
print("\nAvailable classes:")
print(model.names)
