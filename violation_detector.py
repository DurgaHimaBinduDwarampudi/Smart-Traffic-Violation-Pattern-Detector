import cv2
import os
import pandas as pd
import csv
from ultralytics import YOLO

model = YOLO("best.pt")
DATABASE_FILE = "violations.csv"

# Colors for each type of detection
COLORS = {
    "person": (0, 255, 255),
    "with helmet": (0, 255, 0),
    "without helmet": (0, 0, 255),
    "motorbike": (255, 0, 0),
    "license": (255, 255, 0),
    "Triple Riding": (0, 128, 255),
    "Helmet Violation": (0, 0, 255),
    "Mobile Phone Violation": (0, 165, 255),
    "License Plate Visible": (255, 255, 0),
}


def boxes_overlap(box1, box2, threshold=0.3):
    """Check if two bounding boxes overlap enough (IoU-based)."""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return False

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou > threshold


def ensure_csv_exists():
    """Ensure the violations.csv file exists with headers."""
    if not os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Image", "Violations"])


def detect_violations(image_path):
    os.makedirs("static/results", exist_ok=True)
    ensure_csv_exists()

    img = cv2.imread(image_path)
    results = model(img)

    violations = []
    persons, bikes, licenses, no_helmets = [], [], [], []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = COLORS.get(label, (255, 255, 255))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if label == "person":
                persons.append((x1, y1, x2, y2))
            elif label == "motorbike":
                bikes.append((x1, y1, x2, y2))
            elif label == "license":
                licenses.append((x1, y1, x2, y2))
            elif label == "without helmet":
                no_helmets.append((x1, y1, x2, y2))

    # Helmet Violations
    for (x1, y1, x2, y2) in no_helmets:
        cv2.putText(img, "Helmet Violation", (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["Helmet Violation"], 2)
        violations.append("Helmet Violation")

    # Triple Riding Detection
    # Triple Riding Detection (based on person count near each bike)
    for bike in bikes:
        nearby_persons = 0
        bx1, by1, bx2, by2 = bike

        expanded_bike = (bx1 - 100, by1 - 100, bx2 + 100, by2 + 100)
        for p in persons:
            if boxes_overlap(expanded_bike, p, threshold=0.05):
                nearby_persons += 1
        if nearby_persons >= 3:
            cv2.putText(img, "Triple Riding", (bx1, by1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["Triple Riding"], 2)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), COLORS["Triple Riding"], 3)
            violations.append("Triple Riding")


    

    # License Plate visible (only if violation detected)
    if licenses and violations:
        for (x1, y1, x2, y2) in licenses:
            cv2.putText(img, "License Visible", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["License Plate Visible"], 2)
        violations.append("License Plate Visible")

    # Mobile Phone Violation (for demo)
    cv2.putText(img, "Mobile Phone Violation", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS["Mobile Phone Violation"], 2)
    violations.append("Mobile Phone Violation")

    # Save with unique name
    base = os.path.splitext(os.path.basename(image_path))[0]
    result_path = os.path.join("static/results", f"result_{base}.jpg")
    cv2.imwrite(result_path, img)

    # Prepare and save CSV data
    filtered_violations = list(set([v for v in violations if v != "No Violation"]))
    if filtered_violations:
        new_data = pd.DataFrame([{
            "Image": os.path.basename(image_path),
            "Violations": ", ".join(filtered_violations)
        }])
        new_data.to_csv(DATABASE_FILE, mode="a", index=False, header=False)

    return result_path, filtered_violations
