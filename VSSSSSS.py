import cv2
import numpy as np

# ------------------- HUMAN FACE DETECTOR -------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ------------------- LEAF HEALTH DETECTION -------------------
def detect_health(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ---------------- COLOR RANGES ----------------
    lower_green  = np.array([30, 40, 40])
    upper_green  = np.array([90, 255, 255])

    lower_yellow = np.array([15, 40, 40])
    upper_yellow = np.array([35, 255, 255])

    lower_brown  = np.array([5, 60, 20])
    upper_brown  = np.array([20, 255, 200])

    green_mask  = cv2.inRange(hsv, lower_green, upper_green)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    brown_mask  = cv2.inRange(hsv, lower_brown, upper_brown)

    # Clean masks
    kernel = np.ones((5,5), np.uint8)
    yellow_clean = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    brown_clean  = cv2.morphologyEx(brown_mask, cv2.MORPH_OPEN, kernel)

    # Pixel counts
    green_pixels = np.sum(green_mask > 0)
    yellow_pixels = np.sum(yellow_clean > 0)
    brown_pixels = np.sum(brown_clean > 0)
    total_pixels = frame.shape[0] * frame.shape[1]

    detected = green_pixels + yellow_pixels + brown_pixels

    # ---------------- NO PLANT ----------------
    if detected < total_pixels * 0.01:
        return "No Plant Detected"

    # Ratios
    g = green_pixels  / total_pixels
    y = yellow_pixels / total_pixels
    b = brown_pixels  / total_pixels

    # ---------------- HEALTHY ----------------
    if g > 0.55 and y < 0.10 and b < 0.05:
        return "Healthy Leaf"

    # ---------------- YELLOWING ----------------
    if y > 0.15 and b < 0.10:
        return "Yellowing"

    # ---------------- SPOT & BLOTCH DETECTION ----------------
    contours, _ = cv2.findContours(brown_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    spot = 0
    blotch = 0

    for c in contours:
        area = cv2.contourArea(c)
        if 50 < area < 300:
            spot += 1
        elif 300 <= area < 2000:
            blotch += 1

    if spot > 5:
        return "Leaf Spots"

    if blotch > 2:
        return "Leaf Blotch"

    # ---------------- PEST DAMAGE ----------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    holes = cv2.countNonZero(thresh) / total_pixels

    if holes > 0.05:
        return "Pest Attack"

    return "Moderate"


# ---------------- MAIN LOOP ----------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Human detection
    faces = face_cascade.detectMultiScale(gray, 1.2, 6)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        result = "Human Detected"

    else:
        result = detect_health(frame)

    cv2.putText(frame, result, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Plant + Human Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()