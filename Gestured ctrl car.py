import cv2
import mediapipe as mp
import serial
import threading

# ---------------- BLUETOOTH SETUP ----------------
COM_PORT = "COM13"        # Your ESP32 COM port
BAUD_RATE = 9600

try:
    bt = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    print(f"CONNECTED to ESP32 on {COM_PORT}")
except Exception as e:
    print(f"Failed to connect to {COM_PORT}: {e}")
    exit()

# ---------------- MEDIAPIPE HANDS ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

last_cmd = ""
gesture_history = []

# ---------------- HELPER FUNCTIONS ----------------
def count_fingers_strict(hand_landmarks):
    fingers = []
    tips = [8, 12, 16, 20]  # index, middle, ring, pinky

    for tip in tips:
        if hand_landmarks[tip].y < hand_landmarks[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    # Thumb
    if hand_landmarks[4].x > hand_landmarks[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    return sum(fingers), fingers

def send_command(cmd):
    try:
        bt.write(cmd.encode())
    except Exception as e:
        print("Bluetooth error:", e)

# ---------------- MAIN LOOP ----------------
print("Starting gesture control. Press ESC to exit.")

while True:

    ret, frame = cap.read()
    if not ret:
        print("Camera error.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    cmd = ""   # reset every loop

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        finger_count, _ = count_fingers_strict(hand.landmark)

        # Smoothing
        gesture_history.append(finger_count)
        if len(gesture_history) > 3:
            gesture_history.pop(0)
        smoothed = round(sum(gesture_history) / len(gesture_history))

        # ---- COMMAND LOGIC ----
        if smoothed >= 5:
            cmd = "F"    # Forward
        elif smoothed == 4:
            cmd = "S"    # Spin
        elif smoothed == 3:
            cmd = "L"    # Left
        elif smoothed == 2:
            cmd = "R"    # Right
        elif smoothed <= 1:
            cmd = "B"    # Backward

    else:
        cmd = "X"   # NO GESTURE → STOP CAR

    # Send only when changed
    if cmd != last_cmd:
        print("Sending:", cmd)
        threading.Thread(target=send_command, args=(cmd,)).start()
        last_cmd = cmd

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
bt.close()
cv2.destroyAllWindows()
