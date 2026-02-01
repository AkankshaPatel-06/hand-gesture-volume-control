import cv2
import mediapipe as mp
import numpy as np
import math
import subprocess

# Camera SetUp
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

# MediaPipe Hand SetUp
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

print("Hand Gesture Volume Control Started (Mac)")
print("Press 'q' to quit")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break
    
    # Convert BGR to RGB for MediaPipe processing
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

            if len(lmList) >= 9:  # Ensure we have enough landmarks
                x1, y1 = lmList[4][1], lmList[4][2]    # Thumb tip
                x2, y2 = lmList[8][1], lmList[8][2]    # Index finger tip

                # Draw circles and line
                cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                
                # Draw midpoint
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(img, (cx, cy), 8, (0, 255, 0), cv2.FILLED)

                # Calculate distance
                length = math.hypot(x2 - x1, y2 - y1)

                # Convert distance to volume percentage (0-100)
                vol = np.interp(length, [30, 200], [0, 100])
                
                # Set Mac system volume using osascript
                try:
                    subprocess.run(['osascript', '-e', f'set volume output volume {int(vol)}'], 
                                   check=True, capture_output=True)
                except Exception as e:
                    print(f"Volume control error: {e}")
                
                # Visual feedback - red when fingers very close
                if length < 30:
                    cv2.circle(img, (cx, cy), 8, (0, 0, 255), cv2.FILLED)

                # Display volume percentage
                cv2.putText(img, f'Volume: {int(vol)}%',
                            (40, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            3)
                
                # Volume bar visualization
                cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
                cv2.rectangle(img, (50, int(np.interp(vol, [0, 100], [400, 150]))), 
                             (85, 400), (0, 255, 0), cv2.FILLED)

            # Draw hand landmarks
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Hand Gesture Volume Control", img)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()