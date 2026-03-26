import cv2
import mediapipe as mp
import pyttsx3

engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    gesture = "No Hand"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers = 0
            for tip in [8, 12, 16, 20]:
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip-2].y:
                    fingers += 1

            if fingers == 0:
                gesture = "Hello"
            elif fingers == 1:
                gesture = "Yes"
            elif fingers == 2:
                gesture = "No"

    cv2.putText(frame, gesture, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Gesture System", frame)

    key = cv2.waitKey(1)

    if key == ord('g'):
        speak(gesture)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()