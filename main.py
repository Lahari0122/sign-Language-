import cv2
import mediapipe as mp
import pyttsx3
import speech_recognition as sr
import joblib
import time
import os
import threading

# ---------------- LOAD MODEL ----------------
if not os.path.exists("model.pkl"):
    print("❌ model.pkl not found. Run train_model.py first.")
    exit()

model = joblib.load("model.pkl")

# ---------------- TEXT TO SPEECH ----------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

def speak(text):
    def run():
        print("🔊 Speaking:", text)
        engine.say(text)
        engine.runAndWait()

    threading.Thread(target=run).start()

# ---------------- SPEECH TO TEXT ----------------
recognizer = sr.Recognizer()

def listen():
    try:
        with sr.Microphone() as source:
            print("🎤 Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

            text = recognizer.recognize_google(audio)
            return text.lower()
    except:
        return "unknown"

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# ---------------- LOAD GESTURE IMAGES ----------------
gesture_images = {}

if not os.path.exists("gestures"):
    print("❌ gestures folder missing")
    exit()

for file in os.listdir("gestures"):
    name = file.split(".")[0].lower()
    img = cv2.imread(os.path.join("gestures", file))
    if img is not None:
        gesture_images[name] = img

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

gesture = "No Hand"
voice_gesture = ""
spoken_text = ""

last_spoken = ""
last_time = 0
predictions = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    detected_gestures = []

    # ---------------- MULTI-HAND GESTURE ----------------
    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            prediction = model.predict([landmarks])[0]
            detected_gestures.append(prediction)

        # Remove duplicates
        detected_gestures = sorted(list(set(detected_gestures)))

        # ---------------- SPEAK ----------------
        gesture = detected_gestures[0]
        current_time = time.time()

        if (gesture != last_spoken) or (current_time - last_time > 3):

            if gesture != "No Hand":
                speak(gesture)
                last_spoken = gesture
                last_time = current_time

    else:
        gesture = "No Hand"
        last_spoken = ""
        predictions.clear()

    # ---------------- VOICE INPUT ----------------
    key_press = cv2.waitKey(10)

    if key_press == ord('v'):
        spoken_text = listen()

        if any(word in spoken_text for word in ["hello", "hi"]):
            voice_gesture = "hello"
        elif "yes" in spoken_text:
            voice_gesture = "yes"
        elif "no" in spoken_text:
            voice_gesture = "no"
        elif any(word in spoken_text for word in ["thank", "thanks"]):
            voice_gesture = "thankyou"
        elif "good" in spoken_text:
            voice_gesture = "good"
        elif "love" in spoken_text:
            voice_gesture = "love"
        elif "bye" in spoken_text:
            voice_gesture = "bye"
        else:
            voice_gesture = "unknown"

    elif key_press == ord('q'):
        break

    # ---------------- OVERLAY ----------------
    key = voice_gesture.lower()
    if key in gesture_images:
        overlay = cv2.resize(gesture_images[key], (150, 150))
        frame[10:160, w-160:w-10] = overlay

    # ---------------- DISPLAY ----------------
    cv2.putText(frame, "Hand: " + gesture, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(frame, "Voice: " + spoken_text, (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.putText(frame, "Voice->Gesture: " + voice_gesture, (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)

    cv2.imshow("System", frame)

cap.release()
cv2.destroyAllWindows()