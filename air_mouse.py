import cv2
import mediapipe as mp
import pyautogui
import util
from pynput.mouse import Button, Controller
import random

# --- NEW IMPORTS FOR THREADING & TIMING ---
import time
import threading
from tts_utils import speak  # Importing our fixed offline TTS

# Screen size & mouse controller
screen_width, screen_height = pyautogui.size()
mouse = Controller()

# MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

# --- GLOBAL STATE VARIABLES ---
prev_thumb_closed = False
anchor_offset_x = 0
anchor_offset_y = 0

# Scroll State
SCROLL_SPEED = 50

# Zoom State
prev_zoom_mode_active = False
zoom_anchor_y = 0
ZOOM_SENSITIVITY = 10

# --- SPEECH CONTROL (Prevents lagging/spamming) ---
last_speech_time = 0
SPEECH_COOLDOWN = 3  # Seconds to wait before speaking again

# ---------- Helpers ----------

def smart_speak(text):
    """
    Checks if enough time has passed since the last speech.
    Runs the speech in a separate thread so the camera doesn't freeze.
    """
    global last_speech_time
    current_time = time.time()

    if current_time - last_speech_time > SPEECH_COOLDOWN:
        last_speech_time = current_time
        # Run audio in a background thread
        threading.Thread(target=speak, args=(text,), daemon=True).start()

def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        return hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
    return None

def get_finger_angles(landmarks_list):
    if len(landmarks_list) < 21:
        return None
    angles = {}
    angles['idx'] = util.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8])
    angles['mid'] = util.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12])
    angles['ring'] = util.get_angle(landmarks_list[13], landmarks_list[14], landmarks_list[16])
    angles['pinky'] = util.get_angle(landmarks_list[17], landmarks_list[18], landmarks_list[20])
    return angles

def is_ok_sign(thumb_index_tip_distance):
    return thumb_index_tip_distance < 35

def is_peace_sign(angles, thumb_index_distance):
    return (angles['idx'] > 90 and angles['mid'] > 90 and
            angles['ring'] < 50 and angles['pinky'] < 50 and
            thumb_index_distance > 50)

def are_scroll_fingers_down(angles):
    return (angles['mid'] < 50 and angles['ring'] < 50 and angles['pinky'] < 50)

def are_scroll_fingers_up(angles):
    return (angles['mid'] > 90 and angles['ring'] > 90 and angles['pinky'] > 90)

def is_left_click(angles, thumb_index_distance):
    return (angles['idx'] < 50 and angles['mid'] > 90 and thumb_index_distance > 50)

def is_right_click(angles, thumb_index_distance):
    return (angles['mid'] < 50 and angles['idx'] > 90 and thumb_index_distance > 50)

def is_double_click(angles, thumb_index_distance):
    return (angles['idx'] < 50 and angles['mid'] < 50 and thumb_index_distance > 50)

def is_screenshot(angles, thumb_index_distance):
    # Hang Loose 🤙
    return (thumb_index_distance > 50 and angles['idx'] < 50 and
            angles['mid'] < 50 and angles['ring'] < 50 and angles['pinky'] > 90)

# ---------- Gesture detection ----------
def detect_gestures(frame, landmarks_list, processed):
    global prev_thumb_closed, anchor_offset_x, anchor_offset_y
    global prev_zoom_mode_active, zoom_anchor_y

    if len(landmarks_list) >= 21:
        index_finger_tip = find_finger_tip(processed)
        angles = get_finger_angles(landmarks_list)
        if angles is None: return

        middle_finger_tip_y = landmarks_list[12][1]
        
        # Distances
        thumb_index_dist = util.get_distance([landmarks_list[4], landmarks_list[5]])
        thumb_index_tip_dist = util.get_distance([landmarks_list[4], landmarks_list[8]])

        # Active Modes
        thumb_closed = thumb_index_dist < 50
        # RELAXED THRESHOLD: Changed from 35 to 60
        ok_sign_active = is_ok_sign(thumb_index_tip_dist) 
        peace_sign_active = is_peace_sign(angles, thumb_index_dist)

        # === DEBUGGING TEXT (Shows strictly what mode is active) ===
        mode_text = "Mode: IDLE"
        if thumb_closed: mode_text = "Mode: MOVE"
        elif ok_sign_active: mode_text = "Mode: SCROLL"
        elif peace_sign_active: mode_text = "Mode: ZOOM"
        
        cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 0), 2)
        # ==========================================================

        # --- 1. MOVE MODE ---
        if thumb_closed:
            if (not prev_thumb_closed) and index_finger_tip is not None:
                finger_x = int(index_finger_tip.x * screen_width)
                finger_y = int(index_finger_tip.y * screen_height)
                cur_mouse_x, cur_mouse_y = pyautogui.position()
                anchor_offset_x = cur_mouse_x - finger_x
                anchor_offset_y = cur_mouse_y - finger_y
            
            if index_finger_tip is not None:
                finger_x = int(index_finger_tip.x * screen_width)
                finger_y = int(index_finger_tip.y * screen_height)
                target_x = int(finger_x + anchor_offset_x)
                target_y = int(finger_y + anchor_offset_y)
                # Smooth move
                pyautogui.moveTo(target_x, target_y)

        # --- 2. SCROLL MODE ---
        elif ok_sign_active:
            cv2.putText(frame, "SCROLL ACTIVE", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            
            if are_scroll_fingers_down(angles):
                smart_speak("Scrolling Down")
                pyautogui.scroll(-SCROLL_SPEED)
                cv2.putText(frame, "Action: DOWN", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif are_scroll_fingers_up(angles):
                smart_speak("Scrolling Up")
                pyautogui.scroll(SCROLL_SPEED)
                cv2.putText(frame, "Action: UP", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Debugging: If we are in scroll mode but fingers aren't recognized
                cv2.putText(frame, "Action: Neutral", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

        # --- 3. ZOOM MODE ---
        elif peace_sign_active:
            cv2.putText(frame, "ZOOM MODE", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2)
            
            if not prev_zoom_mode_active:
                zoom_anchor_y = middle_finger_tip_y

            delta_y = middle_finger_tip_y - zoom_anchor_y
            zoom_steps = int(delta_y * screen_height / ZOOM_SENSITIVITY)
            
            if abs(zoom_steps) > 0:
                if zoom_steps < 0:
                    smart_speak("Zooming In")
                    pyautogui.hotkey('ctrl', '+')
                else:
                    smart_speak("Zooming Out")
                    pyautogui.hotkey('ctrl', '-')
                zoom_anchor_y = middle_finger_tip_y

        # --- 4. CLICKS / SCREENSHOT ---
        else:
            if is_screenshot(angles, thumb_index_dist):
                smart_speak("Taking Screenshot")
                im1 = pyautogui.screenshot()
                label = random.randint(1, 1000)
                im1.save(f'my_screenshot_{label}.png')
                cv2.putText(frame, "Screenshot!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                time.sleep(0.5)

            elif is_left_click(angles, thumb_index_dist):
                smart_speak("Clicking Left")
                mouse.press(Button.left)
                mouse.release(Button.left)
                cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                time.sleep(0.2)

            elif is_right_click(angles, thumb_index_dist):
                smart_speak("Clicking Right")
                mouse.press(Button.right)
                mouse.release(Button.right)
                cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                time.sleep(0.2)

            elif is_double_click(angles, thumb_index_dist):
                smart_speak("Double Click")
                pyautogui.doubleClick()
                cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                time.sleep(0.3)

        prev_thumb_closed = thumb_closed
        prev_zoom_mode_active = peace_sign_active

# ---------- Main ----------
def main():
    cap = cv2.VideoCapture(0)
    draw = mp.solutions.drawing_utils

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)
            landmarks_list = []

            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmarks_list.append((lm.x, lm.y))

                detect_gestures(frame, landmarks_list, processed)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
