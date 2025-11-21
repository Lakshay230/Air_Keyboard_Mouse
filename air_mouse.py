import cv2
import mediapipe as mp
from tts_utils import speak
import util
import pyautogui
from pynput.mouse import Button, Controller
import random

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

# State for anchor / thumb transitions
prev_thumb_closed = False
anchor_offset_x = 0
anchor_offset_y = 0
SCROLL_SPEED = 50
prev_zoom_mode_active = False
zoom_anchor_y = 0
ZOOM_SENSITIVITY = 10


# ---------- Helpers ----------

def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        return hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
    return None


def get_finger_angles(landmarks_list):
    """Helper to get angles for all 4 fingers (index, middle, ring, pinky)."""
    angles = {}
    
    # We need 21 landmarks to do this
    if len(landmarks_list) < 21:
        return None

    # Index Finger
    angles['idx'] = util.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8])
    # Middle Finger
    angles['mid'] = util.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12])
    # Ring Finger
    angles['ring'] = util.get_angle(landmarks_list[13], landmarks_list[14], landmarks_list[16])
    # Pinky Finger
    angles['pinky'] = util.get_angle(landmarks_list[17], landmarks_list[18], landmarks_list[20])
    
    return angles


def is_ok_sign(thumb_index_tip_distance):
    """Checks for 'OK' gesture (thumb tip to index tip)."""
    # Adjust 35 if it's too sensitive or not sensitive enough
    return thumb_index_tip_distance < 35


def is_peace_sign(angles, thumb_index_distance):
    """Peace Sign ✌️: Index/Middle UP, Ring/Pinky DOWN, Thumb OPEN"""
    return (angles['idx'] > 90 and
            angles['mid'] > 90 and
            angles['ring'] < 50 and
            angles['pinky'] < 50 and
            thumb_index_distance > 50) # Thumb is OPEN

def are_scroll_fingers_down(angles):
    """Checks if middle, ring, and pinky are curled down."""
    return (angles['mid'] < 50 and
            angles['ring'] < 50 and
            angles['pinky'] < 50)

def are_scroll_fingers_up(angles):
    """Checks if middle, ring, and pinky are extended up."""
    return (angles['mid'] > 90 and
            angles['ring'] > 90 and
            angles['pinky'] > 90)

def is_left_click(angles, thumb_index_distance):
    # Index down, Middle up, Thumb open
    return (angles['idx'] < 50 and
            angles['mid'] > 90 and
            thumb_index_distance > 50)

def is_right_click(angles, thumb_index_distance):
    # Middle down, Index up, Thumb open
    return (angles['mid'] < 50 and
            angles['idx'] > 90 and
            thumb_index_distance > 50)

def is_double_click(angles, thumb_index_distance):
    # Index down, Middle down, Thumb open
    return (angles['idx'] < 50 and
            angles['mid'] < 50 and
            thumb_index_distance > 50)

# --- CHANGED: Screenshot is now Hang Loose
def is_screenshot(angles, thumb_index_distance):
    """Hang Loose 🤙: Thumb OPEN, Pinky OPEN, others closed"""
    return (thumb_index_distance > 50 and # Thumb is "open"
            angles['idx'] < 50 and
            angles['mid'] < 50 and
            angles['ring'] < 50 and
            angles['pinky'] > 90)

# ---------- Gesture detection with anchor offset ----------
def detect_gestures(frame, landmarks_list, processed):
    # Import all globals
    global prev_thumb_closed, anchor_offset_x, anchor_offset_y
    global prev_zoom_mode_active, zoom_anchor_y

    if len(landmarks_list) >= 21:
        
        # --- 1. Get ALL sensor data first ---
        index_finger_tip = find_finger_tip(processed)
        angles = get_finger_angles(landmarks_list)
        
        if angles is None: return

        # Get the Y-coordinate of the middle finger tip for zoom anchoring
        middle_finger_tip_y = landmarks_list[12][1]

        # Distances
        thumb_index_dist = util.get_distance([landmarks_list[4], landmarks_list[5]]) # To knuckle
        thumb_index_tip_dist = util.get_distance([landmarks_list[4], landmarks_list[8]]) # To tip

        # --- 2. Check for active modes ---
        thumb_closed = thumb_index_dist < 50
        ok_sign_active = is_ok_sign(thumb_index_tip_dist)
        peace_sign_active = is_peace_sign(angles, thumb_index_dist)


        # === STATE MACHINE (Modes have priority) ===

        # --- 1. MOVE MODE (Thumb to index knuckle) ---
        if thumb_closed:
            # (Your existing move code - no changes)
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
                target_x = max(0, min(screen_width - 1, target_x))
                target_y = max(0, min(screen_height - 1, target_y))
                pyautogui.moveTo(target_x, target_y)

        # --- 2. SCROLL MODE ("OK" sign) ---
        elif ok_sign_active:
            # (Your existing scroll code - no changes)
            cv2.putText(frame, "SCROLL MODE", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            if are_scroll_fingers_down(angles):
                speak("Scrolling Down")
                pyautogui.scroll(-SCROLL_SPEED) # Scroll Down
                cv2.putText(frame, "DOWN", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif are_scroll_fingers_up(angles):
                speak("Scrolling Up")
                pyautogui.scroll(SCROLL_SPEED) # Scroll Up
                cv2.putText(frame, "UP", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # --- 3. NEW ZOOM MODE ("Peace" sign ✌️) ---
        elif peace_sign_active:
            cv2.putText(frame, "ZOOM MODE", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2)
            
            # Transition: Zoom mode just activated -> set anchor
            if not prev_zoom_mode_active:
                zoom_anchor_y = middle_finger_tip_y # Anchor to middle finger

            # Action: Calculate zoom amount based on vertical movement
            delta_y = middle_finger_tip_y - zoom_anchor_y
            zoom_steps = int(delta_y * screen_height / ZOOM_SENSITIVITY)
            
            if abs(zoom_steps) > 0:
                # If finger moves UP (delta_y < 0), we want to ZOOM IN (ctrl +)
                # If finger moves DOWN (delta_y > 0), we want to ZOOM OUT (ctrl -)
                if zoom_steps < 0:
                    speak("Zooming In")
                    pyautogui.hotkey('ctrl', '+')
                    cv2.putText(frame, "ZOOM IN", (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    speak("Zooming Out")
                    pyautogui.hotkey('ctrl', '-')
                    cv2.putText(frame, "ZOOM OUT", (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Re-anchor for continuous zoom
                zoom_anchor_y = middle_finger_tip_y

        # --- 4. IDLE / CLICK MODE (No modes active) ---
        else:
            # (These are now conflict-free)
            
            # Check for screenshot (Hang Loose 🤙)
            if is_screenshot(angles, thumb_index_dist):
                 speak("Taking Screenshot")
                 im1 = pyautogui.screenshot()
                 label = random.randint(1, 1000)
                 im1.save(f'my_screenshot_{label}.png')
                 cv2.putText(frame, "Screenshot!", (50, 50),
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            # Clicks
            elif is_left_click(angles, thumb_index_dist):
                speak("Clicking Left")
                mouse.press(Button.left)
                mouse.release(Button.left)
                cv2.putText(frame, "Left Click", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            elif is_right_click(angles, thumb_index_dist):
                speak("Clicking Right")
                mouse.press(Button.right)
                mouse.release(Button.right)
                cv2.putText(frame, "Right Click", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            elif is_double_click(angles, thumb_index_dist):
                speak("Double Click")
                pyautogui.doubleClick()
                cv2.putText(frame, "Double Click", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # --- Update all prev states ---
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