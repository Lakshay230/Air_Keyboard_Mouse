import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import os
import math

from tts_utils import speak

# --- Try to import pynput for system-wide keyboard control ---
try:
    from pynput.keyboard import Key, Controller
    keyboard = Controller()
    print("pynput loaded. Script WILL control your system keyboard.")
    print("   Click on Notepad or Word after starting to type!")
except ImportError:
    print("'pynput' module not found. (Run: pip install pynput)")
    print("   Script will only show text in the OpenCV window.")
    keyboard = None

# --- Configuration ---
MODEL_PATH = "emnist_byclass_cnn.pth"
NUM_CLASSES = 62
CLASS_MAPPING = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} not found. Train model first (run train_model_byclass.py).")

# NEW: --- Create a directory to save screenshots ---
os.makedirs("screenshots", exist_ok=True)
screenshot_counter = 0
print("Screenshots will be saved to the 'screenshots' folder.")

# PyTorch model definition (must match training) ------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("PyTorch 'byclass' model loaded:", MODEL_PATH)

# ------------------ MediaPipe hands ------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ------------------ Video capture & canvas ------------------
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    speak('cannot open camera')
    raise RuntimeError("Cannot open webcam")
h_frame, w_frame = frame.shape[:2]
canvas = np.zeros((h_frame, w_frame), dtype=np.uint8) 

# ------------------ Drawing control & state ------------------
prev_point = None
text_output = ""
action_locked = False # State to prevent repeated actions

# Parameters
LINE_THICKNESS = 14
MIN_AREA = 500 # Min area for a letter to be detected

# ----------------- Gesture Helper Functions -----------------
def check_gesture(lm_list):
    """Checks landmarks and returns a string for the detected gesture."""
    
    # 1. Calculate distances
    # Distance for "OK" (Thumb tip [4] to Index tip [8])
    tx, ty = lm_list[4].x, lm_list[4].y
    ix, iy = lm_list[8].x, lm_list[8].y
    dist_ok = math.hypot(tx - ix, ty - iy)
    
    # Distance for "DRAW" (Thumb tip [4] to Index base [5])
    ibx, iby = lm_list[5].x, lm_list[5].y
    dist_draw = math.hypot(tx - ibx, ty - iby)

    # 2. Check "OK" first (most specific)
    if dist_ok < 0.07:
        return "ok"
        
    # 3. Get individual finger states
    try:
        thumb_up = lm_list[4].y < lm_list[3].y
        index_up = lm_list[8].y < lm_list[6].y
        middle_up = lm_list[12].y < lm_list[10].y
        ring_up = lm_list[16].y < lm_list[14].y
        pinky_up = lm_list[20].y < lm_list[18].y
    except Exception as e:
        return "none"

    # 4. Check for Fist (All 4 fingers DOWN, ignore thumb)
    if not index_up and not middle_up and not ring_up and not pinky_up:
        return "fist"
        
    # 5. Check for Peace (Index/Middle UP, Ring/Pinky DOWN, ignore thumb)
    if index_up and middle_up and not ring_up and not pinky_up:
        return "peace"
        
    # 6. Check for Open Hand (All 5 fingers UP)
    if thumb_up and index_up and middle_up and ring_up and pinky_up:
        return "pause" # This is our Backspace
        
    # 7. Check for Draw (Thumb to base) - This is checked LAST
    if dist_draw < 0.05:
        return "draw"
        
    return "none" # No clear gesture

# ----------------- Model & Drawing Helper Functions -----------------

def preprocess_crop_for_model(crop_gray):
    """Prepares the drawing for the model."""
    global screenshot_counter # NEW: Access the global counter
    if crop_gray.size == 0:
        return None
    h, w = crop_gray.shape
    size = max(h, w)
    sq = np.zeros((size, size), dtype=np.uint8)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    sq[y_off:y_off+h, x_off:x_off+w] = crop_gray

    img28 = cv2.resize(sq, (28, 28), interpolation=cv2.INTER_AREA)

    # Apply the orientation fix to match the EMNIST dataset
    img28 = cv2.rotate(img28, cv2.ROTATE_90_CLOCKWISE)
    img28 = cv2.flip(img28, 1)

    # cv2.imshow("2. Preprocessed (28x28)", img28) # <-- REMOVED THIS LINE
    # NEW: Save the preprocessed screenshot
    cv2.imwrite(f"screenshots/preprocessed_{screenshot_counter}.png", img28)

    # Normalize to [-1, 1]
    img28 = img28.astype(np.float32) / 255.0
    img28 = (img28 - 0.5) / 0.5
    tensor = torch.from_numpy(img28).unsqueeze(0).unsqueeze(0).float().to(device)
    return tensor

def find_and_merge_boxes(canvas_img, min_area=MIN_AREA):
    """Finds the single bounding box for the drawn letter."""
    blur = cv2.GaussianBlur(canvas_img, (3,3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None

    min_x, min_y = w_frame, h_frame
    max_x, max_y = 0, 0
    total_area = 0
    
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 100: continue
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)
        total_area += cv2.contourArea(c)

    if total_area < min_area:
        return None
        
    return (min_x, min_y, max_x, max_y)


def predict_box_letter(box, canvas_img):
    """Crops the canvas to the box and predicts the letter."""
    global screenshot_counter # NEW: Access the global counter
    x1,y1,x2,y2 = box
    pad = 8
    x1p = max(0, x1-pad); y1p = max(0, y1-pad)
    x2p = min(canvas_img.shape[1], x2+pad); y2p = min(canvas_img.shape[0], y2+pad)
    crop = canvas_img[y1p:y2p, x1p:x2p]

    # cv2.imshow("1. Raw Crop", crop) # <-- REMOVED THIS LINE
    # NEW: Save the raw crop screenshot
    cv2.imwrite(f"screenshots/raw_crop_{screenshot_counter}.png", crop)

    tensor = preprocess_crop_for_model(crop)
    if tensor is None:
        return None
        
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)
        prob_val, idx_tensor = torch.max(probs, 1)
        idx = int(idx_tensor.item())
        
        if prob_val.item() > 0.6: 
            if idx < len(CLASS_MAPPING):
                return CLASS_MAPPING[idx]
            else:
                return '?'
        else:
            return '?'

def process_and_commit_letter():
    """
    Finds the drawn letter, predicts it, appends it to our text_output,
    and types it using the system keyboard.
    """
    global canvas, text_output, screenshot_counter # NEW: Access the global counter
    
    box = find_and_merge_boxes(canvas)
    letter_to_type = None
    
    if box:
        letter = predict_box_letter(box, canvas)
        if letter and letter != '?':
            text_output += letter
            letter_to_type = letter # Store letter to be typed
            print(f"Added: {letter}")
        else:
            print("Prediction failed (low confidence or no box)")
        
        # NEW: Increment counter *after* processing is done
        screenshot_counter += 1 
    
    # print("Press any key in the pop-up windows to continue...") # <-- REMOVED
    # cv2.waitKey(0) # Pauses until you press a key in one of the windows # <-- REMOVED
    # cv2.destroyAllWindows() # Closes the pop-up windows # <-- REMOVED
    # Always clear canvas after action
    canvas = np.zeros_like(canvas)

    # Type the letter to the system
    if keyboard and letter_to_type:
        speak(f'{letter_to_type}')
        keyboard.type(letter_to_type)


# ----------------- main loop -----------------
print("--- Instructions (System Keyboard) ---")
print("1. Run this script.")
print("2. IMPORTANT: Click on the window you want to type in (e.g., Notepad, Word).")
print("--- Gestures (New) ---")
print(" - Thumb to Index Base: DRAW (and RESET lock)")
print(" - OK Sign: Commit letter (NEXT LETTER)")
print(" - Peace Sign: Commit letter + SPACE (Fires ONCE)")
print(" - Open Hand (5 Fingers): BACKSPACE (Fires ONCE)")
print(" - Fist: CLEAR ALL (Fires ONCE)")
print(" - Keys: q quit")

while True:
    ret, frame = cap.read()
    if not ret: break
    img = cv2.flip(frame, 1) # Flip for mirror view
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)
    h, w = img.shape[:2]

    overlay = img.copy()
    current_gesture = "none"

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0]
        lm_list = lm.landmark
        
        current_gesture = check_gesture(lm_list)
        color = (0,0,255)
        
        # --- 1. Handle Drawing ---
        if current_gesture == "draw":
            color = (0,255,0) # Green for drawing
            
            # We still draw with the index tip (landmark 8)
            ix = int(lm_list[8].x * w); iy = int(lm_list[8].y * h) 
            
            if prev_point is None:
                prev_point = (ix, iy)
            cv2.line(canvas, prev_point, (ix, iy), 255, LINE_THICKNESS)
            prev_point = (ix, iy)
            action_locked = False # <-- RESET the lock on draw
        else:
            prev_point = None
            # We DON'T update lock or gesture here

        # We still show the circle on the index tip
        cv2.circle(img, (int(lm_list[8].x * w), int(lm_list[8].y * h)), 10, color, -1)
            
        # --- 2. Handle Actions (NEW: 'action_locked' logic) ---
        if current_gesture != "none" and current_gesture != "draw" and not action_locked:
            
            if current_gesture == "ok":
                print("ACTION: OK (Commit Letter)")
                process_and_commit_letter()
                action_locked = True # <-- Lock after action
            
            elif current_gesture == "peace":
                print("ACTION: Peace (Commit + Space)")
                process_and_commit_letter() # Types the letter
                text_output += " "
                if keyboard:
                    keyboard.type(' ') # Then types the space
                action_locked = True # <-- Lock after action
                
            elif current_gesture == "pause": # 'pause' is Open Hand (NEW: Backspace)
                print("ACTION: Open Hand (Backspace)")
                if len(text_output) > 0:
                    text_output = text_output[:-1]
                    if keyboard:
                        keyboard.press(Key.backspace)
                        keyboard.release(Key.backspace)
                canvas = np.zeros_like(canvas) # Clear canvas on action
                action_locked = True # <-- Lock after action
            
            elif current_gesture == "fist": # (NEW: Clear All)
                print("ACTION: Fist (Clear All)")
                if keyboard and len(text_output) > 0:
                    # Press backspace for every char in text_output
                    for _ in range(len(text_output)):
                        keyboard.press(Key.backspace)
                        keyboard.release(Key.backspace)
                
                # Clear our internal state
                canvas = np.zeros_like(canvas)
                text_output = ""
                action_locked = True # <-- Lock after action
            
    else:
        prev_point = None

    # --- 3. Update Display ---
    mask = canvas > 10
    overlay[mask] = (255,255,255) 

    cv2.putText(overlay, "Text: " + text_output, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    
    # Display current gesture and lock state
    state_text = f"Gesture: {current_gesture.upper()}"
    if action_locked and current_gesture != "draw":
        state_text += " (LOCKED)"
    cv2.putText(overlay, state_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)


    cv2.imshow("Air Keyboard (System Control) (q to quit)", overlay)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
