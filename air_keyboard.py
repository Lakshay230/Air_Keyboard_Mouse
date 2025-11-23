import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import os
import math

# Try to import pynput for system-wide keyboard control
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
# Make sure this matches the name in your train.py!
MODEL_PATH = "emnist_final_model.pth" 
NUM_CLASSES = 62
CLASS_MAPPING = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} not found. Train model first.")

# Create a directory to save screenshots ---
os.makedirs("screenshots", exist_ok=True)
screenshot_counter = 0

# ==============================================================
#  UPDATED MODEL ARCHITECTURE (MUST MATCH TRAIN.PY)
# ==============================================================
class WiderCNN(nn.Module):
    def __init__(self):
        super(WiderCNN, self).__init__()
        
        # Block 1: 1 -> 64
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1) 
        )
        
        # Block 2: 64 -> 128
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2) 
        )
        
        # Block 3: 128 -> 256
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Dropout(0.2)
        )
        
        # Fully Connected
        self.fc = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024), # Flattened size
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, NUM_CLASSES)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1) # Flatten
        out = self.fc(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize the NEW model class
model = WiderCNN().to(device)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"✅ SUCCESS: Loaded {MODEL_PATH}")
except RuntimeError as e:
    print(f"❌ ERROR: Architecture mismatch. Did you train ResNet instead?")
    print(e)
    exit()

model.eval()

# ------------------ MediaPipe hands ------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ------------------ Video capture & canvas ------------------
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Cannot open webcam")
h_frame, w_frame = frame.shape[:2]
canvas = np.zeros((h_frame, w_frame), dtype=np.uint8) 

# ------------------ Drawing control & state ------------------
prev_point = None
text_output = ""
action_locked = False 

# Parameters
LINE_THICKNESS = 14
MIN_AREA = 500 

# ----------------- Gesture Helper Functions -----------------
def check_gesture(lm_list):
    # 1. Calculate distances
    tx, ty = lm_list[4].x, lm_list[4].y
    ix, iy = lm_list[8].x, lm_list[8].y
    dist_ok = math.hypot(tx - ix, ty - iy)
    
    ibx, iby = lm_list[5].x, lm_list[5].y
    dist_draw = math.hypot(tx - ibx, ty - iby)

    # 2. Check "OK" first
    if dist_ok < 0.07: return "ok"
        
    # 3. Get individual finger states
    try:
        thumb_up = lm_list[4].y < lm_list[3].y
        index_up = lm_list[8].y < lm_list[6].y
        middle_up = lm_list[12].y < lm_list[10].y
        ring_up = lm_list[16].y < lm_list[14].y
        pinky_up = lm_list[20].y < lm_list[18].y
    except: return "none"

    # 4. Check for Fist (Clear)
    if not index_up and not middle_up and not ring_up and not pinky_up: return "fist"
    # 5. Check for Peace (Space)
    if index_up and middle_up and not ring_up and not pinky_up: return "peace"
    # 6. Check for Open Hand (Backspace)
    if thumb_up and index_up and middle_up and ring_up and pinky_up: return "pause"
    # 7. Check for Draw
    if dist_draw < 0.05: return "draw"
        
    return "none"

# ----------------- Model & Drawing Helper Functions -----------------
def preprocess_crop_for_model(crop_gray):
    if crop_gray.size == 0: return None
    h, w = crop_gray.shape
    size = max(h, w) + 20 # Add a little padding to the square
    sq = np.zeros((size, size), dtype=np.uint8)
    
    # Center the image
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    sq[y_off:y_off+h, x_off:x_off+w] = crop_gray

    # Resize to 28x28
    img28 = cv2.resize(sq, (28, 28), interpolation=cv2.INTER_AREA)

    # IMPORTANT: EMNIST Transformation (Rotate & Flip)
    img28 = cv2.rotate(img28, cv2.ROTATE_90_CLOCKWISE)
    img28 = cv2.flip(img28, 1)

    # Normalize
    img28 = img28.astype(np.float32) / 255.0
    img28 = (img28 - 0.5) / 0.5
    tensor = torch.from_numpy(img28).unsqueeze(0).unsqueeze(0).float().to(device)
    return tensor

def find_and_merge_boxes(canvas_img, min_area=MIN_AREA):
    blur = cv2.GaussianBlur(canvas_img, (3,3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return None

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

    if total_area < min_area: return None
    return (min_x, min_y, max_x, max_y)

def predict_box_letter(box, canvas_img):
    x1,y1,x2,y2 = box
    pad = 10
    x1p = max(0, x1-pad); y1p = max(0, y1-pad)
    x2p = min(canvas_img.shape[1], x2+pad); y2p = min(canvas_img.shape[0], y2+pad)
    crop = canvas_img[y1p:y2p, x1p:x2p]

    tensor = preprocess_crop_for_model(crop)
    if tensor is None: return None
        
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)
        prob_val, idx_tensor = torch.max(probs, 1)
        idx = int(idx_tensor.item())
        
        # Mapping logic for 95% accuracy (Merge similar letters)
        # If the model predicts 'o' (lowercase), type 'O' (uppercase)
        # If the model predicts 'l' (lowercase L), type 'L' (uppercase)
        predicted_char = CLASS_MAPPING[idx]
        
        # Smart Merge for typing
        merge_dict = {
            'c':'C', 'i':'I', 'j':'J', 'k':'K', 'l':'L', 'm':'M', 
            'o':'O', 'p':'P', 's':'S', 'u':'U', 'v':'V', 'w':'W', 
            'x':'X', 'y':'Y', 'z':'Z'
        }
        if predicted_char in merge_dict:
            predicted_char = merge_dict[predicted_char]

        if prob_val.item() > 0.5: 
            return predicted_char
        else:
            return '?'

def process_and_commit_letter():
    global canvas, text_output
    box = find_and_merge_boxes(canvas)
    
    if box:
        letter = predict_box_letter(box, canvas)
        if letter and letter != '?':
            text_output += letter
            print(f"Added: {letter}")
            if keyboard: keyboard.type(letter)
        else:
            print("Prediction failed/Low confidence")
    
    canvas = np.zeros_like(canvas)

# ----------------- Main Loop -----------------
print("--- Instructions ---")
print("1. DRAW with Thumb touching Index Base")
print("2. OK Sign to TYPE")
print("3. PEACE Sign to SPACE")
print("4. OPEN HAND to BACKSPACE")
print("5. FIST to CLEAR ALL (Optional)")

while True:
    ret, frame = cap.read()
    if not ret: break
    img = cv2.flip(frame, 1) 
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
            color = (0,255,0)
            ix = int(lm_list[8].x * w); iy = int(lm_list[8].y * h) 
            
            if prev_point is None: prev_point = (ix, iy)
            cv2.line(canvas, prev_point, (ix, iy), 255, LINE_THICKNESS)
            prev_point = (ix, iy)
            action_locked = False 
        else:
            prev_point = None

        cv2.circle(img, (int(lm_list[8].x * w), int(lm_list[8].y * h)), 10, color, -1)
            
        # --- 2. Handle Actions ---
        if current_gesture != "none" and current_gesture != "draw" and not action_locked:
            if current_gesture == "ok":
                process_and_commit_letter()
                action_locked = True
            
            elif current_gesture == "peace":
                process_and_commit_letter()
                text_output += " "
                if keyboard: keyboard.type(' ')
                action_locked = True
                
            elif current_gesture == "pause": # Backspace
                if len(text_output) > 0:
                    text_output = text_output[:-1]
                    if keyboard: 
                        keyboard.press(Key.backspace)
                        keyboard.release(Key.backspace)
                canvas = np.zeros_like(canvas) # Clear canvas too
                action_locked = True
            
            elif current_gesture == "fist":
                pass # Optional: Clear all
            
    else:
        prev_point = None

    # --- 3. Update Display ---
    mask = canvas > 10
    overlay[mask] = (255,255,255) 

    cv2.putText(overlay, "Text: " + text_output, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    state_text = f"Gesture: {current_gesture.upper()}"
    if action_locked and current_gesture != "draw": state_text += " (LOCKED)"
    cv2.putText(overlay, state_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("Air Keyboard", overlay)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
