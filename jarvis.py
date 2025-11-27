import speech_recognition as sr
import os
import sys
import subprocess
import time

# --- CONFIGURATION ---
MOUSE_SCRIPT = "air_mouse_2.py"
KEYBOARD_SCRIPT = "key.py"

# Greetings
START_GREETING = "Jarvis Online. Hello Lakshay, I am listening."
END_GREETING = "Shutting down system. Goodbye, sir."

def speak(text):
    """
    Uses VBScript (Native Windows Automation) to speak.
    This works even if Python or PowerShell are restricted.
    """
    print(f"JARVIS: {text}")
    
    # Escape quotes for VBScript
    safe_text = text.replace('"', '')
    
    # Create a temporary VBScript file
    vbs_file = "temp_speech.vbs"
    
    # VBScript Code:
    # 1. Create SAPI Object
    # 2. Try to set Voice 1 (Female)
    # 3. Speak
    vbs_code = f"""
    Set Sapi = Wscript.CreateObject("SAPI.SpVoice")
    On Error Resume Next
    Set Sapi.Voice = Sapi.GetVoices.Item(1) ' Try to set Female Voice
    On Error GoTo 0
    Sapi.Rate = 1
    Sapi.Volume = 100
    Sapi.Speak "{safe_text}"
    """
    
    try:
        # Write the VBS file
        with open(vbs_file, "w") as f:
            f.write(vbs_code)
            
        # Run it using Windows Script Host (cscript)
        # //Nologo prevents the startup banner
        subprocess.run(["cscript", "//Nologo", vbs_file], check=True)
        
    except Exception as e:
        print(f"Speech Error: {e}")
        
    finally:
        # Clean up the file
        if os.path.exists(vbs_file):
            os.remove(vbs_file)

def listen_command():
    """Continuously listens for audio."""
    r = sr.Recognizer()
    r.dynamic_energy_threshold = True
    r.energy_threshold = 4000 
    
    with sr.Microphone() as source:
        print("\nListening...")
        r.pause_threshold = 1.0
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=5)
        except sr.WaitTimeoutError:
            return "none"

    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language="en-in")
        print(f"USER: {query}")
        return query.lower()
    except Exception:
        return "none"

def kill_process(process):
    """Force kills a process."""
    if process:
        try:
            process.terminate()
            process.wait(timeout=1)
        except:
            process.kill()

# --- MAIN PROGRAM ---
if __name__ == "__main__":
    # 1. CALIBRATION
    r_init = sr.Recognizer()
    with sr.Microphone() as source:
        print("Calibrating background noise... (Please wait)")
        r_init.adjust_for_ambient_noise(source, duration=1)
    
    # 2. GREETING
    speak(START_GREETING)

    current_process = None
    current_tool_name = "None"

    # 3. MAIN LOOP
    while True:
        query = listen_command()

        if query == "none":
            continue

        # --- OPEN MOUSE ---
        if "mouse" in query and ("open" in query or "start" in query):
            speak("Opening Virtual Mouse")
            
            if current_process:
                kill_process(current_process)
            
            if os.path.exists(MOUSE_SCRIPT):
                current_process = subprocess.Popen([sys.executable, MOUSE_SCRIPT])
                current_tool_name = "Virtual Mouse"
            else:
                speak("Mouse script not found.")

        # --- OPEN KEYBOARD ---
        elif "keyboard" in query and ("open" in query or "start" in query):
            speak("Opening Virtual Keyboard")

            if current_process:
                kill_process(current_process)

            if os.path.exists(KEYBOARD_SCRIPT):
                current_process = subprocess.Popen([sys.executable, KEYBOARD_SCRIPT])
                current_tool_name = "Virtual Keyboard"
            else:
                speak("Keyboard script not found.")

        # --- CLOSE TOOL ---
        elif "close" in query or "stop" in query:
            if current_process:
                speak(f"Closing {current_tool_name}")
                kill_process(current_process)
                current_process = None
                current_tool_name = "None"
            else:
                speak("Nothing is running.")

        # --- EXIT SYSTEM ---
        elif "exit" in query or "bye" in query or "terminate" in query or "sleep" in query:
            if current_process:
                kill_process(current_process)
            
            speak(END_GREETING)

            sys.exit()
