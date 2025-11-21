import sys
import subprocess

def speak(text: str):
    """
    Uses PowerShell to speak text in a completely separate process.
    Does NOT wait for the audio to finish. Returns immediately.
    """
    if not text:
        return

    print(f"DEBUG: Requesting speech -> '{text}'") # Debug print

    try:
        if sys.platform == 'win32':
            # Sanitize text to prevent PowerShell errors
            safe_text = text.replace("'", "").replace('"', '')
            
            # Powershell command to speak using .NET System.Speech
            # We use Popen (Process Open) instead of run() so we don't wait!
            cmd = [
                "powershell", 
                "-Command", 
                f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{safe_text}');"
            ]
            
            # checking=False, creationflags to hide window
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            subprocess.Popen(cmd, startupinfo=startupinfo)
            
        else:
            # Linux/Mac fallback (espeak)
            subprocess.Popen(["espeak", text])
            
    except Exception as e:
        print(f"TTS Error: {str(e)}")

if __name__ == "__main__":
    speak("Testing one")
    import time
    time.sleep(1)
    speak("Testing two - fire and forget")
