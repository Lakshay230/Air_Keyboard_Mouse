"""
Text-to-Speech (TTS) utility for the Air Keyboard Mouse project.
Provides a simple interface to convert text to speech using gTTS and system audio player.
"""
import os
import sys
import tempfile
import subprocess
from typing import Optional

# Global variable to track if the TTS system is available
tts_available = True

# Determine the platform and appropriate audio player
AUDIO_PLAYER = None
if sys.platform == 'win32':
    # Windows
    try:
        import winsound
        AUDIO_PLAYER = 'winsound'
    except ImportError:
        AUDIO_PLAYER = 'start' if os.system('where start >nul 2>nul') == 0 else None
else:
    # Linux/Unix/MacOS
    for player in ['mpg123', 'mpg321', 'mpv', 'mplayer', 'ffplay', 'play']:
        if os.system(f'which {player} >/dev/null 2>&1') == 0:
            AUDIO_PLAYER = player
            break

try:
    from gtts import gTTS
except ImportError:
    tts_available = False
    print("gTTS is not installed. Please install it with: pip install gTTS")

class TTSUnavailableError(Exception):
    """Exception raised when TTS functionality is not available."""
    pass

def speak(text: str, lang: str = 'en', slow: bool = False) -> None:
    """
    Convert the given text to speech and play it.
    
    Args:
        text (str): The text to be converted to speech
        lang (str, optional): Language code (e.g., 'en' for English, 'es' for Spanish).
        slow (bool, optional): Whether to speak slowly. Defaults to False.
        
    Raises:
        TTSUnavailableError: If TTS functionality is not available
        Exception: For any other errors during TTS conversion or playback
    """
    if not tts_available:
        print("TTS functionality is not available. Please install required packages:")
        print("pip install gTTS playsound")
        return
    
    if not text or not isinstance(text, str):
        print("Invalid text input for TTS")
        return
    
    if not tts_available:
        print("TTS is not available. Please install gTTS: pip install gTTS")
        return
        
    if not AUDIO_PLAYER:
        print("No suitable audio player found. Please install one of: mpg123, mpg321, mpv, mplayer, ffplay, or sox")
        return
    
    try:
        # Create a temporary file to store the speech
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_file = fp.name
        
        try:
            # Convert text to speech
            tts = gTTS(text=text, lang=lang, slow=slow)
            tts.save(temp_file)
            
            # Play the audio file using the system's audio player
            if AUDIO_PLAYER == 'winsound':
                import winsound
                winsound.PlaySound(temp_file, winsound.SND_FILENAME)
            else:
                # Use subprocess to play the audio file
                if AUDIO_PLAYER in ['mpg123', 'mpg321', 'mpv', 'mplayer', 'ffplay', 'play']:
                    cmd = [AUDIO_PLAYER, temp_file]
                    # Make the player run in the background and be quiet
                    if AUDIO_PLAYER in ['mpg123', 'mpg321']:
                        cmd.extend(['-q'])
                    elif AUDIO_PLAYER in ['mpv', 'mplayer']:
                        cmd.extend(['-really-quiet', '-noautosub'])
                    
                    # Start the player and wait for it to finish
                    process = subprocess.Popen(cmd, 
                                            stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL)
                    process.wait()
                
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file)
            except (OSError, PermissionError) as e:
                print(f"Warning: Could not clean up temporary file: {e}")
                
    except Exception as e:
        print(f"Error in TTS: {str(e)}")
        raise

def is_tts_available() -> bool:
    """
    Check if TTS functionality is available.
    
    Returns:
        bool: True if TTS is available, False otherwise
    """
    return tts_available

# Example usage
if __name__ == "__main__":
    if is_tts_available():
        print("Testing TTS functionality...")
        speak("Hello, this is a test of the text-to-speech system.")
    else:
        print("TTS is not available. Please install the required packages:")
        print("pip install gTTS")
        print("And one of these audio players: mpg123, mpg321, mpv, mplayer, ffplay, or sox")
