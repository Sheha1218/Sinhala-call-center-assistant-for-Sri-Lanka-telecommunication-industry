"""
Google Cloud Text-to-Speech using service account JSON.
Uses sinhala-call-center-agent-8031ac1e97ce.json for Sinhala TTS.
"""
import os
from pathlib import Path

# Set credentials before importing google client
TTS_CREDENTIALS_PATH = os.getenv(
    "GOOGLE_TTS_CREDENTIALS",
    str(Path(__file__).parent.parent / "sinhala-call-center-agent-8031ac1e97ce.json")
)
if os.path.exists(TTS_CREDENTIALS_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = TTS_CREDENTIALS_PATH


def text_to_speech(text: str, language_code: str = "si-LK") -> bytes:
    """
    Convert Sinhala text to speech using Google Cloud TTS.
    Returns MP3 audio bytes.
    Uses sinhala-call-center-agent-8031ac1e97ce.json for auth.
    """
    try:
        from google.cloud import texttospeech
    except ImportError:
        raise RuntimeError("Install: pip install google-cloud-texttospeech")

    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=0.95,
    )
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config,
    )
    return response.audio_content
