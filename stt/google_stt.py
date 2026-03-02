"""
Google Cloud Speech-to-Text for Sinhala.
Uses same credentials as TTS (sinhala-call-center-agent-8031ac1e97ce.json).
Expects LINEAR16 PCM, 16kHz, mono.
"""
import os
from pathlib import Path

STT_CREDENTIALS_PATH = os.getenv(
    "GOOGLE_TTS_CREDENTIALS",
    str(Path(__file__).parent.parent / "sinhala-call-center-agent-8031ac1e97ce.json")
)
if os.path.exists(STT_CREDENTIALS_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = STT_CREDENTIALS_PATH


def speech_to_text(audio_bytes: bytes, sample_rate: int = 16000) -> str:
    """
    Convert audio (LINEAR16 PCM, 16kHz mono) to text using Google Cloud Speech-to-Text.
    Returns transcribed text in Sinhala.
    """
    try:
        from google.cloud import speech
    except ImportError:
        raise RuntimeError("Install: pip install google-cloud-speech")

    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=audio_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code="si-LK",
    )
    response = client.recognize(config=config, audio=audio)
    text_parts = []
    for result in response.results:
        if result.alternatives:
            text_parts.append(result.alternatives[0].transcript)
    return " ".join(text_parts).strip()
