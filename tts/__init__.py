try:
    from .google_tts import text_to_speech
except ImportError:
    text_to_speech = None

__all__ = ["text_to_speech"]
