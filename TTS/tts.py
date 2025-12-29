import os
import io
import warnings
import pygame
from google.cloud import texttospeech_v1 as texttospeech

class TTS:

    warnings.filterwarnings("ignore", category=UserWarning)


    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    r"D:\Way to Denmark\Projects\Sinhala-call-center-assistant-for-Sri-Lanka-telecommunication-industry\sinhala-call-center-agent.json"
    )


    client = texttospeech.TextToSpeechClient()


    text = "Sim change කිරීමට original customer තමන්ගේ identification document—NIC, Driving License හෝ Passport—ගෙන branch එකට පැමිණිය යුතුය. මෙම සේවාව සඳහා Rs.100 ක් service charge එකක් අය වේ. Request එක process වීමෙන් පසු SIM activation සම්පූර්ණ වීමට සාමාන්‍යයෙන් පැය 3ක් පමණ කාලයක් ගත වන අතර, activation අවසන් වූ පසු ඔබට SIM card එක phone එකට insert කර භාවිතා කළ හැක."


    synthesis_input = texttospeech.SynthesisInput(text=text)


    voice = texttospeech.VoiceSelectionParams(
        language_code="si-LK",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )


    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )


    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )


    audio_bytes = io.BytesIO(response.audio_content)

    pygame.mixer.init()
    pygame.mixer.music.load(audio_bytes, "mp3")
    pygame.mixer.music.play()


    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    print("Sinhala TTS played from memory (no file saved)")
