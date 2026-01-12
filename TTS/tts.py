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


    text = "Sim change එකක් කරන්නෙ කෙසේද? එයට branch එකට පැමිණෙන විට, branch එකට පැමිණෙන customer තමන්ගේ valid NIC හෝ Driving License එකක් තිබිය යුතුය. Passport එකක් භාවිතා කරන විට, අලුත්ම මාස 3ක් ඇතුළත billing proof එකක් (Light bill, Water bill වැනි) ඉදිරිපත් කළ යුතුය. Reload කරන SIM එකක් ගෙන එමත් අනිවාර්යයි. Priority customer නොවන පුද්ගලයන්ට Rs.100 ක් reload fee ගෙවිය යුතුය. Priority customer නම"


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
