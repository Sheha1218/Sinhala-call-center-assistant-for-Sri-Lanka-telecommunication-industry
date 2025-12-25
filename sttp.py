from fastapi import FastAPI
import os
from google.cloud import texttospeech


app=FastAPI()

@app.route('/',methods=['POST'])
def sst():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"D:\Way to Denmark\Projects\Sinhala-call-center-assistant-for-Sri-Lanka-telecommunication-industry\sinhala-call-center-agent-19df5cd3b8ff.json"
    client = texttospeech.TextToSpeechClient()



    voice =texttospeech.VoiceSelectionParams(
    language_code="si-LK",
    name='si-LK-Standard-A'
)


    audio_config =texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3,
    speaking_rate=1.0,
    pitch=0.0
)
    return sst()
