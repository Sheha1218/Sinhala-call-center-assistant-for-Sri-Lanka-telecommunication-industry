import os
import io
import warnings
import pygame
import requests
import time
from dotenv import load_dotenv
from google.cloud import texttospeech_v1 as texttospeech

load_dotenv()

class start:
    def __init__(self, connection_number=None, customer_nic=None, customer_name=None):
        self.connection_number = connection_number or "0730000000"
        self.customer_nic = customer_nic or "000000000V"
        self.customer_name = customer_name or "Guest"
        self.stt_api = os.getenv("STT_API")
        self.captured_transcription = None
        self.run()
    
    def run(self):

        warnings.filterwarnings("ignore", category=UserWarning)


        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
        r"D:\Way to Denmark\Projects\Sinhala-call-center-assistant-for-Sri-Lanka-telecommunication-industry\sinhala-call-center-agent.json"
        )


        client = texttospeech.TextToSpeechClient()


        text="ආයුබෝවන් ABC ආයතනයේ මම වෙනෝරා කෙසෙද මා ඔබට සහයවෙන්නේ"


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
            time.sleep(0.1)
        
        # Capture transcription after first greeting
        if self.stt_api:
            try:
                print("Recording customer response to greeting...")
                payload = {
                    "duration": 45,  # 45 seconds to respond
                    "connection_number": self.connection_number,
                    "customer_nic": self.customer_nic,
                    "customer_name": self.customer_name
                }
                stt_response = requests.post(self.stt_api, json=payload)
                stt_result = stt_response.json()
                self.captured_transcription = stt_result.get('transcription', '')
                print(f"Initial Transcription: {self.captured_transcription}")
            except Exception as e:
                print(f"Error capturing initial transcription: {str(e)}")

        print("First greeting workflow completed - waiting for second.py")