import os
import io
import warnings
import pygame
import requests
import time
import tempfile
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from dotenv import load_dotenv
from google.cloud import texttospeech_v1 as texttospeech
from openai import OpenAI

# Direct imports — no HTTP round-trips for extraction / verification
from filters.name_nic import (
    extract_name_from_text,
    extract_nic_from_text,
    extract_connection_number_from_text,
)
from db.verification import verification

load_dotenv()

SAMPLE_RATE = 16000
CHANNELS = 1
WORKFLOW_API = os.getenv("WORKFLOW_API")


class second:
    def __init__(self, connection_number=None, customer_nic=None, customer_name=None):
        self.connection_number = connection_number or "0000000000"
        self.customer_nic = customer_nic or "000000000V"
        self.customer_name = customer_name or "Guest"
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Results populated by run()
        self.captured_transcription = None
        self.extracted_name = None
        self.extracted_nic = None
        self.extracted_connection_number = None
        self.verification_result = None
        self.llm_response = None

        self.run()

    # ── mic helpers ──────────────────────────────────────────────
    def record_audio(self, duration: int):
        """Record audio from the local microphone."""
        print(f"Recording for {duration} seconds...")
        audio = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
        )
        sd.wait()
        print("Recording complete")
        return audio

    def transcribe_audio(self, audio_data):
        """Send recorded WAV to OpenAI Whisper and return the text."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav.write(tmp.name, SAMPLE_RATE, audio_data)
            tmp_path = tmp.name

        try:
            with open(tmp_path, "rb") as f:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language="si",
                )
            return transcript.text
        finally:
            os.remove(tmp_path)

    # ── main flow ────────────────────────────────────────────────
    def run(self):
        warnings.filterwarnings("ignore", category=UserWarning)

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
            r"D:\Way to Denmark\Projects\Sinhala-call-center-assistant-for-Sri-Lanka-telecommunication-industry\sinhala-call-center-agent.json"
        )

        # ── 1. Play second prompt via Google TTS ─────────────────
        client = texttospeech.TextToSpeechClient()

        text = "මට ඔබේ නම සහ හැදුරුම්පත් අංකය සහ සබදතා අංකය පැවසිය හැකිද"

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="si-LK",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )

        audio_bytes = io.BytesIO(response.audio_content)

        pygame.mixer.init()
        pygame.mixer.music.load(audio_bytes, "mp3")
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        # ── 2. Record user response (30 s) & transcribe via Whisper ──
        try:
            print("Recording name, NIC number, and connection number...")
            audio_data = self.record_audio(duration=30)
            self.captured_transcription = self.transcribe_audio(audio_data)
            print(f"Transcription: {self.captured_transcription}")
        except Exception as e:
            print(f"Error capturing transcription: {str(e)}")
            self.captured_transcription = ""
            return

        if not self.captured_transcription:
            print("No transcription captured")
            return

        # ── 3. Regex extraction of name / NIC / connection number ──
        print("\n=== Extracting customer details (regex) ===")
        self.extracted_name = extract_name_from_text(self.captured_transcription)
        self.extracted_nic = extract_nic_from_text(self.captured_transcription)
        self.extracted_connection_number = extract_connection_number_from_text(
            self.captured_transcription
        )

        # Fallback: use the value passed in if regex didn't find one
        if not self.extracted_connection_number:
            self.extracted_connection_number = self.connection_number

        print(f"  Name           : {self.extracted_name}")
        print(f"  NIC            : {self.extracted_nic}")
        print(f"  Connection No. : {self.extracted_connection_number}")

        # ── 4. Verify extracted values against database ──────────
        print("\n=== Verifying customer against database ===")
        verifier = None
        try:
            verifier = verification()
            is_valid, customer_data = verifier.verify_customer(
                self.extracted_connection_number,
                self.extracted_nic,
                self.extracted_name,
            )

            self.verification_result = {
                "is_valid": is_valid,
                "customer_data": str(customer_data) if customer_data else None,
            }

            if is_valid:
                print(f"Customer verified: {customer_data}")
            else:
                print("Customer verification failed")
        except Exception as e:
            print(f"Database verification error: {str(e)}")
            self.verification_result = {"is_valid": False, "error": str(e)}
        finally:
            if verifier:
                verifier.close()

        # ── 5. Send latest transcription to WORKFLOW_API (LLM) ───
        print("\n=== Sending transcription to WORKFLOW_API (LLM) ===")
        if WORKFLOW_API:
            try:
                payload = {
                    "message": self.captured_transcription,
                    "customer": {
                        "connection_number": self.extracted_connection_number,
                        "customer_name": self.extracted_name,
                        "customer_nic": self.extracted_nic,
                    },
                }

                llm_resp = requests.post(WORKFLOW_API, json=payload, timeout=120.0)

                if llm_resp.ok:
                    self.llm_response = llm_resp.json() if llm_resp.text else "Processing..."
                    print("LLM response received")
                else:
                    self.llm_response = {"error": llm_resp.text}
                    print(f"LLM API error: {llm_resp.status_code}")
            except Exception as e:
                print(f"Error calling WORKFLOW_API: {str(e)}")
                self.llm_response = {"error": str(e)}
        else:
            print("WORKFLOW_API not configured in .env")
            self.llm_response = {"warning": "WORKFLOW_API not configured"}

        print("Second workflow completed")
