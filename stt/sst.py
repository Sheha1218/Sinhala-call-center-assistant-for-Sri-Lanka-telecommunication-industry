from fastapi import APIRouter
from pydantic import BaseModel
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import os
import time
import requests
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import create_engine, text
from datetime import datetime
from urllib.parse import quote_plus


load_dotenv()


api_key = os.getenv("OPENAI_API_KEY")
db_url = os.getenv("db_url")
VERIFICATION_API = os.getenv("VERIFICATION_API", "http://127.0.0.1:8000/verify-and-process")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

if not db_url:
    raise ValueError("db_url not found in .env file")


client = OpenAI(api_key=api_key)
engine = create_engine(db_url)

STT_route= APIRouter()

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 30

def record_audio(duration, sample_rate):
    print("Listening...")
    audio = sd.rec(int(duration * sample_rate),
                   samplerate=sample_rate,
                   channels=CHANNELS,
                   dtype='int16')
    sd.wait()
    return audio

def transcribe_audio(audio_data, sample_rate):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        wav.write(tmpfile.name, sample_rate, audio_data)
        tmpfile_path = tmpfile.name

    try:
        with open(tmpfile_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text
    finally:
        os.remove(tmpfile_path)

def main():
    print("Real-Time Multilingual Whisper Started")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            audio_data = record_audio(CHUNK_DURATION, SAMPLE_RATE)
            text = transcribe_audio(audio_data, SAMPLE_RATE)

            if text.strip():
                print("Transcription:", text)
            else:
                print("...")

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("Stopped.")


def save_transcription_to_db(transcription: str):
    """Save transcription to the database"""
    try:
        with engine.connect() as connection:
            # Create table if it doesn't exist
            create_table_query = """
            CREATE TABLE IF NOT EXISTS stt_transcriptions (
                id SERIAL PRIMARY KEY,
                transcription TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            connection.execute(text(create_table_query))
            
            # Insert the transcription
            insert_query = """
            INSERT INTO stt_transcriptions (transcription, created_at)
            VALUES (:transcription, :created_at)
            """
            connection.execute(
                text(insert_query),
                {
                    "transcription": transcription,
                    "created_at": datetime.now()
                }
            )
            connection.commit()
            return True
    except Exception as e:
        print(f"Error saving to database: {str(e)}")
        return False


class TranscribeRequest(BaseModel):
    duration: int = 5  # Duration in seconds to record
    connection_number: str
    customer_nic: str
    customer_name: str


@STT_route.post("/transcribe")
async def transcribe_endpoint(request: TranscribeRequest):
    """
    API endpoint to capture speech from microphone and convert to text
    Records audio from mic for specified duration, transcribes it, and sends to verification API
    """
    try:
        # Record from microphone
        print(f"Recording for {request.duration} seconds...")
        audio_data = record_audio(request.duration, SAMPLE_RATE)
        
        # Transcribe audio
        transcription_text = transcribe_audio(audio_data, SAMPLE_RATE)
        
        print(f"Transcription: {transcription_text}")
        
        # Send to verification router
        verification_payload = {
            "connection_number": request.connection_number,
            "customer_nic": request.customer_nic,
            "customer_name": request.customer_name,
            "transcription": transcription_text
        }
        
        verification_response = requests.post(
            VERIFICATION_API,
            json=verification_payload,
            timeout=60.0
        )
        
        return {
            "status": "success",
            "transcription": transcription_text,
            "timestamp": datetime.now().isoformat(),
            "duration": request.duration,
            "verification_status": verification_response.json() if verification_response.ok else {"error": verification_response.text}
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


