from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import io
from google.cloud import texttospeech_v1 as texttospeech

TTS_route = APIRouter()


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "sinhala-call-center-agent.json"

client = texttospeech.TextToSpeechClient()


class TTSRequest(BaseModel):
    text: str


@TTS_route.post("/tts")
async def tts(request: TTSRequest):

    
    synthesis_input = texttospeech.SynthesisInput(text=request.text)

    # Sinhala female voice
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

    return StreamingResponse(
        audio_bytes,
        media_type="audio/mpeg"
    )
