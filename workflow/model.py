from fastapi import APIRouter, File, Form, UploadFile
import asyncio
import os
import torch
import pandas as pd
from pathlib import Path
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import io
import logging
from dotenv import load_dotenv
from .first import models

load_dotenv()

logger = logging.getLogger(__name__)


def get_telecom_knowledge(query: str = "") -> str:
    try:
        knowledge_prompt = f"""<s>### Instruction:
You are a Sri Lankan telecom expert. Provide brief, relevant telecom knowledge to assist customer support.
Only provide telecom-related information (SIM change, payment, connections, plans, troubleshooting, etc).

### Customer Query:
{query if query.strip() else "General telecom knowledge for Sri Lanka"}

### Telecom Knowledge:
"""
        
        inputs = models().tokenizer(
            knowledge_prompt,
            return_tensors="pt"
        ).to(models().model.device)
        
        with torch.no_grad():
            outputs = models().model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                eos_token_id=models().tokenizer.eos_token_id
            )
        
        knowledge = models().tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        if "### Telecom Knowledge:" in knowledge:
            knowledge = knowledge.split("### Telecom Knowledge:")[-1].strip()
        
        return knowledge if knowledge and len(knowledge.strip()) > 3 else "No knowledge available."
    except Exception as e:
        return f"Error generating knowledge: {str(e)}"
TTS_API = os.getenv('TTS_API_URL')
TTS_CREDENTIALS = os.getenv(
    "GOOGLE_TTS_CREDENTIALS",
    str(Path(__file__).parent.parent / "sinhala-call-center-agent-8031ac1e97ce.json")
)
USE_GOOGLE_TTS = Path(TTS_CREDENTIALS).exists()

workflow = APIRouter()

try:
    logger.info("Initializing LLM instance...")
    llm_ins = models()
    logger.info("LLM instance initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LLM instance: {str(e)}", exc_info=True)
    llm_ins = None

class LlmRequest(BaseModel):
    message: str
    session_id: str | None = None  


class TtsRequest(BaseModel):
    text: str
    
class modeloutput:
    def __init__(self):
        self.llm = llm_ins

        self.prompt = """<s>### Instruction:
You are a telecommunicational AI assistant specialized in handling customer inquiries for a Sri Lankan telecommunication industry.
You are fluent in Sinhala language and can understand and respond to customer.

## Your role and capabilities:
- Greet the customer in Sinhala.
- Ensure customer verification.
- Provide telecom services support:
  SIM change, Payment transfer, New connection,
  Post to pre and Ownership transfer.
- Close the conversation politely.

## Important guidelines:
- Use Sinhala + English mix.
- If you don't know the answer, say මොහොතක්  රැදී සිටින්න.
Example:
"Original receipt එක අරන් යන්න ඔනේ. Printout එකක් විදියට bank through තමයි සල්ලි වැටෙන්නේ"

### Customer:
"""

    def generate_reponse(self, message: str) -> str:
        try:
            logger.info(f"LLM generation started - Input: {message[:100]}{'...' if len(message) > 100 else ''}")
            full_prompt = self.prompt + message + "\n\n### Assistant:\n"

            inputs = self.llm.tokenizer(
                full_prompt,
                return_tensors="pt"
            ).to(self.llm.model.device)
            
            with torch.no_grad():
                outputs = self.llm.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    eos_token_id=self.llm.tokenizer.eos_token_id
                )

            reply = self.llm.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            if "### Assistant:" in reply:
                reply = reply.split("### Assistant:")[-1].strip()

           
            DONT_KNOW = "මොහොතක් රැදී සිටින්න"
            if not reply or len(reply.strip()) < 3 or "නොදනි" in reply or "දන්නේ නැ" in reply:
                logger.warning(f"LLM returned don't-know or invalid reply (length={len(reply) if reply else 0})")
                return DONT_KNOW
            logger.info(f"✓ AI message generation successful - Response: {reply[:100]}{'...' if len(reply) > 100 else ''}")
            return reply
        except Exception as e:
            logger.error(f"✗ LLM generation error: {e}", exc_info=True)
            return "මොහොතක් රැදී සිටින්න"

model_handler=modeloutput()

GREETING = "ආයුබෝවන් ABC ආයතනයෙන් මම කෙයාරා කෙසෙද මා ඔබට සහය වන්නේ."
FOLLOW_UP = "ඔබට තවත් දෙයක් ගැනිමට හෝ මගෙන් වෙනත් උපකාරයක් අවශදද?"
YES_PROMPT = "ඔව් මොකක්ද ඔබට මගෙන් අවශ්‍ය උපකාරය."
GOODBYE = "ඔබට අවශ්‍ය උපකාරය ලබා දීමට මට සතුටුයි. ඔබට සුභ දවසක්."
RATE_MSG = "Service එක rate කරන්න කියලා කාරුනිකව ඉල්ලා සිටිනවා."


@workflow.get("/phrases")
async def get_phrases():
    
    return {
        "greeting": GREETING,
        "follow_up": FOLLOW_UP,
        "yes_prompt": YES_PROMPT,
        "goodbye": GOODBYE,
        "rate_msg": RATE_MSG,
    }


@workflow.post("/tts")
async def tts_route(request: TtsRequest):
    
    try:
        if USE_GOOGLE_TTS:
            from tts.google_tts import text_to_speech
            loop = asyncio.get_event_loop()
            audio = await loop.run_in_executor(None, lambda: text_to_speech(request.text))
        elif TTS_API:
            async with httpx.AsyncClient() as client:
                tts_response = await client.post(TTS_API, json={"text": request.text}, timeout=60.0)
                tts_response.raise_for_status()
                audio = tts_response.content
        else:
            return {"error": "TTS not configured. Set GOOGLE_TTS_CREDENTIALS or TTS_API_URL."}
        return StreamingResponse(io.BytesIO(audio), media_type="audio/mpeg")
    except Exception as e:
        return {"error": str(e)}


@workflow.post("/llm")
async def llm_model(request: LlmRequest):
    msg = (request.message or "").strip()
    logger.info(f"LLM API request received - message={msg[:100]}{'...' if len(msg) > 100 else ''}, session_id={request.session_id}")
    llm_text = model_handler.generate_reponse(msg)
    logger.info(f"LLM API response generated - text={llm_text[:100]}{'...' if len(llm_text) > 100 else ''}")

    if request.session_id:
        try:
            from feedback.session_store import save_ai_response, save_customer_message
            save_customer_message(request.session_id, msg)
            save_ai_response(request.session_id, llm_text)
        except Exception:
            pass
        try:
            from feedback.feedback import save_ai_response_to_db
            save_ai_response_to_db(llm_text, request.session_id, msg)
        except Exception:
            pass

    return {"text": llm_text}


def _resample_to_16k(audio_bytes: bytes, from_rate: int) -> bytes:
    if from_rate == 16000:
        return audio_bytes
    import numpy as np
    arr = np.frombuffer(audio_bytes, dtype=np.int16)
    n = len(arr)
    new_n = int(n * 16000 / from_rate)
    indices = np.linspace(0, n - 1, new_n).astype(np.int32)
    resampled = arr[indices]
    return resampled.tobytes()


@workflow.post("/voice-to-llm")
async def voice_to_llm(
    audio: UploadFile = File(...),
    session_id: str = Form(None),
    sample_rate: str = Form("16000"),
):
  
    try:
        audio_bytes = await audio.read()
    except Exception as e:
        return {"error": f"Failed to read audio: {e}", "text": "", "transcript": ""}

    sr = int(sample_rate) if sample_rate else 16000
    if sr != 16000:
        audio_bytes = _resample_to_16k(audio_bytes, sr)

    if not audio_bytes:
        transcript = ""
    else:
        try:
            from stt.google_stt import speech_to_text
            loop = asyncio.get_event_loop()
            transcript = await loop.run_in_executor(
                None, lambda: speech_to_text(audio_bytes, sample_rate=16000)
            )
        except Exception as e:
            return {"error": f"STT failed: {e}", "text": "", "transcript": ""}

    print(f"[Voice-to-LLM] transcript={repr(transcript[:100])}{'...' if len(transcript) > 100 else ''}, session_id={session_id}")

    llm_text = model_handler.generate_reponse(transcript or "")

    if session_id:
        try:
            from feedback.session_store import save_ai_response, save_customer_message
            save_customer_message(session_id, transcript or "")
            save_ai_response(session_id, llm_text)
        except Exception:
            pass
        try:
            from feedback.feedback import save_ai_response_to_db
            save_ai_response_to_db(llm_text, session_id, transcript or None)
        except Exception:
            pass

    return {"text": llm_text, "transcript": transcript}


@workflow.post("/voice-to-text")
async def voice_to_text(
    audio: UploadFile = File(...),
    sample_rate: str = Form("16000"),
):
    try:
        audio_bytes = await audio.read()
    except Exception as e:
        return {"error": str(e), "transcript": ""}
    sr = int(sample_rate) if sample_rate else 16000
    if sr != 16000:
        audio_bytes = _resample_to_16k(audio_bytes, sr)
    if not audio_bytes:
        return {"transcript": ""}
    try:
        from stt.google_stt import speech_to_text
        loop = asyncio.get_event_loop()
        transcript = await loop.run_in_executor(
            None, lambda: speech_to_text(audio_bytes, sample_rate=16000)
        )
        return {"transcript": transcript}
    except Exception as e:
        return {"error": str(e), "transcript": ""}

