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
from dotenv import load_dotenv
from .first import models

load_dotenv()


_telecom_knowledge_df = None


def get_telecom_knowledge(query: str = "") -> str:
    
    global _telecom_knowledge_df
    data_path = Path(__file__).parent.parent / "data.csv"
    if not data_path.exists():
        return "Telecom knowledge base not found. Use general Sri Lankan telecom guidelines."
    try:
        if _telecom_knowledge_df is None:
            _telecom_knowledge_df = pd.read_csv(data_path)
        df = _telecom_knowledge_df
      
        proc_col = "process" if "process" in df.columns else df.columns[-1]
        svc_col = df.columns[0]
        if query and query.strip():
            q = query.strip().lower()
            mask = df.astype(str).apply(
                lambda r: q in " ".join(r.values).lower(),
                axis=1,
            )
            df = df[mask] if mask.any() else df
        processes = df.head(5)[proc_col].dropna().astype(str).tolist()
        return "\n\n".join(processes) if processes else "No matching knowledge found."
    except Exception as e:
        return f"Error loading knowledge: {str(e)}"
TTS_API = os.getenv('TTS_API_URL')
TTS_CREDENTIALS = os.getenv(
    "GOOGLE_TTS_CREDENTIALS",
    str(Path(__file__).parent.parent / "sinhala-call-center-agent-8031ac1e97ce.json")
)
USE_GOOGLE_TTS = Path(TTS_CREDENTIALS).exists()

workflow = APIRouter()

llm_ins=models()

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
        full_prompt = self.prompt + message + "\n\n### Assistant:\n"

        inputs = self.llm.tokenizer(
            full_prompt,
            return_tensors="pt"
        ).to(self.llm.model.device)
        
        with torch.no_grad():
            outputs = self.llm.model.generate(
                **inputs,
                max_new_tokens=50,
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
            return DONT_KNOW
        return reply

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
    print(f"[LLM API] POST /llm received: message={repr(msg[:100])}{'...' if len(msg) > 100 else ''}, session_id={request.session_id}")
    llm_text = model_handler.generate_reponse(msg)

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

    # Return text only - frontend uses Speech Synthesis to speak it
    return {"text": llm_text}


def _resample_to_16k(audio_bytes: bytes, from_rate: int) -> bytes:
    """Resample 16-bit PCM from from_rate to 16000 Hz."""
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
    """
    Receive raw audio (PCM 16-bit mono), run STT on backend, then LLM.
    Frontend sends captured audio; backend does STT + LLM.
    """
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
    """STT only - returns transcript. Used for yes/no follow-up."""
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

