from fastapi import APIRouter
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
                max_new_tokens=80,
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


@workflow.post("/tts")
async def tts_route(request: TtsRequest):
    """Convert text to speech (for greeting etc). Uses sinhala-call-center-agent-8031ac1e97ce.json for Google Cloud TTS."""
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


@workflow.get("/greeting")
async def get_greeting():
    """Return greeting text for reference."""
    return {"text": GREETING}


@workflow.post("/llm")
async def llm_model(request: LlmRequest):
    """Finetuned LLM: message -> text -> TTS -> audio. Saves ai_response to session + customer_feedback.
    Always processes - even when message is empty (nothing captured)."""
    msg = (request.message or "").strip()
    print(f"[LLM API] POST /llm received: message={repr(msg[:100])}{'...' if len(msg) > 100 else ''}, session_id={request.session_id}")
    llm_text = model_handler.generate_reponse(msg)

    if request.session_id:
        try:
            from feedback.session_store import save_ai_response
            save_ai_response(request.session_id, llm_text)
        except Exception:
            pass
        try:
            from feedback.feedback import save_ai_response_to_db
            save_ai_response_to_db(llm_text, request.session_id)
        except Exception:
            pass

    if not USE_GOOGLE_TTS and not TTS_API:
        return {"error": "TTS not configured. Add sinhala-call-center-agent-8031ac1e97ce.json or set TTS_API_URL.", "text": llm_text}
    try:
        if USE_GOOGLE_TTS:
            from tts.google_tts import text_to_speech
            loop = asyncio.get_event_loop()
            audio = await loop.run_in_executor(None, lambda: text_to_speech(llm_text))
        else:
            async with httpx.AsyncClient() as client:
                tts_response = await client.post(TTS_API, json={"text": llm_text}, timeout=60.0)
                tts_response.raise_for_status()
                audio = tts_response.content
        return StreamingResponse(io.BytesIO(audio), media_type="audio/mpeg")
    except Exception as e:
        return {"error": str(e), "text": llm_text}

    
    
