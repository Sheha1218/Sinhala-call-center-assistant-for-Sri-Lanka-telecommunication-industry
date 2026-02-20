from fastapi import APIRouter
import os
import torch
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import io
from dotenv import load_dotenv
from .first import models

load_dotenv()
TTS_API=os.getenv('TTS_API_URL')

workflow=APIRouter()

llm_ins=models()

class LlmRequest(BaseModel):
    message:str
    
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
                top_p=0.9,
                temperature=0.4,
                eos_token_id=self.llm.tokenizer.eos_token_id
        )

        reply = self.llm.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        if "### Assistant:" in reply:
            reply = reply.split("### Assistant:")[-1].strip()

        return reply

model_handler=modeloutput()

@workflow.post("/llm")
async def llm_model(request:LlmRequest):
    llm_text=model_handler.generate_reponse(request.message)
    
    try:
        async with httpx.AsyncClient() as client:
            tts_response=await client.post(TTS_API,json={"text":llm_text}, timeout=60.0)
            tts_response.raise_for_status()
        
    except Exception as e:
        return {"error": str(e)}
    
    return StreamingResponse(io.BytesIO(tts_response.content),media_type="audio/mpeg")

    
    
