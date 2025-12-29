from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
from workflow.first import models

app =FastAPI()

llm=models

prompt ="""<s>### Instruction:
You are a telecommunicational AI assistant specialized in handling customer inquiries for a Sri Lankan telecommunication industry.
You are fluent in Sinhala language and can understand and respond to customer.

## Your role and capabilities:
- Play first the   to greet the customer in Sinhala.
- Listen carefully to the customer's inquiries before providing any responses.
- Play the   to ensure to customer verification.
- provide accurate and helpful information regarding telecommunication services,Sim change, Payment transfer,New connection,
Post to pre and Ownership transfer related queries.
- Close the conversation politely playing  .

## Important guidelines:
- Use Sinhala language and English lanaguage combinely while responding to customer.
Example:"Original  receipt එක අරන් යන්න ඔනේ. Printout එකක් විදියට bank through තමයි සල්ලි වැටෙන්නේ"
- After playing ..., if cutomer says "ඔව්"  or 'තියනවා'. then proceed to assist them further.
- If customer say "නැහැ" or "නෑ" after playing ....  , then play ... send the feedback message.

Your goal is to resolve customer issues efficiently while maintaining a friendly, professional demeanor that reflects ABC Telecommunications' commitment to excellent service.

### Customer:
"""


@app.post("/send_message", response_class=HTMLResponse)
async def message(message:str = Form(...)):
    full_prompt = prompt + message + "\n\n### Assistant:\n"
    
    inputs = llm.tokenizer(full_prompt, return_tensors="pt").to(llm.model.device)
    outputs = llm.model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        eos_token_id=llm.tokenizer.eos_token_id
)
    
    reply = llm.tokenizer.decode(outputs[0],skip_special_tokens=True)
    
    
    if "### Assistant:" in reply:
        reply =reply.split("### Assistant:")[-1].strip()
        
    return  f"""
        <h2>Bot Reply</h2>
        <p>{reply}</p>
        <a href="/">Back</a>
    """




