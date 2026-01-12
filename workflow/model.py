import torch
from workflow.first import models
 

class modeloutput:
    def __init__(self):
        self.llm = models()

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

    def message(self, message: str) -> str:
        full_prompt = self.prompt + message + "\n\n### Assistant:\n"

        inputs = self.llm.tokenizer(
            full_prompt,
            return_tensors="pt"
        ).to(self.llm.model.device)

        outputs = self.llm.model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            eos_token_id=self.llm.tokenizer.eos_token_id
        )

        reply = self.llm.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        if "### Assistant:" in reply:
            reply = reply.split("### Assistant:")[-1].strip()

        return reply