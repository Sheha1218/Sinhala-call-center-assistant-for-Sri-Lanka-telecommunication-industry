from trl import PPOTrainer,PPOConfig
import torch
import pandas as pd
from RL_enviroment.lora_adapter import lora

model_path = "./model"

lora =lora
class rl_agent:
    def __init__(self):
        config =PPOConfig(
        learning_rate=5e-6,
        batch_size=5,
        mini_batch_size=1,
    
)

        ppo_trainer =PPOTrainer(
        config=config,
        model=lora.model,
        tokenizer=lora.tokenizer
)

        prompts =["""<s>### Instruction:

config =PPOConfig(
    learning_rate=5e-6,
    batch_size=5,
    mini_batch_size=1,
    
)

ppo_trainer =PPOTrainer(
    config=config,
    model=lora.model,
    tokenizer=lora.tokenizer
)

prompts =["""<s>### Instruction:
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
"""]

        response = [pd.read_csv('data.csv')]

        rating =["x"]

        rewards =[(r-10)/10 for r in rating ]

        for prompt ,response,reward in zip(prompts, response,rewards):
            q = lora.tokenizer(prompt,return_tensors='pt').to(lora.model.device)
            r = lora.tokenizer(response,return_tensors='pt').to(lora.model.device)
    
    
        ppo_trainer.step(
            queries=[q['input']],
         response =[r['input']],
            rewards=[reward]
        
    )
    

        ppo_trainer.model.save_prereained()
                
response = [pd.read_csv('data.csv')]

rating =["x"]

rewards =[(r-10)/10 for r in rating ]

for prompt ,response,reward in zip(prompts, response,rewards):
    q = lora.tokenizer(prompt,return_tensors='pt').to(lora.model.device)
    r = lora.tokenizer(response,return_tensors='pt').to(lora.model.device)
    
    
    ppo_trainer.step(
        queries=[q['input']],
        response =[r['input']],
        rewards=[reward]
        
    )
    

ppo_trainer.model.save_prereained()
                
        
        
            
        
    
    

ppo_trainer.model.save_prereained()
