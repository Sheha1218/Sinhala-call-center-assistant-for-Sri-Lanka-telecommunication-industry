from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import LoraConfig,get_peft_model

class lora:
    def __init__(self):
        self.model_path='./model'

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model =AutoTokenizer.from_pretrained(self.model_path,device_map='cpu')


        lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=['q_proj','v_proj'],
    loara_dropout=0.05,
    bies='none',
    task_type='CAUSAL_LM'
)

        model =get_peft_model(model,lora_config)
        self.model.print_trainable_parameters()