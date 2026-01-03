from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
class models:
    def __init__(self):
            self.model_path =r'D:\Way to Denmark\Projects\Sinhala-call-center-assistant-for-Sri-Lanka-telecommunication-industry\model'
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model=AutoModelForCausalLM.from_pretrained(self.model_path,
                                                            device_map='auto',
                                                            torch_dtype=torch.floate16 if torch.cuda.is_available() else torch.float32)
            self.model.to('cpu')
            self.model.eval()
class models:
    def __init__(self):
            self.model_path = (r'D:\Way to Denmark\Projects\Sinhala-call-center-assistant-for-Sri-Lanka-telecommunication-industry\model')

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model=AutoModelForCausalLM.from_pretrained(self.model_path)
            self.model.eval()
            
            

