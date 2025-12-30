from transformers import AutoTokenizer,AutoModelForCausalLM
class models:
    def __init__(self):
            self.model_path = (r'D:\Way to Denmark\Projects\Sinhala-call-center-assistant-for-Sri-Lanka-telecommunication-industry\model')

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model=AutoModelForCausalLM.from_pretrained(self.model_path)
            self.model.eval()