from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class models:
    def __init__(self):
        self.model_path = r"model"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map={"": torch.device("cpu")},
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model.eval()