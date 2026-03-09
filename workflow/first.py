from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class models:
    def __init__(self):
        self.model_path = r"model"
        try:
            logger.info(f"Starting model loading from path: {self.model_path}")
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            logger.info("Tokenizer loaded successfully")
            
            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=torch.float32,
                low_cpu_mem_usage=True,
            ).to("cpu")
            logger.info("Model loaded successfully")
            
            self.model.eval()
            logger.info("Model set to evaluation mode")
            logger.info("✓ Model initialization completed successfully")
        except FileNotFoundError as e:
            logger.error(f"✗ Model path not found: {self.model_path} - {str(e)}")
            raise
        except Exception as e:
            logger.error(f"✗ Error loading model: {str(e)}", exc_info=True)
            raise