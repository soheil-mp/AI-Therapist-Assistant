import os
import torch
import logging
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quantize_model(model_path: str, output_path: str):
    """Quantize the model for efficient inference."""
    logger.info(f"Loading model from {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load PEFT adapter if it exists
    adapter_path = os.path.join(model_path, "adapter_model")
    if os.path.exists(adapter_path):
        logger.info("Loading PEFT adapter...")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    # Merge LoRA weights if applicable
    if hasattr(model, "merge_and_unload"):
        logger.info("Merging LoRA weights...")
        model = model.merge_and_unload()
    
    # Save quantized model
    logger.info(f"Saving quantized model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    logger.info("Quantization complete!")

def main():
    parser = argparse.ArgumentParser(description='Quantize the AI Therapist Assistant model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for quantized model')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    # Quantize model
    quantize_model(args.model_path, args.output_path)

if __name__ == "__main__":
    main() 