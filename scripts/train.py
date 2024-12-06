import os
import json
import torch
import logging
import argparse
from typing import Dict, List
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
)
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_dataset(data_path: str, tokenizer, validation_split: float = 0.1) -> Dict[str, Dataset]:
    """Load and prepare the dataset for training."""
    logger.info(f"Loading dataset from {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Format conversations for training
    formatted_data = []
    for item in tqdm(data, desc="Processing dataset"):
        text = f"### Instruction: {item['instruction']}\n\n### Response: {item['response']}\n\n"
        encodings = tokenizer(text, truncation=True, max_length=512, padding="max_length")
        formatted_data.append({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': encodings['input_ids'].copy()
        })
    
    # Create dataset and split into train/validation
    full_dataset = Dataset.from_list(formatted_data)
    split_dataset = full_dataset.train_test_split(test_size=validation_split, shuffle=True, seed=42)
    
    logger.info(f"Dataset split: {len(split_dataset['train'])} training examples, {len(split_dataset['test'])} validation examples")
    
    return {
        'train': split_dataset['train'],
        'validation': split_dataset['test']
    }

def create_model_and_tokenizer(model_name: str):
    """Initialize the model and tokenizer."""
    logger.info(f"Loading model and tokenizer: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model in half precision
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Configure LoRA with smaller rank and alpha for stability
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Apply to all attention modules
        modules_to_save=["embed_tokens", "lm_head"]  # Save important modules
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Print trainable parameters info
    
    return model, tokenizer

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    logits, labels = eval_pred
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Calculate loss
    loss_fct = torch.nn.CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    loss = loss_fct(shift_logits, shift_labels)
    
    # Calculate perplexity
    perplexity = torch.exp(loss)
    
    return {
        "loss": loss.item(),
        "perplexity": perplexity.item()
    }

def train(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 5e-6,
    validation_split: float = 0.1,
):
    """Train the model."""
    # Create model and tokenizer
    model, tokenizer = create_model_and_tokenizer(model_name)
    
    # Load and tokenize dataset
    datasets = load_dataset(dataset_path, tokenizer, validation_split)
    
    # Set training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        bf16=True,  # Use bfloat16 instead of fp16
        bf16_full_eval=True,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        report_to="none",
        optim="paged_adamw_32bit",  # Use 32-bit Adam
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        logging_dir=os.path.join(output_dir, "logs"),
        push_to_hub=False,
        ddp_find_unused_parameters=False,
    )
    
    # Create trainer with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['validation'],
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.01
            )
        ]
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    # Save best model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save training metrics
    metrics = trainer.evaluate()
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Final metrics: {metrics}")

def main():
    parser = argparse.ArgumentParser(description='Train the AI Therapist Assistant model')
    parser.add_argument('--model_name', type=str, required=True, help='Base model name')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to processed dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for trained model')
    parser.add_argument('--batch_size', type=int, default=2, help='Training batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--validation_split', type=float, default=0.1, help='Validation set size (0-1)')
    
    args = parser.parse_args()
    
    train(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split,
    )

if __name__ == "__main__":
    main() 