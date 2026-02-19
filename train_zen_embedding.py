#!/usr/bin/env python3
"""Training script for Zen Embedding."""

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import json

def train():
    """Train zen-embedding."""
    print("🚀 Training Zen Embedding")
    print("=" * 60)
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "./base-model",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained("./base-model")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load training data
    dataset = load_dataset('json', data_files='training_data.jsonl')['train']
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        learning_rate=2e-5,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        push_to_hub=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    trainer.train()
    
    # Save
    trainer.save_model("./finetuned")
    tokenizer.save_pretrained("./finetuned")
    
    print("✅ Training complete!")

if __name__ == "__main__":
    train()
