import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_loader import load_qlora_model, load_tokenizer

def format_prompt(example):
    """Formats the Countdown problem into a text prompt."""
    prompt = f"Problem: Given the numbers {example['numbers']}, reach the target {example['target']}.\n"
    prompt += f"Solution:\n{example['cot']}"
    return {"text": prompt}

def main():
    parser = argparse.ArgumentParser(description="Train Teacher Model on Countdown")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing train.jsonl and val.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    # 1. Load Tokenizer & Model
    print("Loading model and tokenizer...")
    tokenizer = load_tokenizer()
    model = load_qlora_model()

    # 2. Load Dataset
    print("Loading datasets...")
    dataset = load_dataset("json", data_files={
        "train": os.path.join(args.data_dir, "train.jsonl"),
        "validation": os.path.join(args.data_dir, "val.jsonl")
    })

    # Format and tokenize
    dataset = dataset.map(format_prompt)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

    # 3. Setup Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="logs",
        logging_steps=10,
        report_to="wandb",
        run_name="countdown-teacher-sft",
        fp16=True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
    )

    # 4. Train
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save final model
    trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
    print("Training complete!")

if __name__ == "__main__":
    main()
