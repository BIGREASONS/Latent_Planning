import os
import json
import torch
import pytest
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset

from models.model_loader import load_model, load_tokenizer
from scripts.extract_hidden_states import extract_hidden_states

TEST_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"

@pytest.fixture
def dummy_dataset():
    data = [
        {"numbers": [1, 2, 3], "target": 6, "cot": "1+2=3\n3+3=6"},
        {"numbers": [2, 4], "target": 8, "cot": "2*4=8"}
    ]
    return data

def test_extraction(tmp_path, dummy_dataset):
    model = load_model(model_id=TEST_MODEL, device_map="cpu")
    tokenizer = load_tokenizer(model_id=TEST_MODEL)
    
    output_file = os.path.join(tmp_path, "hidden_states.pt")
    
    # Run extraction (extract from layer 1)
    extract_hidden_states(model, tokenizer, dummy_dataset, target_layer=1, output_file=output_file)
    
    assert os.path.exists(output_file)
    extracted = torch.load(output_file)
    
    assert len(extracted) > 0
    assert "sample_id" in extracted[0]
    assert "token_id" in extracted[0]
    assert "hidden_state" in extracted[0]
    
def test_dummy_train_loop(tmp_path, dummy_dataset):
    model = load_model(model_id=TEST_MODEL, device_map="cpu")
    tokenizer = load_tokenizer(model_id=TEST_MODEL)
    
    # Format dataset
    formatted = [{"text": f"Problem: {d['target']}\nSolution:\n{d['cot']}"} for d in dummy_dataset]
    ds = Dataset.from_list(formatted)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=32)
        
    tokenized_ds = ds.map(tokenize_function, batched=True)
    
    training_args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        num_train_epochs=1,
        save_strategy="no",
        logging_steps=1,
        report_to="none"
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )
    
    # Simply verify that the train loop can start without crashing
    train_result = trainer.train()
    assert train_result.global_step > 0
