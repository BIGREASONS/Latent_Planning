import os
from typing import Optional, Tuple
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig
)
from peft import (
    get_peft_model,
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training
)

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

def load_tokenizer(model_id: str = DEFAULT_MODEL) -> PreTrainedTokenizer:
    """Loads the tokenizer for the specified model."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model(
    model_id: str = DEFAULT_MODEL,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.float16,
    load_in_4bit: bool = False
) -> PreTrainedModel:
    """Loads a frozen base model for inference.

    When ``load_in_4bit`` is True, loads with 4-bit NF4 quantization (for the
    7B/8B teachers that do not fit in fp16 on a single 16GB GPU). No LoRA/PEFT
    is attached — this is inference-only; use ``load_qlora_model`` for trainable
    adapters.
    """
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=torch_dtype
    )
    return model

def load_qlora_model(
    model_id: str = DEFAULT_MODEL,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    device_map: str = "auto"
) -> PreTrainedModel:
    """Loads a model with 4-bit quantization and prepares it for QLoRA."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map
    )
    
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    return model

def save_checkpoint(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, output_dir: str):
    """Saves the model (or PEFT adapter) and tokenizer."""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def load_checkpoint(
    checkpoint_dir: str,
    base_model_id: str = DEFAULT_MODEL,
    is_peft: bool = True,
    device_map: str = "auto",
    load_in_4bit: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Loads a model from a saved checkpoint.

    If is_peft is True, loads the base model and attaches the saved LoRA adapters.

    Parameters
    ----------
    load_in_4bit : bool
        If True, load the base model with 4-bit NF4 quantization (for large
        models on limited VRAM). Default is False (fp16) to preserve backward
        compatibility with V5.3 checkpoints.
    """
    tokenizer = load_tokenizer(checkpoint_dir)
    
    if is_peft:
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                quantization_config=bnb_config,
                device_map=device_map
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                device_map=device_map,
                torch_dtype=torch.float16
            )
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            device_map=device_map,
            torch_dtype=torch.float16
        )
        
    return model, tokenizer
