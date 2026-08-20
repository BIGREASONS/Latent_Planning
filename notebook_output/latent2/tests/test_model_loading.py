import pytest
from models.model_loader import (
    load_tokenizer,
    load_model,
    load_qlora_model,
    load_checkpoint,
)
import torch
import inspect

# Use a tiny dummy model for testing infrastructure instead of the full 1.1B model
TEST_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires CUDA for bitsandbytes"
)
def test_load_qlora_model():
    model = load_qlora_model(model_id=TEST_MODEL)
    assert model is not None
    # Check if peft is applied (should have active_adapters)
    assert hasattr(model, "active_adapters")


def test_load_tokenizer():
    tokenizer = load_tokenizer(model_id=TEST_MODEL)
    assert tokenizer is not None
    assert tokenizer.pad_token is not None


def test_load_base_model():
    model = load_model(model_id=TEST_MODEL, device_map="cpu")
    assert model is not None
    assert isinstance(model, torch.nn.Module)


def test_load_model_4bit_defaults_false():
    """load_model defaults to load_in_4bit=False (V5.3 compatible)."""
    sig = inspect.signature(load_model)
    assert sig.parameters["load_in_4bit"].default is False


def test_load_checkpoint_4bit_defaults_false():
    """load_checkpoint defaults to load_in_4bit=False — 4-bit is opt-in only.

    This is the critical V5.3 backward-compatibility check: older checkpoints
    saved in fp16 must load in fp16 by default, not be forced into 4-bit."""
    sig = inspect.signature(load_checkpoint)
    assert (
        "load_in_4bit" in sig.parameters
    ), "load_checkpoint is missing the load_in_4bit parameter"
    assert (
        sig.parameters["load_in_4bit"].default is False
    ), "load_checkpoint should default to load_in_4bit=False for V5.3 compat"
