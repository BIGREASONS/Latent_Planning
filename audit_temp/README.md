# Diagnosing Failure Modes of Latent Planning in Frozen Language Models

This repository contains the infrastructure and boilerplate for exploring latent planning failure modes in frozen LLMs.

## Repository Structure

- `configs/`: YAML and Python configuration files.
- `data/`: Local raw data.
- `datasets/`: Dataset generation logic and saved datasets.
- `models/`: Utilities for loading and handling base models and PEFT adapters.
- `training/`: Core training loops, Trainer wrappers, and logging utils.
- `evaluation/`: Evaluators and plotting tools.
- `scripts/`: Entry points for generating datasets, training, and extracting hidden states.
- `tests/`: Pytest suite to verify the pipeline.
- `notebooks/`: Jupyter notebooks for exploratory data analysis.
- `reports/`: Generated figures and CSV logs.

## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # OR
   pip install -e .[dev]
   ```

2. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your keys (WANDB_API_KEY, HF_TOKEN)
   ```

## Usage Guides

See the main documentation sections below:
1. **Dataset Generation:** `python scripts/generate_countdown_dataset.py --help`
2. **Training:** `python scripts/train_teacher.py --help`
3. **Hidden State Extraction:** `python scripts/extract_hidden_states.py --help`
4. **Testing:** `pytest tests/`
