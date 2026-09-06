# LatentBench: When Geometry Misleads

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22546288.svg)](https://doi.org/10.5281/zenodo.22546288)

This repository contains the official code, synthetic datasets, and analysis for the manuscript:
**"When Geometry Misleads: Evaluating Semantic Fidelity in Latent Transition Models for LLM Reasoning"**
([https://github.com/BIGREASONS/Latent_Planning](https://github.com/BIGREASONS/Latent_Planning))

## Project Overview
Latent reasoning seeks to perform intermediate computation directly in the hidden-state space of a language model. This project evaluates whether geometric reconstruction metrics (like Cosine Similarity) adequately measure semantic fidelity in latent transition models. We introduce **LatentBench** to evaluate transition models through representation probing, generating explicit metrics for **Semantic Gain** and **Oracle Gap**.

## Environments & Installation
There are two supported ways to run this repository depending on your compute environment:

**1. Local / Server (Recommended)**
If you are running locally or on a standard compute server, install the dependencies via `pyproject.toml`:
```bash
pip install .
```
*(This explicitly installs `torch==2.8.0` and all other required packages).*

**2. Kaggle / Hosted Notebooks**
If you are running in Kaggle, `torch` and `torchvision` are already provided by the environment. Use the provided requirements file to avoid downgrading the pre-installed CUDA environment:
```bash
pip install -r requirements.txt
```

## Data Preparation
Datasets for this project (such as Countdown) are generated procedurally and do not require downloading external proprietary data.
To generate the primary dataset:
```bash
python scripts/generate_countdown_dataset.py
```

## Reproducing Phase 3 Experiments
The canonical experiments (Phase 3) evaluate Linear, MLP, and Transformer transition models on the Countdown task using a frozen TinyLlama teacher.

To run the full evaluation pipeline across all nine primary conditions (3 architectures x 3 seeds):
```bash
python scripts/run_multiseed.py --domain countdown --archs linear mlp transformer --seeds 42 43 44
```
*Limitation Note: While repository integrity and analysis verification have passed, the full nine-run training reproduction was not independently rerun in the clean-clone audit due to compute constraints.*

## Analysis and Artifacts
The canonical outputs mapping to the manuscript are located in the `analysis/` directory.
- `analysis/phase3_master_results.csv`: Contains the canonical Oracle Gap, Semantic Gain, and Cosine values for all 9 Phase 3 runs.
- `scripts/verify_paper.py`: Script to verify that the manuscript metrics perfectly match the generated raw outputs. Run via `python scripts/verify_paper.py manuscript.tex`.

## License
This repository uses a split licensing model:
- **Code**: The software code in this repository is licensed under the [MIT License](LICENSE).
- **Manuscript & Data**: The manuscript (`manuscript.pdf`, `manuscript.tex`), figures, generated datasets, and analysis outputs are licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. 

## Citation
If you use this software or research, please cite it using the metadata in `CITATION.cff`.
