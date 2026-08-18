import argparse
import os
import sys
import subprocess
import shutil
import json

def check_dataset_domain(data_dir, expected_domain):
    train_path = os.path.join(data_dir, "train.jsonl")
    if not os.path.exists(train_path):
        return True, "Will generate dataset."
    
    try:
        with open(train_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if not first_line:
                return False, "train.jsonl is empty."
            data = json.loads(first_line)
            
            target = data.get("target")
            numbers = data.get("numbers", [])
            
            if expected_domain == "game24":
                if target != 24 or len(numbers) != 4:
                    return False, f"Expected Game24 (target=24, 4 numbers) but got target={target}, {len(numbers)} numbers."
            elif expected_domain == "countdown":
                # Countdown typically has 6 numbers and targets can vary, but definitely not strict Game24
                # If it happens to be 24 and 4 numbers, it's ambiguous, but usually it's 6 numbers.
                if target == 24 and len(numbers) == 4:
                    return False, "Dataset looks like Game24 but expected Countdown."
            return True, f"Dataset matches {expected_domain} characteristics."
    except Exception as e:
        return False, f"Failed to parse train.jsonl: {e}"

def main():
    parser = argparse.ArgumentParser(description="Latent Planning Preflight Checklist")
    parser.add_argument("--model", required=True, help="Model ID (e.g., deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)")
    parser.add_argument("--domain", required=True, choices=["countdown", "game24"], help="Problem domain")
    parser.add_argument("--out_dir", required=True, help="Output directory to test")
    parser.add_argument("--data_dir", required=True, help="Data directory to test")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directory")
    args = parser.parse_args()

    print(f"=== LATENT PLANNING PREFLIGHT CHECK ===")
    summary = []

    # 1. GPU / CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        gpu_name = torch.cuda.get_device_name(0)
        
        # VRAM Check
        free, total = torch.cuda.mem_get_info()
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        summary.append(f"GPU: {gpu_name} ({total_gb:.1f} GB total, {free_gb:.1f} GB free)")
        if free_gb < 14.0:
            print(f"[WARN] Less than 14GB free VRAM ({free_gb:.1f}GB). 7B models might OOM even in 4-bit.")
        summary.append("CUDA available")
    except Exception as e:
        print(f"[FAIL] GPU/CUDA check: {e}")
        sys.exit(1)
        
    # 2. Dependencies & Versions
    try:
        import transformers
        import bitsandbytes
        import accelerate
        import peft
        print(f"[INFO] Dependencies: torch={torch.__version__}, transformers={transformers.__version__}, "
              f"accelerate={accelerate.__version__}, bitsandbytes={bitsandbytes.__version__}, peft={peft.__version__}")
        summary.append("Dependencies OK")
    except ImportError as e:
        print(f"[FAIL] Missing dependency: {e}")
        sys.exit(1)

    # 3. Fast Tokenizer & Model Accessibility (meta device)
    try:
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
        print("[..] Checking Hugging Face connection and model access...")
        config = AutoConfig.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if not getattr(tokenizer, "is_fast", False):
            print(f"[FAIL] Tokenizer is not fast (tokenizer.is_fast == False). Required for offset mappings.")
            sys.exit(1)
        summary.append("Fast tokenizer")
        
        # Meta initialization
        with torch.device("meta"):
            _ = AutoModelForCausalLM.from_config(config)
        summary.append("Model accessible")
    except Exception as e:
        err_str = str(e).lower()
        if "401" in err_str or "403" in err_str or "gated" in err_str or "token" in err_str:
            print(f"[FAIL] Authentication Error: The model '{args.model}' requires a valid Hugging Face token. "
                  "Please login via `huggingface-cli login` or set the HF_TOKEN environment variable.")
        else:
            print(f"[FAIL] Model access failed: {e}")
        sys.exit(1)

    # 4. Required Repository Files
    critical_paths = [
        "scripts/run_phase_a.py",
        "requirements.txt",
        "evaluation/",
        "training/",
        "data_processing/"
    ]
    for p in critical_paths:
        if not os.path.exists(p):
            print(f"[FAIL] Critical repository path missing: {p}")
            sys.exit(1)
    summary.append("Repository structure OK")

    # 5. Disk Space
    try:
        total_b, used_b, free_b = shutil.disk_usage(".")
        free_gb = free_b / (1024**3)
        if free_gb < 10.0:
            print(f"[WARN] Low disk space in current directory: {free_gb:.1f} GB free. Long runs might fill the disk.")
        else:
            print(f"[INFO] Disk space: {free_gb:.1f} GB free.")
    except Exception as e:
        print(f"[WARN] Could not check disk space: {e}")

    # 6. Output Directory Collision & Writable Check
    if os.path.exists(args.out_dir):
        # Check for collision artifacts
        artifacts = ["probe_results.csv", "transition_model_action.pt", "trajectories/train.pt"]
        collision = [a for a in artifacts if os.path.exists(os.path.join(args.out_dir, a))]
        
        if collision and not args.force:
            print(f"[FAIL] Output collision: --out_dir '{args.out_dir}' contains artifacts from a previous run: {collision}")
            print("Use --force to overwrite or specify a new --out_dir.")
            sys.exit(1)
        elif len(os.listdir(args.out_dir)) > 0 and not args.force:
            print(f"[FAIL] --out_dir '{args.out_dir}' already exists and is not empty. Use --force to proceed.")
            sys.exit(1)
            
    # Dry-run write check
    try:
        os.makedirs(args.out_dir, exist_ok=True)
        test_file = os.path.join(args.out_dir, ".test_write")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        summary.append("Output directory writable")
    except Exception as e:
        print(f"[FAIL] --out_dir '{args.out_dir}' is not writable: {e}")
        sys.exit(1)

    # 7. Data Directory & Dataset Sanity
    try:
        os.makedirs(args.data_dir, exist_ok=True)
        test_file = os.path.join(args.data_dir, ".test_write")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        print(f"[FAIL] --data_dir '{args.data_dir}' is not writable: {e}")
        sys.exit(1)
        
    valid, msg = check_dataset_domain(args.data_dir, args.domain)
    if not valid:
        print(f"[FAIL] Dataset sanity check failed in {args.data_dir}: {msg}")
        sys.exit(1)
    else:
        print(f"[INFO] Dataset sanity: {msg}")
    summary.append("Dataset directory valid")
            
    print("-" * 40)
    print("=== PREFLIGHT SUMMARY ===")
    for item in summary:
        print(f"✓ {item}")
    print("✓ Ready to launch")

if __name__ == "__main__":
    main()
