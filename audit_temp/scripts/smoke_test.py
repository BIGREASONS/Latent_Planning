import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""Smoke test to verify Phase A repository packaging and pipeline execution."""

import sys
import os
import torch
import warnings

def main():
    try:
        print("0. Testing Repository Root...")
        assert os.path.exists("scripts"), "Must run from repository root"
        assert os.path.exists("models"), "Must run from repository root"
        assert os.path.exists("training"), "Must run from repository root"
        assert os.path.exists("evaluation"), "Must run from repository root"
        print("   [OK] Root directory verified.")

        print("\n1. Testing Imports...")
        from models.model_loader import load_model, load_tokenizer
        from data_processing.trajectory_dataset import load_problems, build_trajectories
        from training.train_transition import train_transition_model, TransitionTrainConfig
        from training.train_decoder import train_decoder_model, DecoderTrainConfig
        print("   [OK] Imports successful.")
        
        print("\n2. Testing Dummy Dataset Generation...")
        import json
        os.makedirs("data", exist_ok=True)
        dummy_problems = [
            {"question": "80 25", "target": 105, "solution": "80 + 25 = 105"}
        ]
        with open("data/smoke_test.jsonl", "w") as f:
            for p in dummy_problems:
                f.write(json.dumps(p) + "\n")
        print("   [OK] Dataset creation successful.")

        print("\n3. Testing Mock Model Loading...")
        # Since we just want to test pipeline connectivity, we won't load TinyLlama 
        # as it would be too slow and require GPU. We'll mock the pipeline logic 
        # similar to what we do in tests/test_pipeline.py
        from data_processing.trajectory_dataset import Trajectory
        
        dummy_traj = Trajectory(
            all_hidden=torch.randn(10, 32),
            input_ids=torch.randint(0, 100, (10,)),
            state_indices=torch.tensor([0, 5, 9]),
            op_ids=torch.tensor([0, 1]),
            operands=torch.randn(2, 2),
            numbers=[80, 25],
            target=105
        )
        print("   [OK] Trajectory mocking successful.")

        print("\n4. Testing Transition Model Training...")
        t_config = TransitionTrainConfig(epochs=1, batch_size=1, mlp_hidden_dim=16)
        train_transition_model([dummy_traj, dummy_traj], config=t_config)
        print("   [OK] Transition training successful.")

        print("\n5. Testing Decoder Training...")
        d_config = DecoderTrainConfig(epochs=1, batch_size=1, mlp_hidden_dim=16)
        train_decoder_model([dummy_traj, dummy_traj], vocab_size=100, config=d_config)
        print("   [OK] Decoder training successful.")

        print("\n===========================================")
        print("SMOKE TEST PASSED: Repository is executable.")
        print("===========================================")
        sys.exit(0)
        
    except Exception as e:
        print("\n===========================================")
        print(f"SMOKE TEST FAILED: {type(e).__name__}")
        print(str(e))
        import traceback
        traceback.print_exc()
        print("===========================================")
        sys.exit(1)

if __name__ == "__main__":
    main()
