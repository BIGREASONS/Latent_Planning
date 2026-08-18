import cProfile
import pstats
import io
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_processing.trajectory_dataset import load_trajectories
from training.train_transition import TransitionTrainConfig, train_transition_model

def main():
    print("[Profile] Loading trajectories...")
    train_traj_path = "reports/trajectories/train.pt"
    # only load first 100 trajectories for quick profile
    trajs = load_trajectories(train_traj_path)[:500]
    
    config = TransitionTrainConfig(
        epochs=1,
        batch_size=256
    )
    
    # --- PHASE 5: CPU Hotspots (cProfile) ---
    print("[Profile] Running cProfile...")
    pr = cProfile.Profile()
    pr.enable()
    train_transition_model(trajs, None, config)
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(50)
    
    with open("cpu_hotspots_raw.txt", "w") as f:
        f.write(s.getvalue())
        
    # --- PHASE 6 & 8: GPU Hotspots & Memory (torch.profiler) ---
    print("[Profile] Running torch.profiler...")
    
    config.epochs = 1
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("model_training"):
            train_transition_model(trajs, None, config)
            
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=20))
    
    with open("gpu_hotspots_raw.txt", "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
        f.write("\n\n=== MEMORY PROFILE ===\n")
        f.write(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=50))

if __name__ == "__main__":
    main()
