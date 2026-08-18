import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.transition_model import TransitionModel, LinearTransitionModel, TransformerTransitionModel

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dim", type=int, default=4096)
    parser.add_argument("--op_embed_dim", type=int, default=16)
    parser.add_argument("--mlp_hidden_dim", type=int, default=512) # Use 512 for MLP inner dim, and Transformer d_model
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--transformer_heads", type=int, default=8)
    args = parser.parse_args()

    linear_model = LinearTransitionModel(hidden_dim=args.hidden_dim, mlp_hidden_dim=args.mlp_hidden_dim, op_embed_dim=args.op_embed_dim)
    mlp_model = TransitionModel(hidden_dim=args.hidden_dim, mlp_hidden_dim=args.mlp_hidden_dim, op_embed_dim=args.op_embed_dim)
    transformer_model = TransformerTransitionModel(
        hidden_dim=args.hidden_dim, 
        mlp_hidden_dim=args.mlp_hidden_dim, 
        op_embed_dim=args.op_embed_dim,
        num_layers=args.transformer_layers,
        nhead=args.transformer_heads
    )

    print(f"Hidden Dim: {args.hidden_dim}")
    print(f"Linear Transition Model params:      {count_parameters(linear_model):,}")
    print(f"MLP Transition Model params:         {count_parameters(mlp_model):,}")
    print(f"Transformer Transition Model params: {count_parameters(transformer_model):,}")

if __name__ == "__main__":
    main()
