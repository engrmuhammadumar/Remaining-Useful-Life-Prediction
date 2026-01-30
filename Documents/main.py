import argparse, os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # path safety

from config import Config
from engine import run_training

def parse_args():
    p = argparse.ArgumentParser(description="Few-Shot Fault Diagnosis (Adaptive Proto + Mahalanobis, fast)")
    cfg = Config()
    p.add_argument("--data_dir", type=str, default=cfg.data_dir, help="Path to folder-of-folders images")
    p.add_argument("--out_dir", type=str, default=cfg.out_dir)
    p.add_argument("--n_way", type=int, default=cfg.n_way)
    p.add_argument("--k_shot", type=int, default=cfg.k_shot)
    p.add_argument("--q_queries", type=int, default=cfg.q_queries)
    p.add_argument("--epochs", type=int, default=cfg.max_epochs)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = Config()
    cfg.data_dir = args.data_dir
    cfg.out_dir = args.out_dir
    cfg.n_way = args.n_way
    cfg.k_shot = args.k_shot
    cfg.q_queries = args.q_queries
    cfg.max_epochs = args.epochs
    run_training(cfg)
