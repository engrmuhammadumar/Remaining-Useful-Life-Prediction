from dataclasses import dataclass

@dataclass
class Config:
    # Paths
    data_dir: str = r"E:\1 Paper MCT\Cutting Tool Paper\Dataset\cutting tool data\test_data_40_images"
    out_dir: str = "./runs/fsl_fault_diag"

    # Data
    image_size: int = 224
    normalize_mean = (0.485, 0.456, 0.406)     # ImageNet
    normalize_std  = (0.229, 0.224, 0.225)

    # Few-shot episode settings
    n_way: int = 5
    k_shot: int = 5
    q_queries: int = 10
    episodes_per_epoch: int = 120   # leaner & faster per epoch
    val_episodes: int = 60
    test_episodes: int = 120

    # Training
    max_epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Backbone / head
    embed_dim: int = 256            # smaller ⇒ much faster cov/solve
    use_pretrained: bool = False    # keep False to avoid weight downloads
    cov_shrinkage: float = 0.1
    temperature: float = 1.0

    # Repro / device
    seed: int = 42
    device: str = "cuda"            # "cuda" or "cpu"


