import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def tsne_plot(embeddings: torch.Tensor, labels: torch.Tensor, save_path: str = None):
    e = embeddings.detach().cpu().numpy()
    l = labels.detach().cpu().numpy()

    # sklearn compatibility: new versions use max_iter, older use n_iter
    try:
        tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30, max_iter=1000)
    except TypeError:
        tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30, n_iter=1000)

    xy = tsne.fit_transform(e)
    plt.figure(figsize=(6, 5))
    for lab in sorted(set(l.tolist())):
        m = (l == lab)
        plt.scatter(xy[m, 0], xy[m, 1], label=f"class {lab}", s=10, alpha=0.8)
    plt.legend(loc="best")
    plt.title("t-SNE of Query Embeddings")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
    return xy
