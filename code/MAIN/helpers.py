"""
Utilities for running multivariable and multivariate regression in SHCS cohort.

Functions exported
- plot_patient_embeddings
- get_device
- get_gpu_memory
- merge_dfs
- indices_of_subset
- load_graph
- node_labels
- get_split_indices_from_masks
- extract_node_embeddings_fullgraph
- save_gnn_embeddings_pickle
- save_pytorch_model
"""
import os
import warnings
import pickle
import pandas as pd
import numpy as np
import networkx as nx
from typing import Optional, Tuple, Iterable, Union, Callable, Any, Dict, Hashable, Literal, Union, List, Sequence
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

__all__ = [
    "plot_patient_embeddings",
    "get_device",
    "get_gpu_memory",
    "merge_dfs",
    "indices_of_subset",
    "load_graph",
    "node_labels",
    "load_pytorch_model",
    "get_split_indices_from_masks",
    "extract_node_embeddings_fullgraph",
    "save_gnn_embeddings_pickle",
    "save_pytorch_model"
]

def save_pytorch_model(
    model,
    path: str,
    optimizer=None,
    extra: dict | None = None,
    save_entire_model: bool = False,
):
    """
    Save a PyTorch model alongside the embeddings file as a .pth file.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model.
    path : str
        Path to save the .pth file.
    optimizer : torch.optim.Optimizer | None
        Optional optimizer to save state from.
    extra : dict | None
        Optional metadata/config to include.
    save_entire_model : bool
        If True, saves the full model object.
        If False, saves only state_dict (recommended).

    Returns
    -------
    model_path : str
        Path to the saved .pth file.
    """
    base, _ = os.path.splitext(path)
    model_path = base + ".pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if save_entire_model:
        payload = {
            "model": model,
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "extra": extra,
        }
    else:
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "extra": extra,
        }

    torch.save(payload, model_path)
    return model_path

def save_gnn_embeddings_pickle(
    out_path: str,
    H: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    y: np.ndarray | None = None,
    extra: dict | None = None,
):
    """
    Save embeddings + split indices (and optionally labels/extra metadata) to a pickle file.

    Parameters
    ----------
    out_path : str
        Where to write the .pkl file.
    H : np.ndarray
        Node embeddings, shape [N, d]
    train_idx, val_idx, test_idx : np.ndarray
        Index arrays into H (and y if provided)
    y : np.ndarray | None
        Optional multi-label targets aligned with H, shape [N, C]
    extra : dict | None
        Any additional metadata you want to store (config, model name, etc.)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    payload = {
        "embeddings": H.astype(np.float32, copy=False),
        "train_idx": train_idx.astype(np.int64, copy=False),
        "val_idx": val_idx.astype(np.int64, copy=False),
        "test_idx": test_idx.astype(np.int64, copy=False),
    }
    if y is not None:
        payload["y"] = y  # could cast to float32 if you want
    if extra is not None:
        payload["extra"] = extra

    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    return out_path

def get_split_indices_from_masks(data):
    """
    Returns train/val/test indices (numpy arrays) from boolean masks on a PyG Data object.
    """
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
    val_idx   = data.val_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
    test_idx  = data.test_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
    return train_idx, val_idx, test_idx


@torch.no_grad()
def extract_node_embeddings_fullgraph(
    model,
    data,
    device=None,
    return_logits=False,
    use_encoder_attr=True,
):
    """
    Extracts embeddings for *all nodes* in `data` in one forward pass (full-batch).

    Assumes your model forward returns: (embeddings, logits)
      h, logits = model(x, edge_index)

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyG model (e.g., GraphSageSimple)
    data : torch_geometric.data.Data
        Must contain x and edge_index
    device : str or torch.device
        If None, inferred from model parameters.
    return_logits : bool
        If True, also return logits.
    use_encoder_attr : bool
        If True and model has `.encoder`, returns model.encoder(x, edge_index) as embeddings.
        Otherwise uses the first output of model(x, edge_index).

    Returns
    -------
    H : np.ndarray, shape [N, emb_dim]
        Node embeddings
    (optional) Z : np.ndarray, shape [N, num_labels]
        Node logits
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    h, logits = model(x, edge_index)
    H = h.detach().cpu().numpy()

    if return_logits:
        if logits is None:
            raise ValueError("No logits available to return. Set use_encoder_attr=False or ensure model returns logits.")
        Z = logits.detach().cpu().numpy()
        return H, Z

    return H


def load_pytorch_model(
    model,
    model_path: str,
    optimizer=None,
    map_location="cpu",
    strict: bool = True,
):
    """
    Load a PyTorch checkpoint saved with save_pytorch_model(..., save_entire_model=False).

    Parameters
    ----------
    model : torch.nn.Module
        An instantiated model with the same architecture as the saved one.
    model_path : str
        Path to the .pth checkpoint file.
    optimizer : torch.optim.Optimizer | None
        Optional optimizer to restore state into.
    map_location : str or torch.device
        Device mapping for torch.load.
    strict : bool
        Passed to model.load_state_dict(...).

    Returns
    -------
    model : torch.nn.Module
        Model with loaded weights.
    optimizer : torch.optim.Optimizer | None
        Optimizer with loaded state if provided and present in checkpoint.
    extra : dict | None
        Extra metadata saved in the checkpoint.
    checkpoint : dict
        Full loaded checkpoint.
    """
    checkpoint = torch.load(model_path, map_location=map_location)

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain 'model_state_dict'. "
            "It may have been saved as a full model instead."
        )

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    extra = checkpoint.get("extra", None)
    return model, optimizer, extra, checkpoint

def indices_of_subset(arr: np.ndarray, subset: np.ndarray) -> np.ndarray:
    """
    Return indices of each item in `subset` within `arr` (1D).
    - If an item appears multiple times in `arr`, returns the index of its first occurrence.
    - If an item from `subset` is not in `arr`, returns -1 for that position.

    Parameters
    ----------
    arr : np.ndarray
        1D array to search within.
    subset : np.ndarray
        1D array of items to locate in `arr`.

    Returns
    -------
    np.ndarray
        1D array of indices with the same shape as `subset`.
    """
    arr = np.asarray(arr).ravel()
    subset = np.asarray(subset).ravel()

    order = np.argsort(arr)
    arr_sorted = arr[order]
    pos = np.searchsorted(arr_sorted, subset, side='left')

    # Determine which subset elements are present
    present = (pos < arr_sorted.size) & (arr_sorted[pos] == subset)

    out = np.full(subset.shape, -1, dtype=int)
    out[present] = order[pos[present]]
    return out


def merge_dfs(left_df, right_df):
    """
    Merges two DataFrames on their indexes with an outer join method.

    Parameters:
        left_df (pd.DataFrame): The left DataFrame to merge.
        right_df (pd.DataFrame): The right DataFrame to merge.

    Returns:
        pd.DataFrame: The resulting DataFrame after merging.
    """    
    # Merging on 'key' and expanding with 'how=outer' to include all records
    return pd.merge(left_df, right_df, left_index=True, right_index=True, how='outer')

def get_gpu_memory():
    """
    Retrieves and prints the current GPU memory usage statistics including total, reserved, and allocated memory amounts.

    Returns:
        None
    """    
    t = torch.cuda.get_device_properties(0).total_memory*(1*10**-9)             
    r = torch.cuda.memory_reserved(0)*(1*10**-9)
    a = torch.cuda.memory_allocated(0)*(1*10**-9)
    
    return print("Total = %1.1fGb \t Reserved = %1.1fGb \t Allocated = %1.1fGb" % (t,r,a))

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def plot_patient_embeddings(
    embeddings: pd.DataFrame,
    labels: pd.Series,
    method: str = "UMAP",               # "PCA", "TSNE", or "UMAP"
    label_type: str = "auto",           # "auto", "continuous", or "discrete"
    standardize: bool = False,          # standardize features before DR
    # t-SNE params
    tsne_perplexity: float = 30.0,
    tsne_learning_rate: str = "auto",
    # UMAP params
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    # plotting params
    figsize: Tuple[int, int] = (7, 6),
    s: int = 30,
    alpha: float = 0.9,
    cmap: str = "viridis",              # for continuous labels
    palette: Optional[list[str]] = None,# for discrete labels; list of color hex/tuples
    random_state: int = 42,
    title: Optional[str] = None,
    savepath: Optional[str] = None,
    show: bool = True,
) -> Dict[str, Any]:
    """
    Plot a 2D projection of patient embeddings with coloring by labels.

    Parameters
    ----------
    embeddings : DataFrame
        Shape (n_patients, n_features). Index are patient IDs.
    labels : Series
        Length n_patients (or subset). Index should match embeddings index.
        Can be numeric (continuous) or categorical/discrete.
    method : {"PCA", "TSNE", "UMAP"}
        Dimensionality reduction method to obtain 2D coordinates.
    label_type : {"auto", "continuous", "discrete"}
        How to interpret `labels`. "auto" tries to infer from dtype and cardinality.
    standardize : bool
        If True, standardize features before DR (recommended for PCA/UMAP).
    tsne_perplexity : float
        Perplexity for t-SNE (will be auto-clamped to valid range).
    tsne_learning_rate : {"auto", float}
        Learning rate for t-SNE.
    umap_n_neighbors : int
        UMAP n_neighbors (will be auto-clamped to < n_samples).
    umap_min_dist : float
        UMAP min_dist.
    figsize : tuple
        Figure size.
    s : int
        Marker size.
    alpha : float
        Marker alpha.
    cmap : str
        Matplotlib colormap name for continuous labels.
    palette : list
        Colors for discrete labels; if None, a categorical colormap is chosen.
    random_state : int
        Random seed for reproducibility (t-SNE/UMAP/PCA).
    title : str or None
        Plot title; if None, a default is constructed.
    savepath : str or None
        If provided, save the figure to this path.
    show : bool
        If True, display the figure.

    Returns
    -------
    out : dict with:
        - "embedding_2d": DataFrame with columns ["x", "y"] and original index
        - "labels": aligned labels (Series)
        - "fig", "ax": Matplotlib figure and axis
        - "palette": dict mapping category -> color (for discrete), else None
    """
    # Align indices and drop rows with missing features or labels
    labels_aligned = labels.reindex(embeddings.index)
    mask = labels_aligned.notna() & ~embeddings.isna().any(axis=1)
    if mask.sum() < len(embeddings):
        warnings.warn(f"Dropping {int((~mask).sum())} rows due to missing data in "
                      "embeddings or labels.")
    X = embeddings.loc[mask].to_numpy()
    y = labels_aligned.loc[mask]

    n, d = X.shape
    if n < 2:
        raise ValueError("Not enough rows after filtering to plot (need at least 2).")

    # Determine label type
    def infer_label_type(series: pd.Series) -> str:
        if pd.api.types.is_numeric_dtype(series):
            # Treat as continuous if sufficiently many unique values
            return "continuous" if series.nunique(dropna=True) > 10 else "discrete"
        else:
            return "discrete"

    kind = label_type.lower()
    if kind == "auto":
        kind = infer_label_type(y)
    elif kind not in {"continuous", "discrete"}:
        raise ValueError("label_type must be 'auto', 'continuous', or 'discrete'.")

    # Optional standardization
    X_proc = X
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(X)

    # Dimensionality reduction
    method_u = method.strip().upper()
    if method_u == "PCA":
        dr_model = PCA(n_components=2, random_state=random_state)
        XY = dr_model.fit_transform(X_proc)
        xlab, ylab = "PC1", "PC2"
    elif method_u == "TSNE":
        # t-SNE constraints: 5 <= perplexity < (n_samples - 1) / 3
        max_perp = max(5, (n - 1) // 3)
        perp = min(tsne_perplexity, max_perp)
        if perp < 5:
            perp = 5
            warnings.warn(f"Adjusted t-SNE perplexity to {perp} for n={n}.")
        dr_model = TSNE(
            n_components=2,
            perplexity=perp,
            learning_rate=tsne_learning_rate,
            init="pca",
            random_state=random_state,
            metric="euclidean",
        )
        XY = dr_model.fit_transform(X_proc)
        xlab, ylab = "t-SNE 1", "t-SNE 2"
    elif method_u == "UMAP":
        try:
            from umap import UMAP  # type: ignore
        except Exception as e:
            raise ImportError(
                "UMAP is not installed. Install with `pip install umap-learn`."
            ) from e
        nn = min(max(2, umap_n_neighbors), max(2, n - 1))
        if nn != umap_n_neighbors:
            warnings.warn(f"Adjusted UMAP n_neighbors to {nn} for n={n}.")
        dr_model = UMAP(
            n_components=2,
            n_neighbors=nn,
            min_dist=umap_min_dist,
            metric="euclidean",
            random_state=random_state,
        )
        XY = dr_model.fit_transform(X_proc)
        xlab, ylab = "UMAP 1", "UMAP 2"
    else:
        raise ValueError("method must be one of {'PCA', 'TSNE', 'UMAP'}.")

    proj = pd.DataFrame(XY, index=embeddings.index[mask], columns=["x", "y"])

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    palette_map = None
    if kind == "continuous":
        sc = ax.scatter(
            proj["x"], proj["y"],
            c=y.to_numpy(dtype=float),
            cmap=cmap,
            s=s, alpha=alpha,
            edgecolors="none"
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(y.name if y.name else "value")
    else:
        y_cat = y.astype("category")
        cats = list(y_cat.cat.categories)
        n_cat = len(cats)

        if palette is None:
            # choose a categorical colormap
            if n_cat <= 10:
                cm = mpl.cm.get_cmap("tab10", n_cat)
            elif n_cat <= 20:
                cm = mpl.cm.get_cmap("tab20", n_cat)
            else:
                # fallback: hsv with many distinct hues
                cm = mpl.cm.get_cmap("hsv", n_cat)
            colors = [mpl.colors.to_hex(cm(i)) for i in range(n_cat)]
        else:
            if len(palette) < n_cat:
                raise ValueError(f"Provided palette has {len(palette)} colors but "
                                 f"{n_cat} categories are present.")
            colors = palette[:n_cat]

        palette_map = {cat: col for cat, col in zip(cats, colors)}

        for cat in cats:
            m = (y_cat == cat).to_numpy()
            ax.scatter(
                proj.loc[m, "x"], proj.loc[m, "y"],
                c=palette_map[cat],
                s=s, alpha=alpha,
                edgecolors="none",
                label=str(cat),
            )

        ax.legend(
            title=(y.name if y.name else "group"),
            bbox_to_anchor=(1.02, 1), loc="upper left",
            borderaxespad=0., frameon=False
        )

    if title is None:
        title = f"{method_u} projection ({proj.shape[0]} patients, {embeddings.shape[1]} dims)"
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return {
        "embedding_2d": proj,
        "labels": y,
        "fig": fig,
        "ax": ax,
        "palette": palette_map,
    }


