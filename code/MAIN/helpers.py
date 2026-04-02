"""
Utilities for running multivariable and multivariate regression in SHCS cohort.

Functions exported
- add_healthy_flag
- label_by_percentiles
- plot_patient_embeddings
- TwoLayerNet
- get_device
- gen_meta_and_data
- get_gpu_memory
"""
import warnings
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Iterable, Union, Callable, Any, Dict, Hashable, Literal, Union, List, Sequence
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

__all__ = [
    "add_healthy_flag",
    "label_by_percentiles",
    "plot_patient_embeddings",
    "TwoLayerNet",
    "get_device",
    "gen_meta_and_data",
    "get_gpu_memory"
]

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

def gen_meta_and_data(data , meta , data_cols_drop , data_cols_meta_keep , idx_col) :
    # set meta index as string
    meta = meta.set_index(idx_col)
    meta.index = meta.index.astype(str)

    # set data index as string
    data = data.set_index(idx_col)
    data.index = data.index.astype(str)

    # filt to overlapping patients
    to_keep = list(set(data.index) & set(meta.index))
    meta = meta.loc[to_keep]
    data = data.loc[to_keep]

    # add meta cols from data to meta
    if len(data_cols_meta_keep) > 0 and type(data_cols_meta_keep) == list :
        for col in data_cols_meta_keep : 
            meta[col] = data[col]

    # drop meta cols from data
    data = data.drop(columns = data_cols_drop)

    return data , meta
    

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class TwoLayerNet(nn.Module):
    """
    A simple two-layer MLP:
    Input -> Linear -> BatchNorm1d -> ReLU -> Dropout(0.1) -> Linear -> Output

    - BatchNorm is applied to the hidden layer (typical usage).
    - Dropout p=0.1 as requested.
    - Expects inputs as (N, in_features). If higher-dimensional (e.g., images),
      it will be flattened automatically.
    """
    def __init__(self, in_features: int, hidden_size: int, num_classes: int, dropout_p: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.drop = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Flatten if input is not already (N, D)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.drop(x)
        x = self.fc2(x)
        return x

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



def label_by_percentiles(
    df,
    columns=None,                     # which columns to label; default = numeric cols
    lower_pct=0.25,                   # lower percentile (0.25 for Q1)
    upper_pct=0.75,                   # upper percentile (0.75 for Q3)
    labels=("lower", "middle", "upper"),
    include_bounds="middle",          # 'middle' | 'lower' | 'upper' | 'both'
    as_category=True,                 # return categorical dtype
    na_label=None,                    # set a label for NaNs; default leaves NaN as NaN
    append=False,                     # append label columns to df (with suffix) instead of returning separately
    suffix="_band",                   # suffix when append=True
    use_qcut=False                    # use pandas.qcut instead of manual thresholds
):
    """
    Vectorized labeling of each numeric column into lower/middle/upper bands based on percentiles.
    - lower_pct and upper_pct accept 0-1 or 0-100.
    - include_bounds determines where exact threshold values go:
        'middle': lower: x < qL; upper: x > qU; edges -> middle (default)
        'lower' : lower: x <= qL; upper: x > qU
        'upper' : lower: x < qL;  upper: x >= qU
        'both'  : lower: x <= qL; upper: x >= qU
    - If use_qcut=True, uses pd.qcut per column (handles quantiles directly), but can drop bins if edges duplicate.
    """
    lower_label, middle_label, upper_label = labels

    # Select columns
    if columns is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        cols = list(columns)
    if not cols:
        raise ValueError("No columns selected to label.")

    # Normalize percentiles
    def _norm(p):
        p = float(p)
        return p/100.0 if p > 1 else p

    lq = _norm(lower_pct)
    uq = _norm(upper_pct)
    if not (0 <= lq < uq <= 1):
        raise ValueError("lower_pct must be < upper_pct and both in [0,1] (or [0,100]).")

    X = df[cols]

    if use_qcut:
        # Use pandas qcut (one Series at a time). Simple but may drop bins if duplicates arise.
        # q boundaries: [0, lq, uq, 1]
        def cut_series(s):
            try:
                return pd.qcut(
                    s, q=[0, lq, uq, 1],
                    labels=[lower_label, middle_label, upper_label],
                    duplicates='drop'
                )
            except ValueError:
                # Fallback: if all values equal or not enough unique values, label as middle
                out = pd.Series(index=s.index, dtype='object')
                out[:] = middle_label
                out[s.isna()] = np.nan
                return out

        labels_df = X.apply(cut_series, axis=0)
    else:
        # Fast vectorized thresholds + comparisons
        quants = X.quantile([lq, uq])
        q_low = quants.loc[lq]
        q_up  = quants.loc[uq]

        if include_bounds in ("both", "lower"):
            lower_mask = X.le(q_low)
        else:
            lower_mask = X.lt(q_low)

        if include_bounds in ("both", "upper"):
            upper_mask = X.ge(q_up)
        else:
            upper_mask = X.gt(q_up)

        middle_mask = (~lower_mask) & (~upper_mask) & X.notna()

        # Build output labels
        labels_df = pd.DataFrame(np.nan, index=X.index, columns=cols, dtype=object)
        labels_df[lower_mask]  = lower_label
        labels_df[upper_mask]  = upper_label
        labels_df[middle_mask] = middle_label

    # Handle NaN label if requested
    if na_label is not None:
        labels_df = labels_df.fillna(na_label)

    # Optional categorical dtype with ordered categories
    if as_category:
        cat = pd.CategoricalDtype(categories=[lower_label, middle_label, upper_label], ordered=True)
        labels_df = labels_df.astype(cat)

    if append:
        return df.join(labels_df.add_suffix(suffix))
    return labels_df


def add_healthy_flag(
    df: pd.DataFrame,
    cols: Sequence[str] = ('CKD', 'phenotype', 'ANG', 'AMI', 'PRO', 'CEI', 'DVT', 'PUL', 'BYP', 'CEH', 'END', 'HTR', 'DMT2'),
    *,
    id_col: str = 'ID',
    out_col: str = 'Healthy',
    group: str = 'row',  # 'row' | 'patient_all' | 'patient_any'
    control_values: Iterable[str] = ('control',),
) -> pd.DataFrame:
    """
    Add a boolean column `out_col` marking rows/patients as Healthy if all values
    across `cols` are NA, zero, or in `control_values` (case-insensitive).

    Parameters
    ----------
    df : DataFrame
        Input DataFrame (modified in-place and also returned).
    cols : sequence of str
        Columns to check.
    id_col : str
        Patient ID column (used when group != 'row').
    out_col : str
        Name of the output boolean column to add.
    group : {'row', 'patient_all', 'patient_any'}
        - 'row': mark each row independently.
        - 'patient_all': mark all rows for an ID True only if every row for that ID is Healthy.
        - 'patient_any': mark rows for an ID True if any row for that ID is Healthy.
    control_values : iterable of str
        String tokens to treat as “control” (case-insensitive; e.g., ('control','ctrl')).

    Returns
    -------
    DataFrame with a new boolean column `out_col`.
    """
    # Validate columns
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if group in ('patient_all', 'patient_any') and id_col not in df.columns:
        raise ValueError(f"id_col '{id_col}' not found in DataFrame.")

    sub = df[list(cols)]

    # Allowed conditions per cell
    is_na = sub.isna()
    # Numeric zero (also captures string numerics like "0", "0.0")
    as_num = sub.apply(pd.to_numeric, errors='coerce')
    is_zero = as_num.eq(0)
    # Control tokens (case-insensitive, trims whitespace)
    control_set = {str(c).strip().lower() for c in control_values}
    is_control = sub.apply(lambda s: s.astype(str).str.strip().str.lower().isin(control_set))

    allowed = is_na | is_zero | is_control
    row_ok = allowed.all(axis=1)

    if group == 'row':
        result = row_ok
    elif group == 'patient_all':
        result = row_ok.groupby(df[id_col]).transform('all')
    elif group == 'patient_any':
        result = row_ok.groupby(df[id_col]).transform('any')
    else:
        raise ValueError("group must be one of {'row', 'patient_all', 'patient_any'}.")

    df[out_col] = result.astype(bool)*1
    return df

    
