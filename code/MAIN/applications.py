"""
Utilities for RAndom k-labELsets (RAKEL) algorithm

Functions exported
- LabelPowerset
- RAkEL
- plot_model_comparison
- plot_km_by_group
- plot_overall_km
- eval_survival_model
- visualise_feature_importance
- make_survival_targets_from_death_df
- plot_per_class_confusion_matrices
"""

import copy
import numpy as np
from time import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from palettable.wesanderson import Darjeeling2_5

from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone

from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index

__all__ = [
    "LabelPowerset",
    "RAkEL",
    "plot_model_comparison",
    "plot_km_by_group",
    "plot_overall_km",
    "eval_survival_model",
    "visualise_feature_importance",
    "make_survival_targets_from_death_df"
    "plot_per_class_confusion_matrices"
]

def plot_per_class_confusion_matrices(
    metrics_dict,
    class_labels=None,
    normalize="true",          # "true" | "pred" | "all" | None
    show_counts=False,         # if True, annotate "prop\n(count)"
    annotate_fmt=".2f",
    figsize_per_plot=(4.2, 3.6),
    cbar=True,                 # one shared colorbar for the whole figure
    max_cols=3,                # max columns per row of subplots
    cbar_width=0.35,           # width (in “subplot-width units”) reserved for the colorbar
):
    """
    Plot one-vs-rest 2x2 confusion matrices per class (optionally normalized),
    arranged in a grid of subplots with up to `max_cols` columns, with a single
    shared colorbar placed in a dedicated axis at the far right.

    Expects:
      metrics_dict['confusion'] with keys 'tp','fp','fn','tn' as arrays (n_classes,)
    """
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib as mpl

    conf = metrics_dict["confusion"]
    tp = np.asarray(conf["tp"], dtype=float)
    fp = np.asarray(conf["fp"], dtype=float)
    fn = np.asarray(conf["fn"], dtype=float)
    tn = np.asarray(conf["tn"], dtype=float)

    n_classes = tp.shape[0]
    if class_labels is None:
        class_labels = [f"Class {i}" for i in range(n_classes)]
    if len(class_labels) != n_classes:
        raise ValueError(f"Expected {n_classes} class_labels, got {len(class_labels)}")

    # Colormap (Wes Anderson Zissou)
    from palettable.wesanderson import Zissou_5_r as _Z
    cmap = _Z.mpl_colormap

    n_cols = min(max_cols, n_classes)
    n_rows = int(math.ceil(n_classes / n_cols))

    # Figure size (add extra width for the colorbar column when enabled)
    fig_w = figsize_per_plot[0] * n_cols + (figsize_per_plot[0] * cbar_width if cbar else 0.0)
    fig_h = figsize_per_plot[1] * n_rows
    fig = plt.figure(figsize=(fig_w, fig_h))

    # GridSpec: last column reserved for colorbar (spans all rows)
    gs = fig.add_gridspec(
        nrows=n_rows,
        ncols=n_cols + (1 if cbar else 0),
        width_ratios=([1] * n_cols + ([cbar_width] if cbar else [])),
        wspace=0.35,
        hspace=0.55,
    )

    axes = np.empty((n_rows, n_cols), dtype=object)
    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c] = fig.add_subplot(gs[r, c])

    cax = fig.add_subplot(gs[:, -1]) if cbar else None  # dedicated colorbar axis

    # Shared color scale
    vmin = 0.0
    if normalize is not None:
        vmax = 1.0
        cbar_label = "Proportion"
    else:
        vmax = float(np.max(np.array([tp, fp, fn, tn])))
        cbar_label = "Count"

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    for i, label in enumerate(class_labels):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]

        cm_counts = np.array([[tp[i], fn[i]],
                              [fp[i], tn[i]]], dtype=float)

        # ---- normalize ----
        if normalize is None:
            cm = cm_counts
        elif normalize == "true":
            denom = cm_counts.sum(axis=1, keepdims=True)
            cm = np.divide(cm_counts, denom, out=np.zeros_like(cm_counts), where=denom != 0)
        elif normalize == "pred":
            denom = cm_counts.sum(axis=0, keepdims=True)
            cm = np.divide(cm_counts, denom, out=np.zeros_like(cm_counts), where=denom != 0)
        elif normalize == "all":
            denom = cm_counts.sum()
            cm = cm_counts / denom if denom != 0 else np.zeros_like(cm_counts)
        else:
            raise ValueError('normalize must be one of: "true", "pred", "all", or None')

        # ---- annotations ----
        if show_counts and normalize is not None:
            ann = np.empty_like(cm, dtype=object)
            for rr in range(2):
                for cc in range(2):
                    ann[rr, cc] = f"{cm[rr, cc]:{annotate_fmt}}\n({int(cm_counts[rr, cc])})"
            annot = ann
            fmt = ""
        else:
            annot = True
            fmt = annotate_fmt if normalize is not None else "d"

        sns.heatmap(
            cm,
            ax=ax,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            cbar=False,          # disable per-axes colorbar
            vmin=vmin,
            vmax=vmax,
            square=True,
            linewidths=1,
            linecolor="white",
            xticklabels=["Pred +", "Pred -"],
            yticklabels=["True +", "True -"],
        )
        ax.set_title(f"{label} (one-vs-rest)\nnormalize={normalize}")
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Hide unused subplot cells
    for j in range(n_classes, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r, c].axis("off")

    # Shared colorbar
    if cbar:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label(cbar_label)

    return fig, axes

def make_survival_targets_from_death_df(
    X,
    death_df: pd.DataFrame,
    id_col: str = "ID",
    death_col: str = "DEATH_DATE",
    sample_col: str = "SAMPLE_DATE",
    censor_date=None,
    censor_strategy: str = "global_max",
    return_linpred: bool = True,
):
    """
    Derive survival durations, event indicators, and an optional linear predictor
    from a death dataframe aligned with embedding matrix X.

    Parameters
    ----------
    X : array-like, shape (N, D)
        Embedding matrix. Must have the same number of rows as death_df.
    death_df : pd.DataFrame
        Must contain columns for sample date (time zero) and death date.
    id_col : str
        Column name for subject IDs (informational only).
    death_col : str
        Column name for death dates (NaT if unknown / censored).
    sample_col : str
        Column name for baseline sample dates (time zero).
    censor_date : str or pd.Timestamp, optional
        Administrative censoring date. Required when censor_strategy="given".
    censor_strategy : {"global_max", "given"}
        "global_max" uses the latest observed date in the dataset;
        "given" uses the provided censor_date.
    return_linpred : bool
        If True, also return a standardised linear predictor derived from X.

    Returns
    -------
    durations : ndarray, shape (N,)  — follow-up time in days (>= 1e-3).
    events    : ndarray, shape (N,)  — 1 if death observed, 0 if censored.
    linpred   : ndarray, shape (N,)  — optional risk score derived from X.
    """
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]

    if len(death_df) != N:
        raise ValueError(
            f"Row mismatch: X has {N} rows but death_df has {len(death_df)} rows. "
            "Align them to the same order first."
        )

    d = death_df.copy()
    d[death_col]  = pd.to_datetime(d[death_col],  errors="coerce")
    d[sample_col] = pd.to_datetime(d[sample_col], errors="coerce")

    if d[sample_col].isna().any():
        bad = d.index[d[sample_col].isna()][:5].tolist()
        raise ValueError(f"Found NaT in {sample_col} (baseline). Example rows: {bad}")

    # Censor date
    if censor_strategy == "given":
        if censor_date is None:
            raise ValueError("censor_strategy='given' requires censor_date.")
        censor_date = pd.to_datetime(censor_date)
    elif censor_strategy == "global_max":
        max_death  = d[death_col].max(skipna=True)
        max_sample = d[sample_col].max(skipna=True)
        censor_date = max(dt for dt in [max_death, max_sample] if pd.notna(dt))
    else:
        raise ValueError("censor_strategy must be 'global_max' or 'given'.")

    # Durations and events
    baseline = d[sample_col]
    death_dt = d[death_col]
    events   = (death_dt.notna() & (death_dt <= censor_date)).astype(np.int32).to_numpy()
    exit_date = death_dt.where((death_dt.notna() & (death_dt <= censor_date)), censor_date)
    durations = ((exit_date - baseline) / np.timedelta64(1, "D")).to_numpy(dtype=np.float64)
    durations = np.clip(durations, 1e-3, None)

    if not return_linpred:
        return durations, events

    # Deterministic linear predictor from X
    Z = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    p = min(8, Z.shape[1])
    w = np.linspace(0.6, 0.2, p)
    linpred = Z[:, :p] @ w
    if p >= 3:
        linpred = linpred + 0.25 * Z[:, 0] * Z[:, 1] - 0.15 * (Z[:, 2] ** 2)
    linpred = (linpred - linpred.mean()) / (linpred.std() + 1e-8)

    return durations, events, linpred

def plot_km_by_group(
    results,
    df_test,
    durations,
    events,
    groups,
    model_name="Cox PH",
    group_label="Group",
    times=None,
    order_by="mean_risk",
    min_group_size=10,
    ci_show=False,
    title=None,
    palette=None,
):
    """
    Plot Kaplan–Meier curves per group using a pre-fitted survival model for
    risk scoring. Pairwise log-rank tests are run between all group pairs and
    an overall significance annotation is added to the plot.

    Parameters
    ----------
    results : list[dict] or dict
        Output of eval_survival_model calls. Each dict must contain at minimum
        {"Model": str, "model": fitted lifelines model}.
    df_test : pd.DataFrame
        Test covariates (may include "time"/"event" columns — ignored for prediction).
    durations, events : array-like
        Observed follow-up times and event indicators for df_test rows.
    groups : array-like
        Group label per row in df_test (e.g. cluster id, phenotype label).
    model_name : str
        Key to select a model from results (matches "Model" field).
    group_label : str
        Human-readable name for the grouping variable used in the legend and
        axis labels (e.g. "Cluster", "Subtype", "Label").
    times : array-like or None
        Optional time grid to restrict the x-axis (xlim).
    order_by : {"mean_risk", "median_risk", None}
        How to order groups in the legend.
    min_group_size : int
        Groups smaller than this are excluded.
    ci_show : bool
        Show 95 % confidence bands on KM curves.
    title : str or None
        Plot title. Defaults to a sensible automatic title.
    palette : list or None
        Matplotlib colour list. Falls back to the current colour cycle.

    Returns
    -------
    risk_scores : ndarray, shape (N,)
        Predicted risk scores for all test samples.
    group_info : list of tuples
        (group_label, n, mean_risk, median_risk) sorted as plotted.
    pairwise_results : list of dicts
        Log-rank test results for each pair of groups.
    """
    # ── Resolve model ─────────────────────────────────────────────────────────
    if isinstance(results, dict):
        res = results
    else:
        matches = [r for r in results if r.get("Model") == model_name]
        if not matches:
            available = [r.get("Model") for r in results]
            raise ValueError(f"Model '{model_name}' not found. Available: {available}")
        res = matches[0]

    mdl = res.get("model")
    if mdl is None:
        raise ValueError("Selected results entry has no fitted model under key 'model'.")

    durations = np.asarray(durations, dtype=float)
    events    = np.asarray(events,    dtype=int)
    groups    = np.asarray(groups)

    if not (len(df_test) == len(durations) == len(events) == len(groups)):
        raise ValueError("df_test, durations, events, and groups must all have the same length.")

    # ── Risk scores ───────────────────────────────────────────────────────────
    X = df_test.drop(columns=[c for c in ["time", "event"] if c in df_test.columns], errors="ignore")
    if hasattr(mdl, "predict_partial_hazard"):
        risk_scores = np.asarray(mdl.predict_partial_hazard(X)).reshape(-1)
    elif "risk_scores" in res:
        risk_scores = np.asarray(res["risk_scores"]).reshape(-1)
    else:
        raise ValueError(
            "Model has no predict_partial_hazard method and no 'risk_scores' key in results."
        )

    # ── Filter and sort groups ────────────────────────────────────────────────
    unique     = np.unique(groups)
    group_info = []
    for g in unique:
        mask = groups == g
        n    = int(mask.sum())
        if n < min_group_size:
            continue
        rs = risk_scores[mask]
        group_info.append((g, n, float(rs.mean()), float(np.median(rs))))

    if not group_info:
        raise ValueError("No groups remain after applying min_group_size filter.")

    sort_key = {"mean_risk": 2, "median_risk": 3}.get(order_by)
    if sort_key is not None:
        group_info.sort(key=lambda x: x[sort_key])

    # ── Pairwise log-rank tests ───────────────────────────────────────────────
    from lifelines.statistics import logrank_test

    pairwise_results = []
    group_ids = [gi[0] for gi in group_info]

    for i in range(len(group_ids)):
        for j in range(i + 1, len(group_ids)):
            g_i, g_j = group_ids[i], group_ids[j]
            m_i = groups == g_i
            m_j = groups == g_j
            lr  = logrank_test(
                durations[m_i], durations[m_j],
                event_observed_A=events[m_i],
                event_observed_B=events[m_j],
            )
            pairwise_results.append({
                "group_A":    g_i,
                "group_B":    g_j,
                "test_stat":  lr.test_statistic,
                "p_value":    lr.p_value,
                "significant": lr.p_value < 0.05,
            })

    # Determine overall significance (any pair significant after Bonferroni)
    n_tests       = len(pairwise_results)
    p_values      = [r["p_value"] for r in pairwise_results]
    min_p         = min(p_values) if p_values else 1.0
    bonferroni_p  = min(min_p * n_tests, 1.0)   # Bonferroni corrected
    any_sig       = bonferroni_p < 0.05

    # ── Plot ──────────────────────────────────────────────────────────────────
    colors = palette if palette else plt.rcParams["axes.prop_cycle"].by_key()["color"]
    kmf    = KaplanMeierFitter()

    fig, ax = plt.subplots(figsize=(9, 6))

    for idx, (g, n, mean_risk, med_risk) in enumerate(group_info):
        mask  = groups == g
        color = colors[idx % len(colors)]
        label = f"{group_label} {g}  (n={n}, mean risk={mean_risk:.2f})"
        kmf.fit(durations[mask], events[mask], label=label)
        kmf.plot_survival_function(ax=ax, ci_show=ci_show, linewidth=2, color=color)

    if title is None:
        title = f"Kaplan–Meier by {group_label} (risk scored by: {model_name})"
    ax.set_title(title, pad=14)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival probability")

    if times is not None:
        ax.set_xlim(np.min(times), np.max(times))

    # ── Significance annotation ───────────────────────────────────────────────
    if pairwise_results:
        def _p_stars(p):
            if p < 0.001: return "***"
            if p < 0.01:  return "**"
            if p < 0.05:  return "*"
            return "ns"

        # Build compact annotation string for pairwise comparisons
        pair_lines = [
            f"{group_label} {r['group_A']} vs {group_label} {r['group_B']}: "
            f"p={r['p_value']:.3g} {_p_stars(r['p_value'])}"
            for r in pairwise_results
        ]
        sig_summary = (
            f"Overall (Bonferroni): p={bonferroni_p:.3g} {_p_stars(bonferroni_p)}"
            + (" — groups differ significantly" if any_sig else " — no significant difference")
        )
        annotation = "\n".join(pair_lines + ["", sig_summary])

        ax.text(
            0.98, 0.97,
            annotation,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor="grey",
                alpha=0.85,
            ),
        )

    plt.tight_layout()
    plt.show()

    # Print pairwise table
    print("\nPairwise log-rank tests:")
    print(f"{'Group A':<12} {'Group B':<12} {'Test stat':>10} {'p-value':>10} {'Sig':>6}")
    print("-" * 55)
    for r in pairwise_results:
        print(
            f"{str(r['group_A']):<12} {str(r['group_B']):<12} "
            f"{r['test_stat']:>10.4f} {r['p_value']:>10.4g} "
            f"{'*' if r['significant'] else '':>6}"
        )
    print(f"\nBonferroni-corrected minimum p: {bonferroni_p:.4g}  "
          f"({'significant' if any_sig else 'not significant'} at α=0.05)")

    return risk_scores, group_info, pairwise_results, fig


def visualise_feature_importance(
    importances,
    mod_dim,
    top_n: int = 10,
    title: str | None = None,
    xlabel: str = "Mean |Conductance|",
    figsize: tuple[int, int] = (12, 8),
    context: str = "talk",
    style: str = "whitegrid",
    min_alpha: float = 0.25,
    max_alpha: float = 1.0,
    value_fmt: str = "{:.3f}",
    value_fontsize: int = 10,
    value_offset: tuple[float, float] = (5, 0),
    despine: bool = True,
    tight_layout: bool = True,
    show: bool = True,
    ax=None,
    # --- modality colouring options ---
    modality_labels=None,              # e.g. ["omics", "imaging", "text"]
    palette=Darjeeling2_5.mpl_colors,   # 5 colours
    legend_title: str = "Modality",
):
    """
    Plot top-N features by variance (as written) and colour bars by modality.

    Modalities are defined by mod_dim, e.g. [766, 1029, 345] meaning:
      modality 1: features [0, 766)
      modality 2: features [766, 766+1029)
      modality 3: features [766+1029, 766+1029+345)

    Parameters
    ----------
    importances : pd.DataFrame or array-like
        Feature scores with shape (n_samples, n_features).
        If DataFrame: columns used as feature labels (and assumed ordered to match mod_dim).
    mod_dim : list[int]
        Number of features per modality, in the same order as concatenation.
    top_n : int
        Number of top features to display.
    modality_labels : list[str] or None
        Optional names for modalities; if None uses "Mod 1", "Mod 2", ...
    palette : list
        List of colours (e.g., Darjeeling2_5.mpl_colors). If #modalities > len(palette),
        colours cycle.
    """

    def _feature_index_to_modality(idx: int, mod_dim_list) -> int:
        """Return modality id in {0,1,2,...} for a feature index."""
        cum = np.cumsum([0] + list(mod_dim_list))  # length M+1
        # Find m s.t. cum[m] <= idx < cum[m+1]
        # np.searchsorted returns insertion point; subtract 1 to get bin index
        m = int(np.searchsorted(cum, idx, side="right") - 1)
        return m

    # Coerce to DataFrame for consistent handling
    if isinstance(importances, pd.DataFrame):
        imp_df = importances.copy()
    else:
        imp_df = pd.DataFrame(importances)

    n_features = imp_df.shape[1]
    if sum(mod_dim) != n_features:
        raise ValueError(
            f"sum(mod_dim)={sum(mod_dim)} but importances has n_features={n_features}. "
            "These must match (assuming modalities concatenated in-order)."
        )

    n_mods = len(mod_dim)
    if modality_labels is None:
        modality_labels = [f"Mod {i+1}" for i in range(n_mods)]
    if len(modality_labels) != n_mods:
        raise ValueError("modality_labels must have the same length as mod_dim.")

    # Compute global importance: variance per feature (matches your current code)
    feat_imp = imp_df.var().sort_values(ascending=False).head(top_n)

    # Build plotting DataFrame
    df_plot = feat_imp.sort_values(ascending=True).reset_index()
    df_plot.columns = ["Feature", "Importance"]

    # Determine original feature index (position in imp_df columns)
    # - If columns are not unique, get_loc can return slice/array; handle by using enumerated mapping.
    col_to_pos = {col: i for i, col in enumerate(imp_df.columns)}
    df_plot["FeatureIndex"] = df_plot["Feature"].map(col_to_pos)

    if df_plot["FeatureIndex"].isna().any():
        # Fallback: if mapping failed (e.g., duplicate column names), use slower but reliable approach
        # by matching on both name and rank; but simplest is to require unique columns.
        raise ValueError(
            "Could not map feature names back to column positions. "
            "Ensure imp_df.columns are unique or pass a DataFrame with unique column names."
        )

    # Assign modality id/label and colour
    df_plot["ModalityId"] = df_plot["FeatureIndex"].apply(lambda i: _feature_index_to_modality(int(i), mod_dim))
    df_plot["Modality"] = df_plot["ModalityId"].apply(lambda m: modality_labels[m])
    df_plot["Color"] = df_plot["ModalityId"].apply(lambda m: palette[m % len(palette)])

    # Style
    sns.set_theme(style=style, context=context)

    # Figure / axis
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Draw bars with per-bar colours
    bars = ax.barh(df_plot["Feature"], df_plot["Importance"], color=df_plot["Color"].tolist())

    # Fade lower-importance features by alpha
    if len(df_plot) > 0 and df_plot["Importance"].max() > 0:
        alphas = df_plot["Importance"] / df_plot["Importance"].max()
    else:
        alphas = pd.Series([1.0] * len(df_plot))

    alphas = min_alpha + (max_alpha - min_alpha) * alphas.clip(0, 1)

    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(float(alpha))

    # Labels and title
    if title is None:
        title = f"Top {top_n} Features by Mean Absolute Layer Conductance"
    ax.set_title(title, pad=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.grid(False)

    # Value labels
    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            value_fmt.format(width),
            (width, bar.get_y() + bar.get_height() / 2),
            ha="left",
            va="center",
            fontsize=value_fontsize,
            xytext=value_offset,
            textcoords="offset points",
        )

    # Legend (one entry per modality present in the plotted top_n)
    present = df_plot[["Modality", "Color"]].drop_duplicates().sort_values("Modality")
    handles = [
        plt.Line2D([0], [0], color=row["Color"], lw=8)
        for _, row in present.iterrows()
    ]
    ax.legend(handles, present["Modality"].tolist(), title=legend_title, loc="best", frameon=True)

    if despine:
        sns.despine()

    if tight_layout:
        fig.tight_layout()

    if show:
        plt.show()

    return fig, ax, df_plot


def eval_survival_model(model, df_tr, df_te, name, pred_type="partial_hazard"):
    t0 = time()
    model.fit(df_tr, duration_col="time", event_col="event")
    elapsed = time() - t0

    if pred_type == "partial_hazard":
        risk_scores = model.predict_partial_hazard(df_te).values.reshape(-1)
        c_index = concordance_index(
            df_te["time"].values,
            -risk_scores,
            df_te["event"].values
        )
    elif pred_type == "median":
        pred_median = model.predict_median(df_te).values.reshape(-1)
        risk_scores = -pred_median
        c_index = concordance_index(
            df_te["time"].values,
            pred_median,
            df_te["event"].values
        )
    else:
        raise ValueError("pred_type must be 'partial_hazard' or 'median'.")

    return {
        "Model": name,
        "Concordance Index": c_index,
        "Time (s)": round(elapsed, 3),
        "risk_scores": risk_scores,
        "model": model,
    }

def plot_overall_km(durations, events, title="Overall Kaplan–Meier Curve", COL='b'):
    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=events, label="All samples")

    fig, ax = plt.subplots(figsize=(8, 5))
    kmf.plot_survival_function(
        ax=ax,
        ci_show=True,
        color=COL,
        ci_alpha=0.20
    )
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    sns.despine()
    plt.tight_layout()
    plt.show()

def plot_model_comparison(df_results, palette=Darjeeling2_5.mpl_colors):
    df_plot = df_results.reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.barplot(
        data=df_plot,
        x="Model",
        y="Concordance Index",
        hue="Model",
        palette=palette[:len(df_plot)],
        legend=False,
        ax=axes[0]
    )
    axes[0].set_title("Model Comparison — Concordance Index")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(
        data=df_plot,
        x="Model",
        y="Time (s)",
        hue="Model",
        palette=palette[:len(df_plot)],
        legend=False,
        ax=axes[1]
    )
    axes[1].set_title("Model Comparison — Runtime")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=20)

    sns.despine()
    plt.tight_layout()
    plt.show()

class LabelPowerset:
    def __init__(self, base_classifier):
        self.clf = base_classifier
        self.le  = LabelEncoder()

    def fit(self, X, y_sub):
        combos = [''.join(map(str, row)) for row in y_sub]
        y_encoded = self.le.fit_transform(combos)
        self.k = y_sub.shape[1]
        self.clf.fit(X, y_encoded)
        return self

    def predict(self, X):
        y_pred_encoded = self.clf.predict(X)
        # inverse_transform can raise if index is out of range — clamp it
        n_classes = len(self.le.classes_)
        y_pred_encoded = np.clip(y_pred_encoded, 0, n_classes - 1)
        y_pred_str = self.le.inverse_transform(y_pred_encoded)

        result = np.zeros((len(y_pred_str), self.k), dtype=int)
        for i, s in enumerate(y_pred_str):
            if len(s) == self.k:
                result[i] = [int(c) for c in s]
        return result


class RAkEL:
    def __init__(self, k=3, n_models=10, base_clf=None, mode='overlapping',
                 threshold=0.5, random_state=42):
        self.k            = k
        self.n_models     = n_models
        self.base_clf     = base_clf
        self.mode         = mode
        self.threshold    = threshold
        self.random_state = random_state

    def _sample_label_subsets(self, n_labels):
        rng = np.random.RandomState(self.random_state)
        if self.mode == 'disjoint':
            subsets = []
            for _ in range(self.n_models):
                perm = rng.permutation(n_labels)
                n_full = n_labels // self.k
                for i in range(min(n_full, self.n_models - len(subsets))):
                    subsets.append(sorted(perm[i*self.k:(i+1)*self.k].tolist()))
                    if len(subsets) == self.n_models:
                        return subsets
            return subsets[:self.n_models]
        else:
            return [
                sorted(rng.choice(n_labels, self.k, replace=False).tolist())
                for _ in range(self.n_models)
            ]

    def fit(self, X, y):
        n_labels = y.shape[1]
        self.n_labels_    = n_labels
        self.classifiers_ = []
        self.label_sets_  = self._sample_label_subsets(n_labels)

        for label_idx in self.label_sets_:
            y_sub = y[:, label_idx]
            # Use clone() — safer than deepcopy for sklearn estimators
            lp = LabelPowerset(clone(self.base_clf))
            lp.fit(X, y_sub)
            self.classifiers_.append(lp)
        return self

    def predict(self, X):
        vote_sum   = np.zeros((X.shape[0], self.n_labels_), dtype=float)
        vote_count = np.zeros(self.n_labels_, dtype=float)

        for lp, label_idx in zip(self.classifiers_, self.label_sets_):
            preds = lp.predict(X)
            vote_sum[:, label_idx]  += preds
            vote_count[label_idx]   += 1

        vote_count = np.where(vote_count == 0, 1, vote_count)
        vote_ratio = vote_sum / vote_count[np.newaxis, :]
        return (vote_ratio >= self.threshold).astype(int)

    def predict_proba(self, X):
        vote_sum   = np.zeros((X.shape[0], self.n_labels_), dtype=float)
        vote_count = np.zeros(self.n_labels_, dtype=float)

        for lp, label_idx in zip(self.classifiers_, self.label_sets_):
            preds = lp.predict(X)
            vote_sum[:, label_idx]  += preds
            vote_count[label_idx]   += 1

        vote_count = np.where(vote_count == 0, 1, vote_count)
        return vote_sum / vote_count[np.newaxis, :]