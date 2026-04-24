"""
Utilities for RAndom k-labELsets (RAKEL) algorithm

Functions exported
- LabelPowerset
- RAkEL
- plot_model_comparison
- plot_risk_stratified_km
- plot_overall_km
- eval_survival_model
- visualise_feature_importance
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
    "plot_risk_stratified_km",
    "plot_overall_km",
    "eval_survival_model",
    "visualise_feature_importance"
]

def visualise_feature_importance(
    importances,
    top_n: int = 10,
    title: str | None = None,
    xlabel: str = "Mean |Conductance|",
    palette_color_index: int = 1,
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
):
    """
    Plot top-N features by mean absolute importance (e.g., layer conductance).

    Parameters
    ----------
    importances : pd.DataFrame or array-like
        Feature importances with features along columns. Typically shape (n_samples, n_features).
        If a DataFrame is provided, column names are used as feature labels.
    top_n : int
        Number of top features to display.
    title : str or None
        Plot title. If None, a default title is used.
    xlabel : str
        X-axis label.
    palette_color_index : int
        Index into Darjeeling2_5.mpl_colors for bar colour.
    figsize : (int, int)
        Figure size if a new figure is created.
    context, style : str
        Seaborn theme settings.
    min_alpha, max_alpha : float
        Alpha range to fade lower-importance bars.
    value_fmt : str
        Format string for value labels, e.g. "{:.3f}".
    value_fontsize : int
        Font size for value labels.
    value_offset : (float, float)
        Offset in points for value labels relative to the bar end.
    despine : bool
        If True, call sns.despine().
    tight_layout : bool
        If True, call plt.tight_layout().
    show : bool
        If True, call plt.show().
    ax : matplotlib.axes.Axes or None
        If provided, draw onto this axis; otherwise create a new figure/axis.

    Returns
    -------
    fig, ax, df_plot
        Matplotlib figure/axis and the plotting DataFrame (Feature, Importance).
    """
    # Coerce to DataFrame for consistent handling
    if isinstance(importances, pd.DataFrame):
        imp_df = importances
    else:
        imp_df = pd.DataFrame(importances)

    # Compute global importance: mean absolute importance per feature
    feat_imp = (
        imp_df.abs()
        .mean(axis=0)
        .sort_values(ascending=False)
        .head(top_n)
    )

    # Put into plotting DataFrame (ascending for barh)
    df_plot = feat_imp.sort_values(ascending=True).reset_index()
    df_plot.columns = ["Feature", "Importance"]

    # Style
    sns.set_theme(style=style, context=context)
    base_color = Darjeeling2_5.mpl_colors[palette_color_index]

    # Figure / axis
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Draw bars
    bars = ax.barh(df_plot["Feature"], df_plot["Importance"], color=base_color)

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

    # Add value labels
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

def plot_risk_stratified_km(durations, events, risk_scores, title, COL_1 = 'b', COL_2 = 'y'):
    risk_scores = np.asarray(risk_scores).reshape(-1)
    threshold = np.median(risk_scores)
    high_risk = risk_scores >= threshold
    low_risk  = ~high_risk

    kmf_low = KaplanMeierFitter()
    kmf_high = KaplanMeierFitter()

    fig, ax = plt.subplots(figsize=(8, 5))

    kmf_low.fit(
        durations[low_risk],
        event_observed=events[low_risk],
        label="Low risk"
    )
    kmf_high.fit(
        durations[high_risk],
        event_observed=events[high_risk],
        label="High risk"
    )

    kmf_low.plot_survival_function(
        ax=ax,
        ci_show=True,
        color=COL_1,
        ci_alpha=0.18
    )
    kmf_high.plot_survival_function(
        ax=ax,
        ci_show=True,
        color=COL_2,
        ci_alpha=0.18
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