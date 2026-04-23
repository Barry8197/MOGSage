"""
Utilities for RAndom k-labELsets (RAKEL) algorithm

Functions exported
- LabelPowerset
- RAkEL
"""

import copy
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone

__all__ = [
    "LabelPowerset",
    "RAkEL"
]

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