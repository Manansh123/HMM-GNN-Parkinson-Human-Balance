"""
Continuous Gaussian HMM for Improvement 1.

Continuous observations:
    [angle, distance]

No fixed discretization is used.

Each trial is fitted with a Gaussian HMM.
Embedding:
    state stationary occupancies
    + maximum angle
    + maximum distance
"""

from __future__ import annotations

import numpy as np
from hmmlearn.hmm import GaussianHMM


class ContinuousGaussianHMM:

    def __init__(
        self,
        n_states=11,
        n_iter=100,
        tol=1e-4,
        random_state=42,
    ):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.model = None

    def _prepare_sequences(self, sequences):

        clean = []

        for seq in sequences:

            X = np.asarray(seq, dtype=np.float64)

            if X.ndim != 2 or X.shape[1] != 2:
                raise ValueError(
                    "Each sequence must have shape (n_samples, 2)."
                )

            finite = np.all(
                np.isfinite(X),
                axis=1,
            )

            X = X[finite]

            if len(X) >= 20:
                clean.append(X)

        if not clean:
            raise ValueError(
                "No usable continuous sequences."
            )

        return clean

    def fit(self, sequences):

        clean = self._prepare_sequences(sequences)

        X = np.concatenate(
            clean,
            axis=0,
        )

        lengths = [
            len(seq)
            for seq in clean
        ]

        # --------------------------------------------------
        # Stable initialization
        # --------------------------------------------------

        rng = np.random.default_rng(
            self.random_state
        )

        # Choose initial means from actual observations.
        replace = len(X) < self.n_states

        indices = rng.choice(
            len(X),
            size=self.n_states,
            replace=replace,
        )

        means_init = X[indices].copy()

        # Global variance gives every state a valid
        # non-zero covariance at initialization.
        global_var = np.var(
            X,
            axis=0,
        )

        global_var = np.maximum(
            global_var,
            1e-6,
        )

        covars_init = np.tile(
            global_var,
            (self.n_states, 1),
        )

        # Start with a valid transition matrix.
        transmat_init = np.full(
            (
                self.n_states,
                self.n_states,
            ),
            1.0 / self.n_states,
        )

        # Uniform initial state distribution.
        startprob_init = np.full(
            self.n_states,
            1.0 / self.n_states,
        )

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,

            # We provide stable initialization ourselves.
            init_params="",

            params="stmc",
            verbose=False,

            startprob_prior=1.0,
            transmat_prior=1.0,
        )

        self.model.startprob_ = startprob_init
        self.model.transmat_ = transmat_init
        self.model.means_ = means_init
        self.model.covars_ = covars_init

        self.model.fit(
            X,
            lengths,
        )

        # --------------------------------------------------
        # Safety checks
        # --------------------------------------------------

        if not np.all(
            np.isfinite(self.model.means_)
        ):
            raise ValueError(
                "HMM produced non-finite means."
            )

        if not np.all(
            np.isfinite(self.model.covars_)
        ):
            raise ValueError(
                "HMM produced non-finite covariances."
            )

        if not np.all(
            np.isfinite(self.model.transmat_)
        ):
            raise ValueError(
                "HMM produced non-finite transition matrix."
            )

        # Normalize transition matrix safely.
        row_sums = self.model.transmat_.sum(
            axis=1,
            keepdims=True,
        )

        row_sums = np.maximum(
            row_sums,
            1e-12,
        )

        self.model.transmat_ = (
            self.model.transmat_
            / row_sums
        )

        return self

    def stationary_distribution(
        self,
        tol=1e-10,
        max_iter=10000,
    ):

        if self.model is None:
            raise RuntimeError(
                "Model has not been fitted."
            )

        A = np.asarray(
            self.model.transmat_,
            dtype=np.float64,
        )

        # Safety normalization.
        row_sums = A.sum(
            axis=1,
            keepdims=True,
        )

        row_sums = np.maximum(
            row_sums,
            1e-12,
        )

        A = A / row_sums

        pi = np.full(
            self.n_states,
            1.0 / self.n_states,
        )

        for _ in range(max_iter):

            new_pi = pi @ A

            if np.max(
                np.abs(new_pi - pi)
            ) < tol:

                pi = new_pi
                break

            pi = new_pi

        pi = np.asarray(
            pi,
            dtype=np.float64,
        )

        pi = np.clip(
            pi,
            0.0,
            None,
        )

        total = pi.sum()

        if not np.isfinite(total) or total <= 0:
            return np.full(
                self.n_states,
                1.0 / self.n_states,
            )

        return pi / total

    def embedding(
        self,
        max_angle,
        max_distance,
    ):

        occupancy = (
            self.stationary_distribution()
        )

        embedding = np.concatenate(
            [
                occupancy,
                [
                    float(max_angle),
                    float(max_distance),
                ],
            ]
        )

        if not np.all(
            np.isfinite(embedding)
        ):
            raise ValueError(
                "Non-finite values in HMM embedding."
            )

        return embedding

    def score(self, sequence):

        if self.model is None:
            raise RuntimeError(
                "Model has not been fitted."
            )

        X = np.asarray(
            sequence,
            dtype=np.float64,
        )

        return float(
            self.model.score(X)
        )