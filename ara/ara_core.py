import numpy as np


class ARA:
    """
    Adaptive Regime Aggregation (ARA)

    Combines multiple informational subsystems into a coordinated score.
    """

    def __init__(self, G, I):
        """
        Parameters
        ----------
        G : numpy array
            Data-driven subsystem matrix
        I : numpy array
            Intuition-based subsystem matrix
        """
        self.G = np.array(G)
        self.I = np.array(I)

        if self.G.shape != self.I.shape:
            raise ValueError("Subsystem matrices must have the same shape")

    def solve(self, alpha=0.618):
        """
        Compute ARA aggregated score.

        Parameters
        ----------
        alpha : float
            coordination regime weight

        Returns
        -------
        scores : numpy array
        """

        X = alpha * self.G + (1 - alpha) * self.I

        scores = X.mean(axis=0)

        return scores

    def rank(self, alpha=0.618):
        """
        Return ranking of projects
        """

        scores = self.solve(alpha)

        ranking = np.argsort(scores)[::-1]

        return ranking, scores
