"""
AHP Analysis Module: Analytic Hierarchy Process for crop selection.
Replicates MATLAB-based AHP pairwise comparison and ranking.
"""
import numpy as np
import pandas as pd


CRITERIA = [
    "Yield Potential",
    "Water Requirement",
    "Market Price",
    "Rainfall Suitability",
    "Temperature Suitability",
    "Soil pH Suitability",
]

PAIRWISE_MATRIX = np.array([
    [1,   3,   2,   4,   5,   6],
    [1/3, 1,   1/2, 2,   3,   4],
    [1/2, 2,   1,   3,   4,   5],
    [1/4, 1/2, 1/3, 1,   2,   3],
    [1/5, 1/3, 1/4, 1/2, 1,   2],
    [1/6, 1/4, 1/5, 1/3, 1/2, 1],
])

RI_VALUES = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def compute_weights(matrix):
    """Compute AHP weights using eigenvalue method."""
    n = matrix.shape[0]
    eigvals, eigvecs = np.linalg.eig(matrix)
    max_idx = np.argmax(eigvals.real)
    weights = eigvecs[:, max_idx].real
    weights = weights / weights.sum()
    return weights


def compute_consistency(matrix, weights):
    """Compute consistency ratio."""
    n = matrix.shape[0]
    aw = matrix @ weights
    lambda_max = (aw / weights).mean()
    ci = (lambda_max - n) / (n - 1)
    ri = RI_VALUES.get(n, 1.12)
    cr = ci / ri if ri > 0 else 0
    return {
        "lambda_max": round(lambda_max.real, 4),
        "ci": round(ci.real, 4),
        "ri": ri,
        "cr": round(cr.real, 4),
        "consistent": cr < 0.1,
    }


def run_ahp(custom_matrix=None):
    """Run full AHP analysis. Returns weights, consistency, and rankings."""
    matrix = custom_matrix if custom_matrix is not None else PAIRWISE_MATRIX
    criteria = CRITERIA
    n = len(criteria)

    if matrix.shape != (n, n):
        return None, f"Matrix must be {n}x{n}"

    weights = compute_weights(matrix)
    consistency = compute_consistency(matrix, weights)

    ranking = pd.DataFrame({
        "Criterion": criteria,
        "Weight": [round(w, 4) for w in weights],
        "Percentage": [f"{round(w * 100, 1)}%" for w in weights],
    }).sort_values("Weight", ascending=False).reset_index(drop=True)
    ranking.index = range(1, len(ranking) + 1)

    return {
        "criteria": criteria,
        "pairwise_matrix": matrix,
        "weights": weights,
        "ranking": ranking,
        "consistency": consistency,
    }, None
