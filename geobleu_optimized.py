import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import gmean


def generate_ngrams(coords, n):
    """Generate n-grams from coordinate array using numpy"""
    coords = np.array(coords)

    # Should actually never occur
    if len(coords) < n:
        return np.empty((0, n, 2))

    # Use sliding window view for efficient n-gram generation
    return np.lib.stride_tricks.sliding_window_view(coords, (n, 2)).reshape(-1, n, 2)


def calculate_proximity_matrix(pred_ngrams, ref_ngrams, beta=0.5):
    """Calculate proximity matrix between predicted and reference n-grams"""
    if len(pred_ngrams) == 0 or len(ref_ngrams) == 0:
        return np.zeros((len(pred_ngrams), len(ref_ngrams)))

    # Flatten n-grams to calculate distances between entire sequences
    pred_flat = pred_ngrams.reshape(len(pred_ngrams), -1)
    ref_flat = ref_ngrams.reshape(len(ref_ngrams), -1)

    # Calculate Euclidean distances between all pairs
    distances = cdist(pred_flat, ref_flat, metric="euclidean")

    # Convert to proximity using exponential decay
    return np.exp(-beta * distances)


def greedy_matching(proximity_matrix):
    """Find matching using greedy algorithm (matches original implementation)"""
    if proximity_matrix.size == 0:
        return 0.0

    n_pred, n_ref = proximity_matrix.shape
    if n_pred == 0:
        return 0.0

    # Create list of all possible matches with their scores
    matches = []
    for i in range(n_pred):
        for j in range(n_ref):
            matches.append((i, j, proximity_matrix[i, j]))

    # Sort by proximity (highest first)
    matches.sort(key=lambda x: x[2], reverse=True)

    used_pred = set()
    used_ref = set()
    total_proximity = 0.0
    match_count = 0

    # Greedy selection
    for pred_idx, ref_idx, proximity in matches:
        if pred_idx not in used_pred and ref_idx not in used_ref:
            used_pred.add(pred_idx)
            used_ref.add(ref_idx)
            total_proximity += proximity
            match_count += 1

    return total_proximity / n_pred  # Average over predicted n-grams


def calculate_precision_n(pred_coords, ref_coords, n, beta=0.5):
    """Calculate precision for n-grams of size n"""
    pred_ngrams = generate_ngrams(pred_coords, n)
    ref_ngrams = generate_ngrams(ref_coords, n)

    if len(pred_ngrams) == 0:
        return 0.0

    proximity_matrix = calculate_proximity_matrix(pred_ngrams, ref_ngrams, beta)
    return greedy_matching(proximity_matrix)


def calculate_brevity_penalty(pred_length, ref_length):
    """Calculate brevity penalty for length differences"""
    if pred_length >= ref_length:
        return 1.0
    return np.exp(1.0 - ref_length / pred_length)


def geobleu_score(pred_coords, ref_coords, max_n=5, beta=0.5):
    """
    Calculate GEO-BLEU score between predicted and reference coordinates

    Args:
        pred_coords: List/array of (x, y) coordinate pairs for prediction
        ref_coords: List/array of (x, y) coordinate pairs for reference
        max_n: Maximum n-gram size to consider
        beta: Spatial decay parameter

    Returns:
        float: GEO-BLEU score between 0 and 1
    """
    pred_coords = np.array(pred_coords)
    ref_coords = np.array(ref_coords)

    if len(pred_coords) == 0 or len(ref_coords) == 0:
        return 0.0

    # Calculate precision for each n-gram size
    precisions = []
    max_possible_n = min(max_n, len(pred_coords), len(ref_coords))

    for n in range(1, max_possible_n + 1):
        precision = calculate_precision_n(pred_coords, ref_coords, n, beta)
        precisions.append(precision)

    if not precisions:
        return 0.0

    # Calculate geometric mean of precisions
    geo_mean = gmean(precisions)

    # Apply brevity penalty
    brevity = calculate_brevity_penalty(len(pred_coords), len(ref_coords))

    return brevity * geo_mean


def geobleu_from_trajectories(pred_traj, ref_traj, max_n=5, beta=0.5):
    """
    Calculate GEO-BLEU from trajectory data with (day, time, x, y) format

    Args:
        pred_traj: List of (day, time, x, y) tuples for prediction
        ref_traj: List of (day, time, x, y) tuples for reference

    Returns:
        float: GEO-BLEU score
    """
    pred_df = pd.DataFrame(pred_traj, columns=["day", "time", "x", "y"])
    ref_df = pd.DataFrame(ref_traj, columns=["day", "time", "x", "y"])

    pred_df = pred_df.sort_values(["day", "time"])
    ref_df = ref_df.sort_values(["day", "time"])

    pred_coords = pred_df[["x", "y"]].values
    ref_coords = ref_df[["x", "y"]].values

    return geobleu_score(pred_coords, ref_coords, max_n, beta)


def geobleu_by_day(pred_traj, ref_traj, max_n=5, beta=0.5):
    """
    Calculate GEO-BLEU score day by day and return average
    (This matches the original implementation approach)
    """
    # Convert to DataFrames
    pred_df = pd.DataFrame(pred_traj, columns=["day", "time", "x", "y"])
    ref_df = pd.DataFrame(ref_traj, columns=["day", "time", "x", "y"])

    # Get all days present in reference
    days = sorted(ref_df["day"].unique())

    daily_scores = []
    for day in days:
        pred_day = pred_df[pred_df["day"] == day][["x", "y"]].values
        ref_day = ref_df[ref_df["day"] == day][["x", "y"]].values

        if len(pred_day) > 0 and len(ref_day) > 0:
            score = geobleu_score(pred_day, ref_day, max_n, beta)
            daily_scores.append(score)

    return np.mean(daily_scores) if daily_scores else 0.0


# # Example usage and testing
# if __name__ == "__main__":
#     # Example 1: Simple coordinate comparison
#     predicted = [(0, 0), (1, 1), (2, 2), (3, 3)]
#     reference = [(0, 0), (1, 0), (2, 1), (3, 2)]

#     score = geobleu_score(predicted, reference, max_n=3, beta=0.5)
#     print(f"Simple GEO-BLEU score: {score:.4f}")

#     # Example 2: Trajectory data with day/time
#     pred_trajectory = [
#         (61, 0, 10, 20),
#         (61, 1, 11, 21),
#         (61, 2, 12, 22),
#         (62, 0, 15, 25),
#         (62, 1, 16, 26),
#         (62, 2, 17, 27),
#     ]

#     ref_trajectory = [
#         (61, 0, 10, 20),
#         (61, 1, 10, 20),
#         (61, 2, 11, 21),
#         (62, 0, 15, 24),
#         (62, 1, 15, 25),
#         (62, 2, 16, 26),
#     ]

#     score_traj = geobleu_from_trajectories(pred_trajectory, ref_trajectory)
#     print(f"Trajectory GEO-BLEU score: {score_traj:.4f}")

#     score_daily = geobleu_by_day(pred_trajectory, ref_trajectory)
#     print(f"Day-by-day GEO-BLEU score: {score_daily:.4f}")

#     # Example 3: Perfect match
#     perfect_pred = [(0, 0), (1, 1), (2, 2)]
#     perfect_ref = [(0, 0), (1, 1), (2, 2)]
#     perfect_score = geobleu_score(perfect_pred, perfect_ref)
#     print(f"Perfect match score: {perfect_score:.4f}")

#     # Example 4: Demonstrate impact of beta parameter
#     slightly_off = [(0, 0), (1, 1), (2, 2)]
#     reference_coords = [(0, 0), (1, 0), (2, 1)]

#     for beta in [0.1, 0.5, 1.0, 2.0]:
#         score = geobleu_score(slightly_off, reference_coords, beta=beta)
#         print(f"Beta {beta}: {score:.4f}")
