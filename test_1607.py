import numpy as np
from sklearn.mixture import GaussianMixture


def blend_movements_with_gmm(top_5_users_movements, n_components=2):
    """
    Blend multiple movement sets using Gaussian Mixture Model

    Args:
        movement_sets: List of movement arrays from different users
        n_components: Number of Gaussian components for GMM

    Returns:
        Sampled movements from the fitted GMM
    """

    user_1 = [(1, 2), (3, 4), (5, 6)]  # Example data for user 1
    user_2 = [(7, 8), (9, 10), (11, 12)]  # Example data for user 2
    user_3 = [(13, 14), (15, 16), (17, 18)]  # Example data for user 3
    user_4 = [(19, 20), (21, 22), (23, 24)]  # Example data for user 4
    user_5 = [(25, 26), (27, 28), (29, 30)]  # Example data for user 5

    # input
    # top_user1: [(dx1_tag61, dy1_tag61) .. (dx48_tag61, dy48__tag61) ... (dx48_tag75, dy48__tag75)] -> 672 paare
    # top_user2: [(dx1_tag61, dy1_tag61) .. (dx48_tag61, dy48__tag61) ... (dx48_tag75, dy48__tag75)] -> 672 paare
    # ...
    # top_user5: [(dx1_tag61, dy1_tag61) .. (dx48_tag61, dy48__tag61) ... (dx48_tag75, dy48__tag75)] -> 672 paare

    # output
    # extension_vorschlag: [(dx1_tag61, dy1_tag61) .. (dx48_tag61, dy48__tag61) ... (dx48_tag75, dy48__tag75)] -> 672 paare

    # Fit GMM
    n_components = min(n_components, len(all_movements))
    gmm = GaussianMixture(
        n_components=n_components, covariance_type="full", random_state=42
    )
    gmm.fit(all_movements)

    max_length = max(len(moves) for moves in movement_sets)

    # Sample from GMM
    sampled_movements, _ = gmm.sample(max_length)

    return sampled_movements
