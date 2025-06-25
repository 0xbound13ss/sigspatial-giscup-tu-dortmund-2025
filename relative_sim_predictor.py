import pandas as pd
import numpy as np
from collections import defaultdict
import random
from pathlib import Path
from geobleu_seq_eval import calc_geobleu_single


def parse_location(loc_str):
    """Parse location string like '97,152' into (x, y) tuple"""
    if pd.isna(loc_str) or loc_str == "":
        return None
    try:
        x, y = map(int, loc_str.split(","))
        return (x, y)
    except:
        return None


def location_to_string(x, y):
    """Convert (x, y) back to location string"""
    return f"{x},{y}"


def extract_trajectory_sequence(user_row, slots):
    """Extract clean trajectory sequence from user data"""
    trajectory = []

    for slot in slots:
        loc_str = user_row[slot]
        loc = parse_location(loc_str)
        if loc:
            trajectory.append(loc)

    return trajectory


def compute_relative_movements(trajectory):
    """Compute relative movement vectors from trajectory"""
    if len(trajectory) < 2:
        return []

    relative_moves = []
    for i in range(1, len(trajectory)):
        prev_loc = trajectory[i - 1]
        curr_loc = trajectory[i]

        # Relative movement: (dx, dy)
        dx = curr_loc[0] - prev_loc[0]
        dy = curr_loc[1] - prev_loc[1]
        relative_moves.append((dx, dy))

    return relative_moves


def apply_relative_movements(start_location, relative_moves):
    """Apply relative movements to generate trajectory from start location"""
    trajectory = [start_location]
    current_loc = start_location

    for dx, dy in relative_moves:
        next_loc = (current_loc[0] + dx, current_loc[1] + dy)

        # Clamp to grid boundaries (0-199)
        next_loc = (max(0, min(199, next_loc[0])), max(0, min(199, next_loc[1])))

        trajectory.append(next_loc)
        current_loc = next_loc

    return trajectory


def trajectory_to_geobleu_format(trajectory, start_day=61):
    """Convert trajectory to geobleu format (day, timeslot, x, y)"""
    geobleu_traj = []

    for i, (x, y) in enumerate(trajectory):
        day = start_day + (i // 48)
        timeslot = i % 48
        geobleu_traj.append((day, timeslot, x, y))

    return geobleu_traj


class RelativeMovementPredictor:
    def __init__(self, data_file):
        """Initialize with processed data file"""
        print("Loading processed data...")
        self.df = pd.read_csv(data_file)

        # Define time periods
        self.training_slots = [f"t_{i}" for i in range(0, 60 * 48)]  # Days 1-60
        self.prediction_slots = [
            f"t_{i}" for i in range(60 * 48, 75 * 48)
        ]  # Days 61-75

        print(f"Loaded {len(self.df)} users")

        # Identify user categories
        self.users_to_predict = []  # First 10 users (no prediction data)
        self.candidate_pool = []  # Remaining users (with complete data)

        for idx, row in self.df.iterrows():
            uid = row["uid"]
            has_prediction = row[self.prediction_slots].notna().any()

            if has_prediction:
                self.candidate_pool.append(uid)
            else:
                self.users_to_predict.append(uid)

        print(f"Users to predict: {len(self.users_to_predict)}")
        print(f"Candidate pool: {len(self.candidate_pool)}")

        # Pre-compute relative movements for candidate pool
        print("Pre-computing relative movements for candidate pool...")
        self.candidate_relative_moves = {}
        self.candidate_extensions = {}

        for uid in self.candidate_pool:
            user_row = self.df[self.df["uid"] == uid].iloc[0]

            # Training trajectory (days 1-60)
            training_traj = extract_trajectory_sequence(user_row, self.training_slots)

            # Prediction trajectory (days 61-75)
            prediction_traj = extract_trajectory_sequence(
                user_row, self.prediction_slots
            )

            if len(training_traj) >= 2 and len(prediction_traj) >= 1:
                # Relative movements for training period
                training_moves = compute_relative_movements(training_traj)

                # Extension moves (from last training location to prediction)
                if training_traj and prediction_traj:
                    last_training = training_traj[-1]
                    # Create extended trajectory: training + prediction
                    full_traj = training_traj + prediction_traj
                    # Get relative moves for the extension part
                    extension_moves = compute_relative_movements(full_traj)[
                        len(training_moves) :
                    ]

                    self.candidate_relative_moves[uid] = training_moves
                    self.candidate_extensions[uid] = extension_moves

        print(f"Successfully processed {len(self.candidate_relative_moves)} candidates")

    def compute_similarity(self, target_moves, candidate_moves, max_length=None):
        """Compute similarity between two relative movement sequences using GeoBleu"""
        if not target_moves or not candidate_moves:
            return 0.0

        # Truncate to same length for comparison
        if max_length:
            target_moves = target_moves[:max_length]
            candidate_moves = candidate_moves[:max_length]

        min_len = min(len(target_moves), len(candidate_moves))
        if min_len == 0:
            return 0.0

        target_moves = target_moves[:min_len]
        candidate_moves = candidate_moves[:min_len]

        # Convert to trajectories starting from origin for comparison
        origin = (100, 100)  # Neutral starting point

        target_traj = apply_relative_movements(origin, target_moves)
        candidate_traj = apply_relative_movements(origin, candidate_moves)

        # Convert to geobleu format
        target_geobleu = trajectory_to_geobleu_format(target_traj, start_day=1)
        candidate_geobleu = trajectory_to_geobleu_format(candidate_traj, start_day=1)

        try:
            similarity = calc_geobleu_single(target_geobleu, candidate_geobleu)
            return similarity
        except:
            return 0.0

    def find_best_match(self, target_uid, comparison_length=500):
        """Find the best matching candidate for a target user"""
        target_row = self.df[self.df["uid"] == target_uid].iloc[0]
        target_traj = extract_trajectory_sequence(target_row, self.training_slots)

        if len(target_traj) < 2:
            print(f"Warning: Target user {target_uid} has insufficient training data")
            return None, 0.0

        target_moves = compute_relative_movements(target_traj)

        print(f"\nFinding best match for user {target_uid}...")
        print(f"Target has {len(target_moves)} relative movements")

        similarities = []

        for candidate_uid in self.candidate_relative_moves:
            candidate_moves = self.candidate_relative_moves[candidate_uid]

            # Compute similarity using a fixed comparison length
            similarity = self.compute_similarity(
                target_moves, candidate_moves, comparison_length
            )
            similarities.append((candidate_uid, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Show top 5 matches
        print(f"Top 5 matches for user {target_uid}:")
        for i, (uid, sim) in enumerate(similarities[:5]):
            print(f"  {i+1}. User {uid}: {sim:.5f}")

        best_match = similarities[0]
        return best_match[0], best_match[1]

    def predict_user_trajectory(self, target_uid):
        """Predict trajectory for a target user using best match"""
        # Find best matching candidate
        best_candidate, similarity = self.find_best_match(target_uid)

        if best_candidate is None:
            print(f"No suitable match found for user {target_uid}")
            return None

        print(f"Best match: User {best_candidate} (similarity: {similarity:.5f})")

        # Get target user's last training location
        target_row = self.df[self.df["uid"] == target_uid].iloc[0]
        target_traj = extract_trajectory_sequence(target_row, self.training_slots)

        if not target_traj:
            print(f"No training data for user {target_uid}")
            return None

        last_training_loc = target_traj[-1]

        # Get extension moves from best candidate
        extension_moves = self.candidate_extensions[best_candidate]

        print(
            f"Applying {len(extension_moves)} extension moves from user {best_candidate}"
        )

        # Apply extension moves to predict trajectory
        predicted_trajectory = apply_relative_movements(
            last_training_loc, extension_moves
        )

        return predicted_trajectory, best_candidate, similarity

    def predict_all_users(self):
        """Predict trajectories for all users in prediction set"""
        print("=" * 60)
        print("PREDICTING TRAJECTORIES USING RELATIVE MOVEMENT MATCHING")
        print("=" * 60)

        predictions = self.df.copy()
        prediction_info = {}

        for target_uid in self.users_to_predict:
            print(f"\n--- Predicting for User {target_uid} ---")

            result = self.predict_user_trajectory(target_uid)

            if result is None:
                print(f"Failed to predict for user {target_uid}")
                continue

            predicted_traj, best_match, similarity = result

            # Store prediction info
            prediction_info[target_uid] = {
                "best_match": best_match,
                "similarity": similarity,
                "trajectory_length": len(predicted_traj),
            }

            # Fill prediction slots in dataframe
            user_idx = self.df[self.df["uid"] == target_uid].index[0]

            for i, (x, y) in enumerate(predicted_traj):
                if i < len(self.prediction_slots):
                    slot = self.prediction_slots[i]
                    predictions.at[user_idx, slot] = location_to_string(x, y)

        return predictions, prediction_info

    def save_predictions(self, predictions, prediction_info, output_file):
        """Save predictions and metadata"""
        # Save main predictions
        predictions.to_csv(output_file, index=False)

        # Save prediction metadata
        info_file = output_file.parent / f"{output_file.stem}_info.txt"
        with open(info_file, "w") as f:
            f.write("RELATIVE MOVEMENT PREDICTION RESULTS\n")
            f.write("=" * 50 + "\n\n")

            for target_uid, info in prediction_info.items():
                f.write(f"User {target_uid}:\n")
                f.write(f"  Best match: User {info['best_match']}\n")
                f.write(f"  Similarity: {info['similarity']:.5f}\n")
                f.write(f"  Trajectory length: {info['trajectory_length']}\n\n")

        print(f"\nPredictions saved to: {output_file}")
        print(f"Metadata saved to: {info_file}")


def main():
    # Check dependencies
    try:
        from geobleu_seq_eval import calc_geobleu_single
    except ImportError:
        print("Error: geobleu_seq_eval.py not found!")
        print("Please make sure geobleu_seq_eval.py is in the same directory.")
        return

    # File paths
    input_file = "city_C_processed.csv"
    output_file = Path("relative_movement_predictions.csv")

    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found!")
        return

    # Initialize predictor
    predictor = RelativeMovementPredictor(input_file)

    # Generate predictions
    predictions, prediction_info = predictor.predict_all_users()

    # Save results
    predictor.save_predictions(predictions, prediction_info, output_file)

    # Print summary
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)

    for target_uid, info in prediction_info.items():
        print(
            f"User {target_uid}: matched with User {info['best_match']} "
            f"(similarity: {info['similarity']:.5f})"
        )

    avg_similarity = np.mean([info["similarity"] for info in prediction_info.values()])
    print(f"\nAverage similarity: {avg_similarity:.5f}")


if __name__ == "__main__":
    main()
