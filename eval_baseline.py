import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import random
from pathlib import Path
import sys

# Import the geobleu evaluation functions
from geobleu_seq_eval import calc_geobleu_single, calc_dtw_single


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


def wide_to_trajectory_format(user_row, training_slots, prediction_slots):
    """Convert wide format user data to trajectory format for geobleu evaluation"""
    trajectory = []

    # Process training period (days 1-60)
    for i, slot in enumerate(training_slots):
        day = (i // 48) + 1  # Day 1-60
        timeslot = i % 48  # Timeslot 0-47

        loc_str = user_row[slot]
        loc = parse_location(loc_str)
        if loc:
            trajectory.append((day, timeslot, loc[0], loc[1]))

    # Process prediction period (days 61-75)
    for i, slot in enumerate(prediction_slots):
        day = (i // 48) + 61  # Day 61-75
        timeslot = i % 48  # Timeslot 0-47

        loc_str = user_row[slot]
        loc = parse_location(loc_str)
        if loc:
            trajectory.append((day, timeslot, loc[0], loc[1]))

    return trajectory


class BaselineEvaluator:
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

        # Identify users with complete data (users 11-100 in your setup)
        self.complete_users = []
        for idx, row in self.df.iterrows():
            uid = row["uid"]
            has_prediction = row[self.prediction_slots].notna().any()
            if has_prediction:
                self.complete_users.append(uid)

        print(f"Users with complete data for evaluation: {len(self.complete_users)}")

    def get_all_locations_training(self, exclude_user=None):
        """Get all training locations, optionally excluding one user"""
        all_locations = []

        for idx, row in self.df.iterrows():
            uid = row["uid"]
            if exclude_user and uid == exclude_user:
                continue

            if uid in self.complete_users:
                for slot in self.training_slots:
                    loc_str = row[slot]
                    if pd.notna(loc_str):
                        loc = parse_location(loc_str)
                        if loc:
                            all_locations.append(loc)

        return all_locations

    def get_user_locations_training(self, user_row):
        """Get all training locations for a specific user"""
        locations = []

        for slot in self.training_slots:
            loc_str = user_row[slot]
            if pd.notna(loc_str):
                loc = parse_location(loc_str)
                if loc:
                    locations.append(loc)

        return locations

    def predict_global_mean(self, target_user_id):
        """Predict using global mean (excluding target user)"""
        all_locations = self.get_all_locations_training(exclude_user=target_user_id)

        if not all_locations:
            return (100, 100)

        mean_x = sum(loc[0] for loc in all_locations) / len(all_locations)
        mean_y = sum(loc[1] for loc in all_locations) / len(all_locations)

        return (round(mean_x), round(mean_y))

    def predict_global_mode(self, target_user_id):
        """Predict using global mode (excluding target user)"""
        all_locations = self.get_all_locations_training(exclude_user=target_user_id)

        if not all_locations:
            return (100, 100)

        location_counts = Counter(all_locations)
        return location_counts.most_common(1)[0][0]

    def predict_per_user_mean(self, user_row):
        """Predict using per-user mean"""
        user_locations = self.get_user_locations_training(user_row)

        if not user_locations:
            return (100, 100)

        mean_x = sum(loc[0] for loc in user_locations) / len(user_locations)
        mean_y = sum(loc[1] for loc in user_locations) / len(user_locations)

        return (round(mean_x), round(mean_y))

    def predict_per_user_mode(self, user_row):
        """Predict using per-user mode"""
        user_locations = self.get_user_locations_training(user_row)

        if not user_locations:
            return (100, 100)

        location_counts = Counter(user_locations)
        return location_counts.most_common(1)[0][0]

    def predict_unigram_model(self, user_row, num_predictions):
        """Predict using unigram model"""
        user_locations = self.get_user_locations_training(user_row)

        if not user_locations:
            return [(100, 100)] * num_predictions

        predictions = []
        for _ in range(num_predictions):
            predictions.append(random.choice(user_locations))

        return predictions

    def predict_bigram_model(self, user_row, num_predictions, top_p=None):
        """Predict using bigram model"""
        # Build sequence from training data
        user_sequence = []
        for slot in self.training_slots:
            loc_str = user_row[slot]
            if pd.notna(loc_str):
                loc = parse_location(loc_str)
                if loc:
                    user_sequence.append(loc)

        if len(user_sequence) < 2:
            # Fallback to unigram
            return self.predict_unigram_model(user_row, num_predictions)

        # Build bigram model
        bigram_model = defaultdict(list)
        for i in range(len(user_sequence) - 1):
            current_loc = user_sequence[i]
            next_loc = user_sequence[i + 1]
            bigram_model[current_loc].append(next_loc)

        # Generate predictions
        predictions = []
        current_loc = user_sequence[-1]  # Start from last training location

        for _ in range(num_predictions):
            if current_loc in bigram_model:
                candidates = bigram_model[current_loc]

                if top_p is not None:
                    # Apply top_p sampling
                    candidate_counts = Counter(candidates)
                    sorted_candidates = sorted(
                        candidate_counts.items(), key=lambda x: x[1], reverse=True
                    )

                    total_count = sum(candidate_counts.values())
                    cumulative_prob = 0
                    filtered_candidates = []

                    for candidate, count in sorted_candidates:
                        prob = count / total_count
                        cumulative_prob += prob
                        filtered_candidates.extend([candidate] * count)

                        if cumulative_prob >= top_p:
                            break

                    if filtered_candidates:
                        next_loc = random.choice(filtered_candidates)
                    else:
                        next_loc = random.choice(candidates)
                else:
                    next_loc = random.choice(candidates)
            else:
                next_loc = random.choice(user_sequence)

            predictions.append(next_loc)
            current_loc = next_loc

        return predictions

    def create_prediction_trajectory(self, predictions, start_day=61):
        """Convert predictions to trajectory format"""
        trajectory = []

        for i, loc in enumerate(predictions):
            day = start_day + (i // 48)
            timeslot = i % 48
            trajectory.append((day, timeslot, loc[0], loc[1]))

        return trajectory

    def evaluate_method(self, method_name, method_func, num_test_users=None):
        """Evaluate a baseline method on complete users"""
        print(f"\nEvaluating {method_name}...")

        # Use subset for testing if specified
        test_users = (
            self.complete_users[:num_test_users]
            if num_test_users
            else self.complete_users
        )

        geobleu_scores = []
        dtw_scores = []

        for i, uid in enumerate(test_users):
            # Get user data
            user_row = self.df[self.df["uid"] == uid].iloc[0]

            # Get ground truth trajectory (days 61-75)
            ground_truth = []
            for j, slot in enumerate(self.prediction_slots):
                day = 61 + (j // 48)
                timeslot = j % 48
                loc_str = user_row[slot]
                loc = parse_location(loc_str)
                if loc:
                    ground_truth.append((day, timeslot, loc[0], loc[1]))

            if not ground_truth:
                continue

            # Generate predictions
            num_predictions = len(self.prediction_slots)
            predictions = method_func(user_row, uid, num_predictions)

            # Convert predictions to trajectory format
            if isinstance(predictions, tuple):  # Single location methods
                pred_locations = [predictions] * num_predictions
            else:  # Sequential methods
                pred_locations = predictions

            pred_trajectory = self.create_prediction_trajectory(pred_locations)

            # Evaluate
            try:
                geobleu_score = calc_geobleu_single(pred_trajectory, ground_truth)
                dtw_score = calc_dtw_single(pred_trajectory, ground_truth)

                geobleu_scores.append(geobleu_score)
                dtw_scores.append(dtw_score)

            except Exception as e:
                print(f"Error evaluating user {uid}: {e}")
                continue

            if (i + 1) % 10 == 0:
                print(f"  Evaluated {i + 1}/{len(test_users)} users...")

        avg_geobleu = np.mean(geobleu_scores) if geobleu_scores else 0
        avg_dtw = np.mean(dtw_scores) if dtw_scores else float("inf")

        return {
            "method": method_name,
            "geobleu": avg_geobleu,
            "dtw": avg_dtw,
            "num_users": len(geobleu_scores),
        }

    def run_all_evaluations(self, num_test_users=20):
        """Run all baseline evaluations"""
        print("=" * 60)
        print("EVALUATING BASELINE METHODS")
        print("=" * 60)

        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)

        results = []

        # Global Mean
        def global_mean_func(user_row, uid, num_pred):
            return self.predict_global_mean(uid)

        results.append(
            self.evaluate_method("Global Mean", global_mean_func, num_test_users)
        )

        # Global Mode
        def global_mode_func(user_row, uid, num_pred):
            return self.predict_global_mode(uid)

        results.append(
            self.evaluate_method("Global Mode", global_mode_func, num_test_users)
        )

        # Per-User Mean
        def per_user_mean_func(user_row, uid, num_pred):
            return self.predict_per_user_mean(user_row)

        results.append(
            self.evaluate_method("Per-User Mean", per_user_mean_func, num_test_users)
        )

        # Per-User Mode
        def per_user_mode_func(user_row, uid, num_pred):
            return self.predict_per_user_mode(user_row)

        results.append(
            self.evaluate_method("Per-User Mode", per_user_mode_func, num_test_users)
        )

        # Unigram Model
        def unigram_func(user_row, uid, num_pred):
            return self.predict_unigram_model(user_row, num_pred)

        results.append(
            self.evaluate_method("Unigram Model", unigram_func, num_test_users)
        )

        # Bigram Model
        # def bigram_func(user_row, uid, num_pred):
        #     return self.predict_bigram_model(user_row, num_pred)

        # results.append(
        #     self.evaluate_method("Bigram Model", bigram_func, num_test_users)
        # )

        # # Bigram Model (top_p=0.7)
        # def bigram_top_p_func(user_row, uid, num_pred):
        #     return self.predict_bigram_model(user_row, num_pred, top_p=0.7)

        # results.append(
        #     self.evaluate_method(
        #         "Bigram Model (top_p=0.7)", bigram_top_p_func, num_test_users
        #     )
        # )

        return results

    def print_results(self, results):
        """Print evaluation results in a nice table"""
        print("\n" + "=" * 80)
        print("BASELINE EVALUATION RESULTS")
        print("=" * 80)

        print(f"{'Method':<25} {'GEO-BLEU':<12} {'DTW':<12} {'Users':<8}")
        print("-" * 80)

        for result in results:
            method = result["method"]
            geobleu = result["geobleu"]
            dtw = result["dtw"]
            users = result["num_users"]

            print(f"{method:<25} {geobleu:<12.5f} {dtw:<12.2f} {users:<8}")

        print("-" * 80)

        # Best performers
        best_geobleu = max(results, key=lambda x: x["geobleu"])
        best_dtw = min(results, key=lambda x: x["dtw"])

        print(
            f"\nBest GEO-BLEU: {best_geobleu['method']} ({best_geobleu['geobleu']:.5f})"
        )
        print(f"Best DTW: {best_dtw['method']} ({best_dtw['dtw']:.2f})")


def main():
    # Check if geobleu module is available
    try:
        from geobleu_seq_eval import calc_geobleu_single, calc_dtw_single
    except ImportError:
        print("Error: geobleu_seq_eval.py not found!")
        print("Please make sure geobleu_seq_eval.py is in the same directory.")
        return

    # File paths
    input_file = "city_C_processed.csv"

    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found!")
        return

    # Initialize evaluator
    evaluator = BaselineEvaluator(input_file)

    # Run evaluations (use subset for faster testing)
    print("Running baseline evaluations on 20 users (for speed)...")
    print("Set num_test_users=None to evaluate all users")

    results = evaluator.run_all_evaluations(num_test_users=20)

    # Print results
    evaluator.print_results(results)

    print(f"\nEvaluation completed!")


if __name__ == "__main__":
    main()
