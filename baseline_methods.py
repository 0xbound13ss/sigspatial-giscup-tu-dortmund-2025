import pandas as pd
import numpy as np
from collections import Counter
import random
from pathlib import Path


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


class BaselineMethods:
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
        print(f"Training period: {len(self.training_slots)} slots")
        print(f"Prediction period: {len(self.prediction_slots)} slots")

        # Identify users with/without prediction data
        self.users_no_prediction = []
        self.users_with_prediction = []

        for idx, row in self.df.iterrows():
            uid = row["uid"]
            has_prediction = row[self.prediction_slots].notna().any()

            if has_prediction:
                self.users_with_prediction.append(uid)
            else:
                self.users_no_prediction.append(uid)

        print(f"Users without prediction data: {len(self.users_no_prediction)}")
        print(f"Users with prediction data: {len(self.users_with_prediction)}")

    def get_all_locations_training(self):
        """Get all non-null locations from training period across all users"""
        all_locations = []

        for idx, row in self.df.iterrows():
            for slot in self.training_slots:
                loc_str = row[slot]
                if pd.notna(loc_str):
                    loc = parse_location(loc_str)
                    if loc:
                        all_locations.append(loc)

        return all_locations

    def get_user_locations_training(self, user_row):
        """Get all non-null locations from training period for a specific user"""
        locations = []

        for slot in self.training_slots:
            loc_str = user_row[slot]
            if pd.notna(loc_str):
                loc = parse_location(loc_str)
                if loc:
                    locations.append(loc)

        return locations

    def global_mean(self):
        """Calculate global mean of all training locations"""
        print("\nCalculating Global Mean...")

        all_locations = self.get_all_locations_training()

        if not all_locations:
            return (100, 100)  # Default fallback

        mean_x = sum(loc[0] for loc in all_locations) / len(all_locations)
        mean_y = sum(loc[1] for loc in all_locations) / len(all_locations)

        global_mean_loc = (round(mean_x), round(mean_y))
        print(f"Global mean location: {global_mean_loc}")

        return global_mean_loc

    def global_mode(self):
        """Calculate global mode (most frequent location) of all training locations"""
        print("\nCalculating Global Mode...")

        all_locations = self.get_all_locations_training()

        if not all_locations:
            return (100, 100)  # Default fallback

        location_counts = Counter(all_locations)
        global_mode_loc = location_counts.most_common(1)[0][0]

        print(
            f"Global mode location: {global_mode_loc} (appeared {location_counts[global_mode_loc]} times)"
        )

        return global_mode_loc

    def predict_global_method(self, global_location, method_name):
        """Predict using global method (mean or mode)"""
        print(f"\nPredicting with {method_name}...")

        predictions = self.df.copy()
        global_loc_str = location_to_string(global_location[0], global_location[1])

        # Only predict for users without existing prediction data
        for idx, row in predictions.iterrows():
            uid = row["uid"]

            if uid in self.users_no_prediction:
                # Fill prediction period with global location
                for slot in self.prediction_slots:
                    predictions.at[idx, slot] = global_loc_str

        return predictions

    def predict_per_user_mean(self):
        """Predict using per-user mean"""
        print("\nPredicting with Per-User Mean...")

        predictions = self.df.copy()

        for idx, row in predictions.iterrows():
            uid = row["uid"]

            if uid in self.users_no_prediction:
                # Calculate user's mean location from training period
                user_locations = self.get_user_locations_training(row)

                if user_locations:
                    mean_x = sum(loc[0] for loc in user_locations) / len(user_locations)
                    mean_y = sum(loc[1] for loc in user_locations) / len(user_locations)
                    user_mean_loc = location_to_string(round(mean_x), round(mean_y))
                else:
                    user_mean_loc = "100,100"  # Fallback

                # Fill prediction period with user's mean location
                for slot in self.prediction_slots:
                    predictions.at[idx, slot] = user_mean_loc

        return predictions

    def predict_per_user_mode(self):
        """Predict using per-user mode"""
        print("\nPredicting with Per-User Mode...")

        predictions = self.df.copy()

        for idx, row in predictions.iterrows():
            uid = row["uid"]

            if uid in self.users_no_prediction:
                # Calculate user's mode location from training period
                user_locations = self.get_user_locations_training(row)

                if user_locations:
                    location_counts = Counter(user_locations)
                    user_mode_loc = location_counts.most_common(1)[0][0]
                    user_mode_str = location_to_string(
                        user_mode_loc[0], user_mode_loc[1]
                    )
                else:
                    user_mode_str = "100,100"  # Fallback

                # Fill prediction period with user's mode location
                for slot in self.prediction_slots:
                    predictions.at[idx, slot] = user_mode_str

        return predictions

    def predict_unigram_model(self):
        """Predict using unigram model (sample from training distribution)"""
        print("\nPredicting with Unigram Model...")

        predictions = self.df.copy()

        for idx, row in predictions.iterrows():
            uid = row["uid"]

            if uid in self.users_no_prediction:
                # Get user's location distribution from training period
                user_locations = self.get_user_locations_training(row)

                if user_locations:
                    # Fill prediction period by sampling from user's location distribution
                    for slot in self.prediction_slots:
                        sampled_loc = random.choice(user_locations)
                        sampled_str = location_to_string(sampled_loc[0], sampled_loc[1])
                        predictions.at[idx, slot] = sampled_str
                else:
                    # Fallback
                    for slot in self.prediction_slots:
                        predictions.at[idx, slot] = "100,100"

        return predictions

    # def predict_bigram_model(self, top_p=None):
    #     """Predict using bigram model"""
    #     method_name = (
    #         "Bigram Model" if top_p is None else f"Bigram Model (top_p={top_p})"
    #     )
    #     print(f"\nPredicting with {method_name}...")

    #     predictions = self.df.copy()

    #     for idx, row in predictions.iterrows():
    #         uid = row["uid"]

    #         if uid in self.users_no_prediction:
    #             # Build bigram model from user's training sequence
    #             user_sequence = []
    #             for slot in self.training_slots:
    #                 loc_str = row[slot]
    #                 if pd.notna(loc_str):
    #                     loc = parse_location(loc_str)
    #                     if loc:
    #                         user_sequence.append(loc)

    #             if len(user_sequence) < 2:
    #                 # Fallback to unigram if insufficient data
    #                 for slot in self.prediction_slots:
    #                     if user_sequence:
    #                         sampled_loc = random.choice(user_sequence)
    #                         predictions.at[idx, slot] = location_to_string(
    #                             sampled_loc[0], sampled_loc[1]
    #                         )
    #                     else:
    #                         predictions.at[idx, slot] = "100,100"
    #             else:
    #                 # Build bigram transitions
    #                 bigram_model = defaultdict(list)
    #                 for i in range(len(user_sequence) - 1):
    #                     current_loc = user_sequence[i]
    #                     next_loc = user_sequence[i + 1]
    #                     bigram_model[current_loc].append(next_loc)

    #                 # Generate predictions
    #                 current_loc = user_sequence[-1]  # Start from last training location

    #                 for slot in self.prediction_slots:
    #                     if current_loc in bigram_model:
    #                         candidates = bigram_model[current_loc]

    #                         if top_p is not None:
    #                             # Apply top_p sampling
    #                             candidate_counts = Counter(candidates)
    #                             sorted_candidates = sorted(
    #                                 candidate_counts.items(),
    #                                 key=lambda x: x[1],
    #                                 reverse=True,
    #                             )

    #                             # Calculate cumulative probabilities
    #                             total_count = sum(candidate_counts.values())
    #                             cumulative_prob = 0
    #                             filtered_candidates = []

    #                             for candidate, count in sorted_candidates:
    #                                 prob = count / total_count
    #                                 cumulative_prob += prob
    #                                 filtered_candidates.extend([candidate] * count)

    #                                 if cumulative_prob >= top_p:
    #                                     break

    #                             if filtered_candidates:
    #                                 next_loc = random.choice(filtered_candidates)
    #                             else:
    #                                 next_loc = random.choice(candidates)
    #                         else:
    #                             # Standard sampling
    #                             next_loc = random.choice(candidates)
    #                     else:
    #                         # Fallback to random location from user's vocabulary
    #                         next_loc = random.choice(user_sequence)

    #                     predictions.at[idx, slot] = location_to_string(
    #                         next_loc[0], next_loc[1]
    #                     )
    #                     current_loc = next_loc

    #     return predictions

    def run_all_baselines(self, output_dir="baseline_results"):
        """Run all baseline methods and save results"""
        Path(output_dir).mkdir(exist_ok=True)

        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)

        print("=" * 60)
        print("RUNNING ALL BASELINE METHODS")
        print("=" * 60)

        results = {}

        # Global methods
        global_mean_loc = self.global_mean()
        global_mode_loc = self.global_mode()

        results["global_mean"] = self.predict_global_method(
            global_mean_loc, "Global Mean"
        )
        results["global_mode"] = self.predict_global_method(
            global_mode_loc, "Global Mode"
        )

        # Per-user methods
        results["per_user_mean"] = self.predict_per_user_mean()
        results["per_user_mode"] = self.predict_per_user_mode()

        # N-gram methods
        results["unigram"] = self.predict_unigram_model()
        # results["bigram"] = self.predict_bigram_model()
        # results["bigram_top_p"] = self.predict_bigram_model(top_p=0.7)

        # Save all results
        for method_name, predictions in results.items():
            output_file = Path(output_dir) / f"{method_name}_predictions.csv"
            predictions.to_csv(output_file, index=False)
            print(f"Saved {method_name} predictions to {output_file}")

        print("\n" + "=" * 60)
        print("ALL BASELINE METHODS COMPLETED")
        print("=" * 60)

        return results


def main():
    # File paths
    input_file = "city_C_processed.csv"

    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found!")
        print("Please run the data processing script first.")
        return

    baseline = BaselineMethods(input_file)
    results = baseline.run_all_baselines()

    print(f"\nGenerated {len(results)} baseline prediction sets:")
    for method_name in results.keys():
        print(f"  - {method_name}")


if __name__ == "__main__":
    main()
