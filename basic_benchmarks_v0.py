import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class MobilityPredictor:
    """Base class for mobility prediction models"""
    
    def __init__(self, name):
        self.name = name
        self.trained = False
    
    def train(self, train_data):
        """Train the model on training data"""
        raise NotImplementedError
    
    def predict(self, user_id, timeline_idx):
        """Predict location for user at given timeline index"""
        raise NotImplementedError


class GlobalMeanPredictor(MobilityPredictor):
    """Global Mean baseline - predict most common location globally"""
    
    def __init__(self):
        super().__init__("Global Mean")
        self.global_mean_location = None
    
    def train(self, train_data):
        """Find globally most common location"""
        all_locations = []
        
        for _, row in train_data.iterrows():
            for col in train_data.columns[1:]:  # Skip 'uid'
                if col.startswith('t_'):
                    timeline_idx = int(col[2:])
                    if timeline_idx < 60 * 48:  # Training period only
                        location = row[col]
                        if pd.notna(location):
                            all_locations.append(location)
        
        # Find most common location
        location_counts = Counter(all_locations)
        self.global_mean_location = location_counts.most_common(1)[0][0]
        self.trained = True
        
        print(f"Global mean location: {self.global_mean_location} "
              f"(appears {location_counts.most_common(1)[0][1]} times)")
    
    def predict(self, user_id, timeline_idx):
        return self.global_mean_location


class GlobalModePredictor(MobilityPredictor):
    """Global Mode baseline - same as Global Mean for this case"""
    
    def __init__(self):
        super().__init__("Global Mode")
        self.global_mode_location = None
    
    def train(self, train_data):
        # Same as Global Mean
        all_locations = []
        
        for _, row in train_data.iterrows():
            for col in train_data.columns[1:]:
                if col.startswith('t_'):
                    timeline_idx = int(col[2:])
                    if timeline_idx < 60 * 48:
                        location = row[col]
                        if pd.notna(location):
                            all_locations.append(location)
        
        location_counts = Counter(all_locations)
        self.global_mode_location = location_counts.most_common(1)[0][0]
        self.trained = True
    
    def predict(self, user_id, timeline_idx):
        return self.global_mode_location


class PerUserMeanPredictor(MobilityPredictor):
    """Per-User Mean - predict most common location for each user"""
    
    def __init__(self):
        super().__init__("Per-User Mean")
        self.user_mean_locations = {}
    
    def train(self, train_data):
        for _, row in train_data.iterrows():
            user_id = row['uid']
            user_locations = []
            
            for col in train_data.columns[1:]:
                if col.startswith('t_'):
                    timeline_idx = int(col[2:])
                    if timeline_idx < 60 * 48:  # Training period
                        location = row[col]
                        if pd.notna(location):
                            user_locations.append(location)
            
            if user_locations:
                location_counts = Counter(user_locations)
                self.user_mean_locations[user_id] = location_counts.most_common(1)[0][0]
            else:
                self.user_mean_locations[user_id] = "0,0"  # Default
        
        self.trained = True
        print(f"Trained per-user means for {len(self.user_mean_locations)} users")
    
    def predict(self, user_id, timeline_idx):
        return self.user_mean_locations.get(user_id, "0,0")


class PerUserModePredictor(MobilityPredictor):
    """Per-User Mode - same as Per-User Mean for this case"""
    
    def __init__(self):
        super().__init__("Per-User Mode")
        self.user_mode_locations = {}
    
    def train(self, train_data):
        for _, row in train_data.iterrows():
            user_id = row['uid']
            user_locations = []
            
            for col in train_data.columns[1:]:
                if col.startswith('t_'):
                    timeline_idx = int(col[2:])
                    if timeline_idx < 60 * 48:
                        location = row[col]
                        if pd.notna(location):
                            user_locations.append(location)
            
            if user_locations:
                location_counts = Counter(user_locations)
                self.user_mode_locations[user_id] = location_counts.most_common(1)[0][0]
            else:
                self.user_mode_locations[user_id] = "0,0"
        
        self.trained = True
    
    def predict(self, user_id, timeline_idx):
        return self.user_mode_locations.get(user_id, "0,0")


class UnigramPredictor(MobilityPredictor):
    """Unigram model - location frequency per user"""
    
    def __init__(self):
        super().__init__("Unigram")
        self.user_location_probs = {}
    
    def train(self, train_data):
        for _, row in train_data.iterrows():
            user_id = row['uid']
            user_locations = []
            
            for col in train_data.columns[1:]:
                if col.startswith('t_'):
                    timeline_idx = int(col[2:])
                    if timeline_idx < 60 * 48:
                        location = row[col]
                        if pd.notna(location):
                            user_locations.append(location)
            
            if user_locations:
                location_counts = Counter(user_locations)
                total_count = sum(location_counts.values())
                # Convert to probabilities
                location_probs = {loc: count/total_count 
                                for loc, count in location_counts.items()}
                self.user_location_probs[user_id] = location_probs
            else:
                self.user_location_probs[user_id] = {"0,0": 1.0}
        
        self.trained = True
    
    def predict(self, user_id, timeline_idx):
        if user_id in self.user_location_probs:
            # Return most probable location
            probs = self.user_location_probs[user_id]
            return max(probs.items(), key=lambda x: x[1])[0]
        return "0,0"


class BigramPredictor(MobilityPredictor):
    """Bigram model - predict based on previous location"""
    
    def __init__(self):
        super().__init__("Bigram")
        self.user_bigram_probs = {}
    
    def train(self, train_data):
        for _, row in train_data.iterrows():
            user_id = row['uid']
            user_sequence = []
            
            # Get user's location sequence in training period
            for i in range(60 * 48):  # Training period
                col = f't_{i}'
                location = row[col]
                if pd.notna(location):
                    user_sequence.append(location)
            
            if len(user_sequence) >= 2:
                # Build bigram counts
                bigram_counts = defaultdict(lambda: defaultdict(int))
                
                for i in range(len(user_sequence) - 1):
                    prev_loc = user_sequence[i]
                    next_loc = user_sequence[i + 1]
                    bigram_counts[prev_loc][next_loc] += 1
                
                # Convert to probabilities
                bigram_probs = {}
                for prev_loc, next_counts in bigram_counts.items():
                    total = sum(next_counts.values())
                    bigram_probs[prev_loc] = {
                        next_loc: count/total 
                        for next_loc, count in next_counts.items()
                    }
                
                self.user_bigram_probs[user_id] = bigram_probs
            else:
                self.user_bigram_probs[user_id] = {}
        
        self.trained = True
    
    def predict(self, user_id, timeline_idx, prev_location=None):
        if (user_id in self.user_bigram_probs and 
            prev_location and 
            prev_location in self.user_bigram_probs[user_id]):
            
            next_probs = self.user_bigram_probs[user_id][prev_location]
            return max(next_probs.items(), key=lambda x: x[1])[0]
        
        # Fallback to unigram
        return "0,0"


def calculate_simple_geo_bleu(predicted_seq, actual_seq, n=4):
    """
    Simple approximation of GEO-BLEU metric
    """
    if len(predicted_seq) != len(actual_seq):
        return 0.0
    
    # Calculate n-gram matches
    scores = []
    
    for gram_size in range(1, min(n + 1, len(actual_seq) + 1)):
        # Get n-grams from both sequences
        pred_ngrams = []
        actual_ngrams = []
        
        for i in range(len(predicted_seq) - gram_size + 1):
            pred_ngrams.append(tuple(predicted_seq[i:i + gram_size]))
            actual_ngrams.append(tuple(actual_seq[i:i + gram_size]))
        
        if not actual_ngrams:
            continue
            
        # Count matches
        pred_counter = Counter(pred_ngrams)
        actual_counter = Counter(actual_ngrams)
        
        matches = 0
        for ngram, count in pred_counter.items():
            matches += min(count, actual_counter.get(ngram, 0))
        
        precision = matches / len(pred_ngrams) if pred_ngrams else 0.0
        scores.append(precision)
    
    # Geometric mean of precisions (simplified BLEU)
    if scores:
        geo_mean = np.exp(np.mean(np.log(np.array(scores) + 1e-10)))
        return geo_mean
    return 0.0


def calculate_dtw_distance(seq1, seq2):
    """
    Calculate Dynamic Time Warping distance between two location sequences
    """
    n, m = len(seq1), len(seq2)
    
    # Parse locations to coordinates
    def parse_location(loc_str):
        try:
            x, y = map(int, loc_str.split(','))
            return (x, y)
        except:
            return (0, 0)
    
    coords1 = [parse_location(loc) for loc in seq1]
    coords2 = [parse_location(loc) for loc in seq2]
    
    # DTW matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Euclidean distance between coordinates
            x1, y1 = coords1[i-1]
            x2, y2 = coords2[j-1]
            cost = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],      # insertion
                dtw_matrix[i, j-1],      # deletion
                dtw_matrix[i-1, j-1]     # match
            )
    
    return dtw_matrix[n, m]


def evaluate_model(model, test_data, prediction_period_start=60*48):
    """
    Evaluate model on test data
    """
    print(f"\nEvaluating {model.name}...")
    
    geo_bleu_scores = []
    dtw_scores = []
    
    for _, row in test_data.iterrows():
        user_id = row['uid']
        
        # Get actual sequence for prediction period (days 61-75)
        actual_sequence = []
        predicted_sequence = []
        
        for i in range(prediction_period_start, 75 * 48):
            col = f't_{i}'
            actual_location = row[col]
            
            if pd.notna(actual_location):
                actual_sequence.append(actual_location)
                
                # Get prediction
                if isinstance(model, BigramPredictor):
                    # For bigram, need previous location
                    prev_col = f't_{i-1}' if i > 0 else None
                    prev_location = row[prev_col] if prev_col and pd.notna(row[prev_col]) else None
                    predicted_location = model.predict(user_id, i, prev_location)
                else:
                    predicted_location = model.predict(user_id, i)
                
                predicted_sequence.append(predicted_location)
        
        if len(actual_sequence) >= 10:  # Minimum sequence length for meaningful evaluation
            # Calculate GEO-BLEU
            geo_bleu = calculate_simple_geo_bleu(predicted_sequence, actual_sequence)
            geo_bleu_scores.append(geo_bleu)
            
            # Calculate DTW
            dtw_dist = calculate_dtw_distance(predicted_sequence, actual_sequence)
            dtw_scores.append(dtw_dist)
    
    avg_geo_bleu = np.mean(geo_bleu_scores) if geo_bleu_scores else 0.0
    avg_dtw = np.mean(dtw_scores) if dtw_scores else float('inf')
    
    print(f"{model.name} Results:")
    print(f"  Average GEO-BLEU: {avg_geo_bleu:.4f}")
    print(f"  Average DTW: {avg_dtw:.2f}")
    print(f"  Evaluated on {len(geo_bleu_scores)} users")
    
    return {
        'model': model.name,
        'geo_bleu': avg_geo_bleu,
        'dtw': avg_dtw,
        'n_users': len(geo_bleu_scores)
    }


def run_baseline_comparison(data_file):
    """
    Run all baseline models and compare results
    """
    print("Loading data...")
    df = pd.read_csv(data_file)
    
    print(f"Data shape: {df.shape}")
    print(f"Users: {len(df)}")
    
    # Initialize models
    models = [
        GlobalMeanPredictor(),
        GlobalModePredictor(),
        PerUserMeanPredictor(),
        PerUserModePredictor(),
        UnigramPredictor(),
        BigramPredictor()
    ]
    
    # Train all models
    print("\n=== Training Models ===")
    for model in models:
        print(f"\nTraining {model.name}...")
        model.train(df)
    
    # Evaluate all models
    print("\n=== Evaluating Models ===")
    results = []
    
    for model in models:
        result = evaluate_model(model, df)
        results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    print("\n=== Final Results ===")
    print(results_df.to_string(index=False))
    
    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # GEO-BLEU comparison
    ax1.bar(results_df['model'], results_df['geo_bleu'])
    ax1.set_title('GEO-BLEU Scores (Higher is Better)')
    ax1.set_ylabel('GEO-BLEU Score')
    ax1.tick_params(axis='x', rotation=45)
    
    # DTW comparison (log scale because DTW can be large)
    ax2.bar(results_df['model'], results_df['dtw'])
    ax2.set_title('DTW Distance (Lower is Better)')
    ax2.set_ylabel('DTW Distance')
    ax2.set_yscale('log')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('baseline_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved: baseline_comparison.png")
    plt.show()
    
    # Save results
    results_df.to_csv('baseline_results.csv', index=False)
    print("Saved: baseline_results.csv")
    
    return results_df, models


def analyze_prediction_patterns(models, test_data, sample_users=3):
    """
    Analyze prediction patterns for sample users
    """
    print(f"\n=== Analyzing Prediction Patterns for {sample_users} Users ===")
    
    sample_user_ids = test_data['uid'].head(sample_users).tolist()
    
    fig, axes = plt.subplots(sample_users, len(models), figsize=(4*len(models), 4*sample_users))
    if sample_users == 1:
        axes = axes.reshape(1, -1)
    
    for user_idx, user_id in enumerate(sample_user_ids):
        user_data = test_data[test_data['uid'] == user_id].iloc[0]
        
        # Get actual trajectory for prediction period
        actual_trajectory = []
        for i in range(60*48, 75*48):
            col = f't_{i}'
            location = user_data[col]
            if pd.notna(location):
                try:
                    x, y = map(int, location.split(','))
                    actual_trajectory.append((x, y))
                except:
                    continue
        
        for model_idx, model in enumerate(models):
            ax = axes[user_idx, model_idx]
            
            # Get predictions
            predicted_trajectory = []
            for i in range(60*48, 75*48):
                if isinstance(model, BigramPredictor):
                    prev_col = f't_{i-1}' if i > 0 else None
                    prev_location = user_data[prev_col] if prev_col and pd.notna(user_data[prev_col]) else None
                    prediction = model.predict(user_id, i, prev_location)
                else:
                    prediction = model.predict(user_id, i)
                
                try:
                    x, y = map(int, prediction.split(','))
                    predicted_trajectory.append((x, y))
                except:
                    predicted_trajectory.append((0, 0))
            
            # Plot trajectories
            if actual_trajectory:
                actual_x, actual_y = zip(*actual_trajectory)
                ax.plot(actual_x, actual_y, 'b-', label='Actual', linewidth=2, alpha=0.7)
                ax.scatter(actual_x, actual_y, c='blue', s=20, alpha=0.7)
            
            if predicted_trajectory:
                pred_x, pred_y = zip(*predicted_trajectory)
                ax.plot(pred_x, pred_y, 'r--', label='Predicted', linewidth=2, alpha=0.7)
                ax.scatter(pred_x, pred_y, c='red', s=20, alpha=0.7, marker='x')
            
            ax.set_title(f'User {user_id} - {model.name}')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_patterns.png', dpi=300, bbox_inches='tight')
    print("Saved: prediction_patterns.png")
    plt.show()


def save_models(models, filename='trained_models.pkl'):
    """
    Save trained models to file
    """
    model_data = {}
    for model in models:
        if hasattr(model, '__dict__'):
            model_data[model.name] = model.__dict__
    
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Saved trained models to {filename}")


def load_models(filename='trained_models.pkl'):
    """
    Load trained models from file
    """
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"Loaded models from {filename}")
    return model_data


if __name__ == "__main__":
    # Run the baseline comparison
    data_file = "city_C_forward_filled.csv"
    
    if not Path(data_file).exists():
        print(f"Error: {data_file} not found!")
        print("Please run v6.py first to generate the forward-filled data.")
    else:
        print("=== Running Baseline Model Comparison ===")
        
        # Run comparison
        results_df, trained_models = run_baseline_comparison(data_file)
        
        # Analyze prediction patterns
        df = pd.read_csv(data_file)
        analyze_prediction_patterns(trained_models, df, sample_users=3)
        
        # Save trained models
        save_models(trained_models)
        
        print("\n=== Baseline Analysis Complete ===")
        print("Generated files:")
        print("- baseline_comparison.png")
        print("- prediction_patterns.png") 
        print("- baseline_results.csv")
        print("- trained_models.pkl")
